# 02 실습 파일_train.py


# ### 실습 준비
# - 데이터셋 설치

import os
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import rpn
from torchvision.models.detection import roi_heads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF
from tqdm import tqdm
from collections import OrderedDict, defaultdict, deque
import warnings
import random
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LinearLR, SequentialLR
import time
import datetime

warnings.filterwarnings('ignore')

# CUDA 설정
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# 타겟 클래스 정의
TARGET_COCO_CLASSES = [1, 2, 3, 4, 6, 8]  # person, bicycle, car, motorcycle, bus, truck
CLASS_NAMES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
NUM_CLASSES = len(TARGET_COCO_CLASSES) + 1  # 6개 객체 클래스 + 1개 배경 클래스 = 7

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size, dropout_p=0.5):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        return x

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

class MyNetwork(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, backbone_name='resnet50',
                 min_size=800, max_size=1333,
                 box_detections_per_img=100,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5):
        super(MyNetwork, self).__init__()

        self.num_classes = num_classes

        # Backbone: ResNet50 with FPN
        extra_blocks = LastLevelP6P7(256, 256)
        self.backbone = resnet_fpn_backbone(
            backbone_name,
            pretrained=True,
            trainable_layers=5,
            returned_layers=[1, 2, 3, 4],
            extra_blocks=extra_blocks
        )

        out_channels = self.backbone.out_channels

        # RPN
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 6

        rpn_anchor_generator = rpn.AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        rpn_head = rpn.RPNHead(
            out_channels,
            rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = rpn.RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n,
            rpn_nms_thresh
        )

        # ROI Pooling
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 2048

        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size,
            dropout_p=0.5
        )

        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes
        )

        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None

        self.roi_heads = roi_heads.RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img
        )

        self.transform = GeneralizedRCNNTransform(
            min_size, max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, losses_dict):
        rpn_cls_loss = losses_dict.get('loss_objectness', 0) * 1.0
        rpn_reg_loss = losses_dict.get('loss_rpn_box_reg', 0) * 1.0
        rcnn_cls_loss = losses_dict.get('loss_classifier', 0) * 1.5
        rcnn_reg_loss = losses_dict.get('loss_box_reg', 0) * 1.0
        total_loss = rpn_cls_loss + rpn_reg_loss + rcnn_cls_loss + rcnn_reg_loss
        return total_loss, {
            'rpn_cls': rpn_cls_loss,
            'rpn_reg': rpn_reg_loss,
            'rcnn_cls': rcnn_cls_loss,
            'rcnn_reg': rcnn_reg_loss
        }

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            if len(bbox) > 0:
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target

class RandomResizedCrop:
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, target):
        scale = random.uniform(self.min_scale, self.max_scale)
        h, w = image.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        image = TF.resize(image, [new_h, new_w])
        boxes = target['boxes'] * scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
        target['boxes'] = boxes
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class COCODataset(Dataset):
    def __init__(self, anno, mode="train2017", transforms=None, max_samples=10000):
        super().__init__()
        from pycocotools.coco import COCO
        self.root = "./cocodataset"
        self.coco = COCO(f"{self.root}/{anno}")

        img_ids = self.coco.getImgIds()
        filtered_img_ids = []
        for img_id in img_ids:
            annIds = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(annIds)
            if any(ann['category_id'] in TARGET_COCO_CLASSES for ann in anns):
                filtered_img_ids.append(img_id)

        self.img_ids = filtered_img_ids[:max_samples]
        self.mode = mode
        self.transforms = transforms

        sorted_classes = sorted(TARGET_COCO_CLASSES)
        self.cat2label = {cat_id: idx for idx, cat_id in enumerate(sorted_classes)}
        self.label2cat = {idx: cat_id for cat_id, idx in self.cat2label.items()}

        self.base_transform = T.Compose([T.ToTensor()])

    def __getitem__(self, index):
        imgId = self.img_ids[index]
        annIds = self.coco.getAnnIds(imgIds=imgId)
        anns = self.coco.loadAnns(annIds)
        image_info = self.coco.loadImgs(imgId)[0]
        file_name = os.path.join(self.root, self.mode, image_info['file_name'])
        img = Image.open(file_name).convert("RGB")

        bboxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            coco_class_id = ann['category_id']
            if coco_class_id not in TARGET_COCO_CLASSES:
                continue

            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                bbox = [x, y, x + w, y + h]
                bboxes.append(bbox)
                internal_label = self.cat2label[coco_class_id]
                labels.append(internal_label)
                areas.append(ann['area'])
                iscrowd.append(ann.get('iscrowd', 0))

        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([imgId])
        target = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        img = self.base_transform(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_ids)

def collate_fn(batch):
    return tuple(zip(*batch))

def download_and_unzip(url, save_path, extract_to):
    zip_path = os.path.join(save_path, os.path.basename(url))
    if not os.path.exists(zip_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print(f"{zip_path} already exists. Skipping download.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

# COCO 데이터셋 다운로드
coco_path = "./cocodataset"
os.makedirs(coco_path, exist_ok=True)

coco_urls = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

for key, url in coco_urls.items():
    download_and_unzip(url, coco_path, coco_path)

print("COCO 2017 dataset is ready at:", coco_path)

# 압축 파일 삭제
os.remove(f"{coco_path}/annotations_trainval2017.zip")
os.remove(f"{coco_path}/train2017.zip")
os.remove(f"{coco_path}/val2017.zip")

# 데이터 증강 설정
train_transforms = Compose([
    RandomHorizontalFlip(0.5),
    RandomResizedCrop(0.8, 1.2),
])

# 데이터셋 로드
train_dataset = COCODataset(
    "annotations/instances_train2017.json",
    "train2017",
    transforms=train_transforms,
    max_samples=20000
)
val_dataset = COCODataset(
    "annotations/instances_val2017.json",
    "val2017",
    max_samples=2000
)

# 데이터로더 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

# 모델 생성
model = MyNetwork(
    num_classes=NUM_CLASSES,
    backbone_name='resnet50',
    min_size=800,
    max_size=1333,
    box_detections_per_img=300,
    box_score_thresh=0.01,
    box_nms_thresh=0.3
)
model.to(device)

# 옵티마이저 설정
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001,
    weight_decay=0.0005,
    betas=(0.9, 0.999)
)

# 학습률 스케줄러
warmup_epochs = 3
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=40 - warmup_epochs, eta_min=0.001 * 0.01)
lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

# 손실 함수
loss_fn = LossFunction()

# 체크포인트 디렉토리 생성
checkpoint_dir = "./model_checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

def train(model, optimizer, device):
    epochs = 40
    model.train()
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        epoch_loss = 0
        num_batches = 0

        for images, targets in tqdm(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            valid_images = []
            valid_targets = []
            for img, target in zip(images, targets):
                if len(target["boxes"]) > 0:
                    valid_images.append(img)
                    valid_targets.append(target)

            if len(valid_images) == 0:
                continue

            loss_dict = model(valid_images, valid_targets)
            losses, _ = loss_fn(loss_dict)

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # 검증
        if (epoch + 1) % 2 == 0:
            val_loss = validation(model, val_loader, device)
            val_losses.append(val_loss)

            print(f"Validation Loss: {val_loss:.4f}")

            # 체크포인트 저장
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "val_loss": val_loss,
                "train_loss": avg_train_loss,
                "cat2label": train_dataset.cat2label,
                "label2cat": train_dataset.label2cat,
                "target_classes": TARGET_COCO_CLASSES,
                "num_classes": NUM_CLASSES
            }

            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(checkpoint, best_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        lr_scheduler.step()

    return train_losses, val_losses

def validation(model, val_loader, device):
    model.eval()
    val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            model.train()  # loss 계산을 위해 training mode로 설정
            loss_dict = model(images, targets)
            losses, _ = loss_fn(loss_dict)
            val_loss += losses.item()
            num_batches += 1

    model.eval()  # 다시 evaluation mode로 변경
    return val_loss / num_batches if num_batches > 0 else 0

# 학습 실행
train_losses, val_losses = train(model, optimizer, device)

# 학습 곡선 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

if val_losses:
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss', marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Validation Check #')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('./training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("Training completed!")
print(f"Best validation loss: {min(val_losses) if val_losses else 'N/A'}") 