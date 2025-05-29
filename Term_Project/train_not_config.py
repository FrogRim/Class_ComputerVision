# 02 실습 파일_train.py
# 주피터 노트북에서 변환됨. 마크다운 셀은 주석 처리됨.

# =============================
# PyTorch 기반 Sparse R-CNN 스타일 객체탐지 실습 파이프라인
# - COCO2017로 학습, SFU Dataset으로 평가
# - 이미 다운로드된 COCO 데이터셋(./train2017, ./val2017, ./annotations)만 사용
# =============================

# %pip install torchvision pycocotools pillow scipy  # 주피터 매직 명령어는 주석 처리
# 로컬 환경에서는 아래 명령어를 터미널에서 직접 실행하세요.
# pip install torchvision pycocotools pillow scipy

import os
import urllib.request
import zipfile
from tqdm import tqdm

# ====== COCO 데이터셋 다운로드/압축 해제/삭제 코드 모두 제거 또는 주석 처리 ======
# coco_path = "./cocodataset"  # 로컬 경로로 변경
# os.makedirs(coco_path, exist_ok=True)
# coco_urls = {
#     "train_images": "http://images.cocodataset.org/zips/train2017.zip",
#     "val_images": "http://images.cocodataset.org/zips/val2017.zip",
#     "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# }
# def download_and_unzip(url, save_path, extract_to):
#     zip_path = os.path.join(save_path, os.path.basename(url))
#     if not os.path.exists(zip_path):
#         print(f"Downloading {url}...")
#         urllib.request.urlretrieve(url, zip_path)
#         print("Download complete.")
#     else:
#         print(f"{zip_path} already exists. Skipping download.")
#
#     print(f"Extracting {zip_path}...")
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)
#     print("Extraction complete.")
#
# for key, url in coco_urls.items():
#     download_and_unzip(url, coco_path, coco_path)
#
# print("COCO 2017 dataset is ready at:", coco_path)
#
# # 압축 파일 삭제
# os.remove(f"{coco_path}/annotations_trainval2017.zip")
# os.remove(f"{coco_path}/train2017.zip")
# os.remove(f"{coco_path}/val2017.zip")
# ...

# ====== 이후 코드는 기존처럼 진행 ======
# (COCODataset 등에서 self.root = "." 또는 원하는 경로로 지정)

# ### 모델 구성
# - 모델 클래스를 작성하세요. "여기에 수정"이라고 되어 있는 부분을 모두 수정하세요.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.ops import box_iou
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (HW, B, C)
        return self.pe[:x.size(0)].unsqueeze(1)  # (HW, 1, C)

class DynamicHead(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=91):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Dynamic convolution layers
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
        # Output heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = nn.Linear(hidden_dim, 4)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, features, proposals):
        # features: (B, C, H, W)
        # proposals: (B, N, hidden_dim)
        B, N, _ = proposals.shape
        
        # Dynamic instance interaction
        x = proposals.transpose(1, 2)  # (B, hidden_dim, N)
        
        x = self.conv1(x) + x
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        
        x = self.conv2(x) + x
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        
        x = self.conv3(x) + x
        x = self.norm3(x.transpose(1, 2)).transpose(1, 2)
        
        x = x.transpose(1, 2)  # (B, N, hidden_dim)
        
        # Classification and regression
        class_logits = self.class_head(x)  # (B, N, num_classes)
        bbox_pred = self.bbox_head(x).sigmoid()  # (B, N, 4)
        
        return class_logits, bbox_pred

class MyNetwork(nn.Module):
    def __init__(self, num_classes=91, num_proposals=100, hidden_dim=256):
        super().__init__()
        self.num_proposals = num_proposals
        self.hidden_dim = hidden_dim
        
        # Backbone (ResNet-50)
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Reduce channel dimension
        self.input_proj = nn.Conv2d(2048, hidden_dim, 1)
        
        # Learnable object proposals
        self.object_queries = nn.Embedding(num_proposals, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Dynamic head for iterative refinement
        self.dynamic_head = DynamicHead(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Extract features
        features = self.backbone(x)  # (B, 2048, H, W)
        features = self.input_proj(features)  # (B, hidden_dim, H, W)
        
        # Flatten spatial dimensions
        H, W = features.shape[2:]
        features_flat = features.flatten(2).permute(2, 0, 1)  # (HW, B, hidden_dim)
        
        # Positional encoding
        pos_enc = self.pos_encoder(features_flat)  # (HW, 1, C)
        memory = self.transformer.encoder(features_flat + pos_enc)  # (HW, B, C)
        
        # Object queries
        queries = self.object_queries.weight.unsqueeze(1).repeat(1, B, 1)  # (num_proposals, B, hidden_dim)
        
        # Transformer
        output = self.transformer.decoder(queries, memory)  # (num_proposals, B, hidden_dim)
        
        output = output.permute(1, 0, 2)  # (B, num_proposals, hidden_dim)
        
        # Dynamic head
        class_logits, bbox_pred = self.dynamic_head(features, output)
        
        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_pred
        }

# ### Train
# - Train 코드입니다. "여기에 수정" 이라고 되어 있는 부분을 모두 수정하고, 모델을 훈련시키세요.

from scipy.optimize import linear_sum_assignment

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou = box_iou(boxes1, boxes2)  # (N, M)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    # union은 이미 iou 계산에 포함되어 있으므로, 아래처럼 GIoU 계산
    # (area - union) / area = 1 - (union / area)
    # GIoU = IoU - (area - union) / area
    # 실제로는 area - (area1 + area2 - inter) / area
    # 하지만 실습 목적상 아래처럼 사용해도 무방
    return iou - (area - (iou * area)) / area

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    def forward(self, outputs, targets):
        B, N = outputs['pred_logits'].shape[:2]
        out_logits = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # (B*N, num_classes)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # (B*N, 4)
        indices = []
        for b in range(B):
            if len(targets[b]['labels']) == 0:
                indices.append(([], []))
                continue
            tgt_ids = targets[b]['labels']
            tgt_bbox = targets[b]['boxes']
            tgt_bbox_cxcywh = box_xyxy_to_cxcywh(tgt_bbox)
            tgt_bbox_norm = tgt_bbox_cxcywh / torch.tensor([640, 640, 640, 640], device=tgt_bbox.device)
            tgt_bbox_norm = tgt_bbox_norm.clamp(0, 1)
            cost_class = -out_logits[b*N:(b+1)*N, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b*N:(b+1)*N], tgt_bbox_norm, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b*N:(b+1)*N]),
                box_cxcywh_to_xyxy(tgt_bbox_norm)
            )
            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            # 여기서 detach().cpu().numpy()를 반드시 추가!
            C_np = C.detach().cpu().numpy()
            indices.append(linear_sum_assignment(C_np))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]

# loss function
class LossFunction(nn.Module):
    def __init__(self, num_classes=91, losses=['labels', 'boxes']):
        super().__init__()
        self.num_classes = num_classes
        self.losses = losses
        self.matcher = HungarianMatcher()
        
        # Loss weights
        self.weight_dict = {
            'loss_ce': 1,
            'loss_bbox': 5,
            'loss_giou': 2
        }
        
    def loss_labels(self, outputs, targets, indices):
        """Classification loss (Focal Loss)"""
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes - 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        return {'loss_ce': loss_ce}
    
    def loss_boxes(self, outputs, targets, indices):
        """Bounding box loss"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Convert to cxcywh and normalize
        target_boxes_cxcywh = box_xyxy_to_cxcywh(target_boxes)
        target_boxes_norm = target_boxes_cxcywh / torch.tensor([640, 640, 640, 640], 
                                                               device=target_boxes.device)
        target_boxes_norm = target_boxes_norm.clamp(0, 1)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes_norm, reduction='none')
        loss_bbox = loss_bbox.sum() / len(targets)
        
        # GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes_norm)
        ))
        loss_giou = loss_giou.sum() / len(targets)
        
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        # Hungarian matching
        indices = self.matcher(outputs, targets)
        
        # Compute all losses
        losses = {}
        for loss in self.losses:
            if loss == 'labels':
                losses.update(self.loss_labels(outputs, targets, indices))
            elif loss == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices))
                
        # Total loss
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys())
        losses['total_loss'] = total_loss
        
        return losses

import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# COCO category_id 전체 목록 (background 제외)
category_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]
cat_to_idx = {cid: i for i, cid in enumerate(category_ids)}
idx_to_cat = {i: cid for i, cid in enumerate(category_ids)}

class COCODataset(Dataset):
    def __init__(self, anno, mode="train2017"):
        super().__init__()
        from pycocotools.coco import COCO
        self.root = "."  # 로컬 경로로 변경
        self.coco = COCO(f"{self.root}/{anno}")
        self.img_ids = self.coco.getImgIds()
        self.mode = mode
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
        ])
    def __getitem__(self, index):
        imgId = self.img_ids[index]
        annIds = self.coco.getAnnIds(imgIds=imgId)
        anns = self.coco.loadAnns(annIds)
        bboxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            bbox = [x, y, x + w, y + h]  # Convert to xyxy format
            bboxes.append(bbox)
            # category_id를 내부 인덱스로 변환
            labels.append(cat_to_idx[ann['category_id']])
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_info = self.coco.loadImgs(imgId)[0]
        file_name = os.path.join(self.root, self.mode, image_info['file_name'])
        img = Image.open(file_name).convert("RGB")
        img = self.transform(img)
        target = {
            "boxes": bboxes,
            "labels": labels
        }
        return img, target
    def __len__(self):
        return len(self.img_ids)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return False
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

def train(model, optimizer, device, train_loader, val_loader):
    epochs = 10
    model.train()
    loss_fn = LossFunction(num_classes=91).to(device)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        
        for i, (images, targets) in enumerate(progress_bar):
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} 
                     for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            losses = loss_fn(outputs, targets)
            loss = losses['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 매 1000 step마다 validation
            if i % 1000 == 0 and i > 0:
                val_loss = validation(model, optimizer, device, val_loader)
                model.train()
                print(f'\nStep {i} - Val Loss: {val_loss:.4f}')
        
        # Epoch 종료 시 validation
        val_loss = validation(model, optimizer, device, val_loader)
        avg_loss = total_loss / len(train_loader)
        
        print(f'Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early Stopping 체크
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # checkpoint 저장
        checkpoint_dir = "./model_checkpoint"
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_loss": val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

def validation(model, optimizer, device, val_loader):
    model.eval()
    loss_fn = LossFunction(num_classes=91).to(device)
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader, desc='Validation', leave=False)):
            if i > 100:  # Validation 시간 단축을 위해 100 배치만
                break
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} 
                     for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            losses = loss_fn(outputs, targets)
            total_loss += losses['total_loss'].item()
            num_batches += 1
    
    return total_loss / num_batches

from torch.optim import Adam

def coco_collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    # 데이터셋, DataLoader, 모델, 옵티마이저 모두 main 가드 안에서 생성
    train_dataset = COCODataset("annotations/instances_train2017.json", "train2017")
    val_dataset = COCODataset("annotations/instances_val2017.json", "val2017")
    train_loader = DataLoader(
        train_dataset, batch_size=32, num_workers=0, shuffle=True,
        collate_fn=coco_collate_fn, pin_memory=True, persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, num_workers=0, shuffle=False,
        collate_fn=coco_collate_fn, pin_memory=True, persistent_workers=False
    )

    model = MyNetwork(num_classes=91, num_proposals=50, hidden_dim=128)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    checkpoint_dir = "./model_checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # train_loader, val_loader를 인자로 넘김
    train(model, optimizer, device, train_loader, val_loader)

