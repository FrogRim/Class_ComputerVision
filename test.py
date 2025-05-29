import os
import zipfile

# Google Drive에서 파일 다운로드 (gdown 사용)
# gdrive_id = "1-tPJHwYIdmnQsk_o_tcynZIH86cfcFzn"
# os.system(f"gdown --id {gdrive_id} -O SFU.zip")

# with zipfile.ZipFile("./SFU.zip", 'r') as zip_ref:
#     zip_ref.extractall("./SFU")

# ### 모델
# - 모델 부분을 작성.
# - ./model_checkpoint 폴더를 만들고, 테스트하려는 체크포인트를 넣으세요.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
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

import json
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# COCO category_id 전체 목록 (background 제외)
category_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]
cat_to_idx = {cid: i for i, cid in enumerate(category_ids)}
idx_to_cat = {i: cid for i, cid in enumerate(category_ids)}

class TestDataset(Dataset):
    def __init__(self, root_dir, dataset_names, transform=None):
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.transform = transform
        self.img_info_list = []  # 모든 데이터셋의 이미지 정보를 저장할 리스트
        self.ann_dict = {}  # 이미지 ID별 어노테이션 저장

        for dataset_name in self.dataset_names:
            dataset_path = os.path.join(root_dir, dataset_name)
            ann_file = os.path.join(dataset_path, f"annotations/{dataset_name}.json")
            img_dir = os.path.join(dataset_path, "images")

            # JSON 파일 로드
            with open(ann_file, "r") as f:
                annotations = json.load(f)

            # 이미지 정보 저장
            for img_info in annotations["images"]:
                img_info["dataset_path"] = img_dir  # 이미지 경로 추가
                self.img_info_list.append(img_info)

            # 어노테이션 저장
            for ann in annotations["annotations"]:
                img_id = ann["image_id"]
                if img_id not in self.ann_dict:
                    self.ann_dict[img_id] = []
                # category_id를 내부 인덱스로 변환
                ann = ann.copy()
                ann["category_id"] = cat_to_idx[ann["category_id"]]
                self.ann_dict[img_id].append(ann)

        # ann_file 속성 추가 (단일 데이터셋 기준)
        if len(self.dataset_names) == 1:
            self.ann_file = ann_file

    def __len__(self):
        return len(self.img_info_list)

    def __getitem__(self, index):
        img_info = self.img_info_list[index]
        img_id = img_info["id"]
        img_path = os.path.join(img_info["dataset_path"], img_info["file_name"])

        # 이미지 로드
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        # 어노테이션 정보 가져오기
        annotations = self.ann_dict.get(img_id, [])

        # 바운딩 박스 및 레이블 변환
        bboxes = []
        labels = []
        for ann in annotations:
            bbox = ann["bbox"]  # [x, y, w, h]
            category_id = ann["category_id"]
            bboxes.append(torch.tensor(bbox, dtype=torch.float32))
            labels.append(torch.tensor(category_id, dtype=torch.int64))

        bboxes = torch.stack(bboxes) if bboxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        # 이미지 변환 적용
        if self.transform:
            img = self.transform(img)

        return img, {"bboxes": bboxes, "labels": labels}

# mAP 평가 함수
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_mAP(model, dataloader, device):
    coco_gt = COCO(dataloader.dataset.ann_file)
    results = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            pred_logits = outputs["pred_logits"]  # (B, N, num_classes)
            pred_boxes = outputs["pred_boxes"]    # (B, N, 4)
            batch_size = images.shape[0]
            img_w, img_h = 640, 640  # 실제 이미지 크기에 맞게 수정
            for img_idx in range(batch_size):
                logits = pred_logits[img_idx]
                boxes = pred_boxes[img_idx]
                scores, labels = torch.max(logits.softmax(-1)[..., :-1], dim=-1)
                keep = scores > 0.01
                boxes = boxes[keep].cpu().numpy()
                scores = scores[keep].cpu().numpy()
                labels = labels[keep].cpu().numpy()
                # 박스 스케일 복원
                boxes[:, 0] *= img_w
                boxes[:, 1] *= img_h
                boxes[:, 2] *= img_w
                boxes[:, 3] *= img_h
                for bbox, score, label in zip(boxes, scores, labels):
                    x, y, w, h = bbox
                    coco_cat_id = idx_to_cat[int(label)]
                    result = {
                        "image_id": dataloader.dataset.img_info_list[img_idx]["id"],
                        "category_id": coco_cat_id,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score),
                    }
                    results.append(result)
    if len(results) == 0:
        print("Warning: No detection results! mAP will be set to 0.")
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]

# 테스트 실행 코드
if __name__ == "__main__":
    data_root = "./SFU/SFU_HW_Obj"  # 로컬 경로로 변경
    transform = T.Compose([
        T.ToTensor(),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNetwork().to(device) # 여기에 수정
    checkpoint_path = "./model_checkpoint/epoch_10.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for data in os.listdir(data_root):
        dataset = TestDataset(data_root, [data], transform)
        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        mAP = evaluate_mAP(model, test_dataloader, device)
        print(f"Dataset {data} - mAP: {mAP:.3f}")

# 테스트용 10 epoch 체크포인트 불러오기 예시 (main 부분에 추가)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyNetwork().to(device)
# checkpoint = torch.load("./model_checkpoint/epoch_10.pth", map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval() 