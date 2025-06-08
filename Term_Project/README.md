# 객체 검출 프로젝트

이 프로젝트는 Faster R-CNN 기반의 객체 검출 시스템을 구현한 것입니다. COCO 데이터셋의 6개 클래스(person, bicycle, car, motorcycle, bus, truck)를 대상으로 합니다.

## 1. 딥러닝 네트워크 구조

Faster R-CNN 기반의 커스텀 네트워크를 구현했습니다.

```python
class MyNetwork(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, backbone_name='resnet50',
                 min_size=800, max_size=1333,
                 box_detections_per_img=300,
                 box_score_thresh=0.01,
                 box_nms_thresh=0.3):
        super(MyNetwork, self).__init__()
        
        # Backbone: ResNet50 with FPN
        extra_blocks = LastLevelP6P7(256, 256)
        self.backbone = resnet_fpn_backbone(
            backbone_name,
            pretrained=True,
            trainable_layers=5,
            returned_layers=[1, 2, 3, 4],
            extra_blocks=extra_blocks
        )
        
        # RPN 설정
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 6
        rpn_anchor_generator = rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # ROI Head 설정
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
```

## 2. Train 코드

학습 과정은 다음과 같이 구현되어 있습니다:

```python
def train(model, optimizer, device):
    epochs = 40
    model.train()
    best_val_loss = float('inf')
    patience = 8
    
    for epoch in range(epochs):
        for images, targets in tqdm(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 유효한 이미지만 처리
            valid_images = []
            valid_targets = []
            for img, target in zip(images, targets):
                if len(target["boxes"]) > 0:
                    valid_images.append(img)
                    valid_targets.append(target)
            
            loss_dict = model(valid_images, valid_targets)
            losses, _ = loss_fn(loss_dict)
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
```

## 3. Loss Function

커스텀 손실 함수는 RPN과 ROI Head의 손실을 조합하여 구현했습니다:

```python
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
```

## 4. Test.py Inference 코드

추론 과정은 다음과 같이 구현되어 있습니다:

```python
@torch.no_grad()
def evaluate_mAP_optimized(model, dataloader, device, label2cat, dataset_name=""):
    model.eval()
    coco_gt = dataloader.dataset.coco
    results = []
    
    for images, targets in tqdm(dataloader, desc="Optimized Evaluating"):
        images = list(img.to(device) for img in images)
        outputs = model(images)
        
        for idx, output in enumerate(outputs):
            image_id = targets[idx]["image_id"].item()
            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()
            
            # 스마트 부스트 및 NMS 적용
            boosted_scores = apply_smart_boost(scores, labels, settings, label2cat)
            boxes, scores, labels = apply_advanced_nms(boxes, boosted_scores, labels, settings)
```

## 5. 다중 오브젝트 처리와 카테고리 ID 매핑

### 카테고리 ID 매핑
```python
# COCO 클래스 정의
TARGET_COCO_CLASSES = [1, 2, 3, 4, 6, 8]  # person, bicycle, car, motorcycle, bus, truck
CLASS_NAMES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
NUM_CLASSES = len(TARGET_COCO_CLASSES) + 1  # 6개 객체 클래스 + 1개 배경 클래스

# 내부 레이블 매핑
sorted_classes = sorted(TARGET_COCO_CLASSES)
cat2label = {cat_id: idx for idx, cat_id in enumerate(sorted_classes)}
label2cat = {idx: cat_id for cat_id, idx in cat2label.items()}
```

### 다중 오브젝트 처리
```python
class COCODataset(Dataset):
    def __getitem__(self, index):
        # 이미지 로드
        imgId = self.img_ids[index]
        annIds = self.coco.getAnnIds(imgIds=imgId)
        anns = self.coco.loadAnns(annIds)
        
        # 다중 오브젝트 처리
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
```

## 주요 특징

1. **다중 오브젝트 처리**:
   - RPN에서 2000개의 제안 영역 생성
   - ROI Head에서 512개의 샘플 처리
   - 배치 단위로 여러 이미지와 객체 처리

2. **카테고리 매핑**:
   - COCO 클래스 ID를 내부 레이블로 변환
   - 배경 클래스 포함 7개 클래스 처리
   - 클래스별 가중치 적용 가능

3. **최적화된 추론**:
   - 스마트 부스트로 클래스별 가중치 적용
   - 고급 NMS로 중복 검출 제거
   - 도메인별 최적화된 설정 적용
