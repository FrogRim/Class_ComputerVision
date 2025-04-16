# Recognition Homework


## ✅ 과제 1: MNIST 분류기 (Fully Connected Network)

### 📌 요구사항
- MNIST 데이터셋 로드
- 훈련/테스트 세트 분할
- 간단한 신경망 모델(FCN) 설계
- 모델 학습 및 정확도 평가

### 💡 주요 구현 포인트
- `torchvision.datasets.MNIST`로 데이터 로딩
- `FCModel` 클래스: 두 개의 은닉층을 포함한 신경망 구조 정의
- `run()` 함수: 학습 및 테스트 루틴을 하나로 통합하여 코드 간결화
- `nn.CrossEntropyLoss` + `Adam` 옵티마이저 사용
- GPU 사용 가능 시 자동 전환

### 📊 실행 결과 (코랩 기준)
```
Epoch 1: Train acc 0.9023 | Test acc 0.9315
Epoch 5: Train acc 0.9781 | Test acc 0.9645
```

### 🖼️ 결과 이미지
![Image](https://github.com/user-attachments/assets/d4bc3097-f117-4857-bbfa-abc2dfb69165)



### 🧾 코드 전문 (Python Script)
```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam

"""
MNIST 데이터셋 이미지를 전처리하기 위한 변환 파이프라인을 정의합니다.
- `transforms.ToTensor()`: PIL 이미지 또는 NumPy 배열을 PyTorch 텐서로 변환합니다. 또한 픽셀 값을 [0, 255] 범위에서 [0.0, 1.0] 범위로 스케일링.
- `transforms.Normalize((0.5,), (0.5,))`: 텐서 이미지를 정규화합니다. 각 채널에 대해 평균(0.5)을 빼고 표준편차(0.5)로 나눈다다
이 변환은 MNIST 데이터셋에 적용되어 훈련 및 테스트 시 일관된 전처리를 보장.
"""')

# 데이터 전처리: 정규화 적용 (평균 0.5, 표준편차 0.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST 데이터 로드
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128)

# 신경망 모델 정의
class FCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                          # 28x28 -> 784
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10)                     # 10개의 숫자 클래스 출력
        )

    def forward(self, x):
        return self.net(x)

# 모델, 손실 함수, 옵티마이저 초기화
# CrossEntropyLoss는 다중 클래스 분류 문제에서 자주 사용되는 손실 함수로, 모델의 출력과 실제 레이블 간의 차이를 측정
# 이 함수를 사용하기에에 모델의 마지막 레이어는 softmax()가 적용되지 않습니다.
# 이유: nn.CrossEntropyLoss 함수 내부에서 log softmax를 적용하여 수치 안정성과 효율성을 확보하므로 별도의 softmax layer가 필요하지 않습니다.
model = FCModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# 학습 및 테스트 함수 정의
def run(loader, train=True):
    model.train() if train else model.eval()
    loss_sum, correct = 0.0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            if train:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()

    return loss_sum / len(loader.dataset), correct / len(loader.dataset)

# 에폭 반복 학습 및 평가
for epoch in range(5):
    tr_loss, tr_acc = run(train_loader, True)
    te_loss, te_acc = run(test_loader, False)
    print(f"Epoch {epoch+1}: Train acc {tr_acc:.4f} | Test acc {te_acc:.4f}")
```

---

## ✅ 과제 2: CIFAR-10 CNN

### 📌 요구사항
- CIFAR-10 데이터셋 로드 및 정규화
- CNN 모델 설계 및 학습
- 모델 성능 평가 및 예측 시각화

### 💡 주요 구현 포인트
- `CIFAR10` 데이터셋 사용 + 80:20 비율로 훈련/검증 분할
- `Conv2D → BatchNorm → ReLU` 블록 3단 구성
- `train()`과 `eval()` 함수로 `MaxPool + Dropout`으로 과적합 방지
- 학습/검증 분할, Early Stopping 구현

### 📊 실행 결과
```
최종 Test Accuracy: 0.855
```

### 🖼️ 결과 이미지
![Image](https://github.com/user-attachments/assets/6cce4f84-eaea-4cad-8d35-0a1be8712287)



### 🧾 코드 전문 (Python Script)
```python
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import Adam

# 전처리 정의 (CIFAR10 통계 기반 정규화)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# 데이터 로드 및 분할
full_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train_ds, val_ds = random_split(full_train, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

# CNN 모델 정의
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 모델, 손실 함수, 옵티마이저
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-3)

# 학습/검증 루프 정의
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    loss_sum, correct = 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)

# 학습 수행
best_acc, best_model = 0, None
for epoch in range(20):
    _, tr_acc = run_epoch(train_loader, True)
    _, val_acc = run_epoch(val_loader, False)
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = model.state_dict()
    print(f"Epoch {epoch+1:02d} | Train {tr_acc:.3f} | Val {val_acc:.3f}")

model.load_state_dict(best_model)
_, test_acc = run_epoch(test_loader, False)
print(f"최종 Test Accuracy: {test_acc:.3f}")
```

---

## ✅ 과제 3: CIFAR-10 전이학습 (VGG16)

### 📌 요구사항
- 사전학습된 VGG16 모델 로드 & 최상위 레이어 제거
- 새 데이터셋에 맞는 레이어 추가
- 모델 훈련 및 평가
- 기존 모델(CNN)과 비교

### 💡 주요 구현 포인트
- `torchvision.models.vgg16(weights=...)`로 사전학습 된 VGG16모델을 로드 + 마지막 분류기 `classifier[6]` 수정
- `VGG16` 특징 추출부는 freeze(가중치 고정)
- CIFAR-10 → VGG 입력(224x224)로 resize & normalize
- 기존 CNN (과제2) 대비 성능 향상 비교

### 📊 실행 결과
```
[VGG16] Epoch 5: Train Acc=0.7864, Test Acc=0.8348
(CNN 대비 약간 낮음)
```

### 🖼️ 결과 이미지
![Image](https://github.com/user-attachments/assets/82754488-9aa0-4a3b-a621-fb6a23a83233)

### 🧾 코드 전문 (Python Script)
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.optim import Adam

# CIFAR-10 -> VGG 입력 정규화 및 리사이즈
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=tf)
test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32)

# VGG16 불러오기 및 classifier 수정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = vgg16(weights=VGG16_Weights.DEFAULT)
for p in vgg.features.parameters():
    p.requires_grad = False
vgg.classifier[6] = nn.Linear(4096, 10)
vgg = vgg.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(vgg.classifier[6].parameters(), lr=1e-3)

# 학습/테스트 루프 정의
def run_epoch(loader, train=True):
    vgg.train() if train else vgg.eval()
    loss_sum, correct = 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad()
            out = vgg(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)

# 5 에폭 학습
for epoch in range(5):
    _, tr_acc = run_epoch(train_loader, True)
    _, te_acc = run_epoch(test_loader, False)
    print(f"[VGG16] Epoch {epoch+1}: Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")
```
