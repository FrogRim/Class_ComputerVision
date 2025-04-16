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
