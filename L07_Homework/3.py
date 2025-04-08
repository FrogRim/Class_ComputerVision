import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights

# 1. VGG16 모델 로드 및 수정
weights = VGG16_Weights.DEFAULT
vgg = vgg16(weights=weights)

# 최상위 레이어 제거 (FC 레이어 수정)
vgg.classifier[6] = nn.Linear(4096, 10)  # MNIST는 10개의 클래스

# 가중치 고정 (Feature Extractor로 사용)
for param in vgg.features.parameters():
    param.requires_grad = False

# 2. 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG16 입력 크기에 맞게 조정
    transforms.Grayscale(3),       # MNIST는 1채널이므로 3채널로 변환
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

# 3. 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = Adam(vgg.classifier[6].parameters(), lr=0.001)  # 새로 추가된 레이어만 학습

# 4. 학습 루프
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (output.argmax(1) == y).sum().item()
    return total_loss / len(loader), total_correct / len(loader.dataset)

def test(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item()
            total_correct += (output.argmax(1) == y).sum().item()
    return total_loss / len(loader), total_correct / len(loader.dataset)

# 5. 학습 및 평가
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)

epochs = 5
for epoch in range(epochs):
    train_loss, train_acc = train(vgg, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(vgg, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


# 기존 모델 로드 (1.py의 모델)
from model_1 import model as fc_model  # 1.py의 모델을 import했다고 가정
from model_1 import train as train_fc, test as test_fc  # 기존 모델의 학습/테스트 함수

# 기존 모델 학습 및 평가
fc_model.to(device)
fc_optimizer = Adam(fc_model.parameters(), lr=0.001)
fc_loss_function = nn.CrossEntropyLoss()

print("\n=== 기존 Fully Connected 모델 학습 및 평가 ===")
for epoch in range(epochs):
    train_loss, train_acc = train_fc(fc_model, train_loader, fc_loss_function, fc_optimizer, device)
    test_loss, test_acc = test_fc(fc_model, test_loader, fc_loss_function, device)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# VGG16 전이 학습 모델 학습 및 평가
print("\n=== VGG16 전이 학습 모델 학습 및 평가 ===")
for epoch in range(epochs):
    train_loss, train_acc = train(vgg, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(vgg, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")