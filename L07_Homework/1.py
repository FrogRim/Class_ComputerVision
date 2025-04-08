import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터셋 로드 및 전처리(train set : 60000 , test set : 10000)
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True)

"""
MNIST 데이터셋 이미지를 전처리하기 위한 변환 파이프라인을 정의합니다.
- `transforms.ToTensor()`: PIL 이미지 또는 NumPy 배열을 PyTorch 텐서로 변환합니다. 또한 픽셀 값을 [0, 255] 범위에서 [0.0, 1.0] 범위로 스케일링.
- `transforms.Normalize((0.5,), (0.5,))`: 텐서 이미지를 정규화합니다. 각 채널에 대해 평균(0.5)을 빼고 표준편차(0.5)로 나눈다다
이 변환은 MNIST 데이터셋에 적용되어 훈련 및 테스트 시 일관된 전처리를 보장.
"""
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set.transform = trans
test_set.transform = trans

batch_size = 128

# DataLoader를 사용하여 배치 단위로 데이터 로드
# DataLoader는 데이터셋을 배치 단위로 나누어주는 역할
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

input_size = 1 * 28 * 28 # 1채널, 28x28 이미지(28x28=784)
num_classes = 10 # 0~9 숫자 클래스 수

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  
    nn.ReLU(), 
    nn.Linear(512, 512),  
    nn.ReLU(),  
    nn.Linear(512, num_classes)  
]

model = nn.Sequential(*layers)
optimizer = Adam(model.parameters(), lr=0.001)

# 손실 함수 정의
# CrossEntropyLoss는 다중 클래스 분류 문제에서 자주 사용되는 손실 함수로, 모델의 출력과 실제 레이블 간의 차이를 측정
# 이 함수를 사용하기에에 모델의 마지막 레이어는 softmax()가 적용되지 않습니다.
# 이유: nn.CrossEntropyLoss 함수 내부에서 log softmax를 적용하여 수치 안정성과 효율성을 확보하므로 별도의 softmax layer가 필요하지 않습니다.
loss_function = nn.CrossEntropyLoss()

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

train_loss, train_acc = [],[]

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, len(train_loader.dataset))

    train_loss.append(loss)
    train_acc.append(accuracy)

    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

test_loss, test_acc = [],[]

def test():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, len(test_loader.dataset))

    test_loss.append(loss)
    test_acc.append(accuracy)

    print('test - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


epochs = 20

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    test()