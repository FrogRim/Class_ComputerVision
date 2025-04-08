import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.optim import Adam

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = vgg16(weights=VGG16_Weights.DEFAULT)
for p in vgg.features.parameters(): p.requires_grad = False
vgg.classifier[6] = nn.Linear(4096, 10)
vgg = vgg.to(device)

optimizer = Adam(vgg.classifier[6].parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def run_epoch(loader, train=True):
    vgg.train() if train else vgg.eval()
    loss, correct = 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad()
            out = vgg(x)
            l = criterion(out, y)
            if train:
                l.backward()
                optimizer.step()
            loss += l.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)

for epoch in range(5):
    _, tr_acc = run_epoch(train_loader, True)
    _, te_acc = run_epoch(test_loader, False)
    print(f"[VGG16] Epoch {epoch+1}: Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")
