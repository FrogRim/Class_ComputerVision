import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128)

class FCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x): return self.net(x)

model = FCModel().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def run(loader, train):
    model.train() if train else model.eval()
    loss, correct = 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad()
            out = model(x)
            l = criterion(out, y)
            if train:
                l.backward()
                optimizer.step()
            loss += l.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)

for epoch in range(5):
    _, tr_acc = run(train_loader, True)
    _, te_acc = run(test_loader, False)
    print(f"Epoch {epoch+1}: Train acc {tr_acc:.4f} | Test acc {te_acc:.4f}")
