import sys
import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==========================================
# 1. 硬件设备配置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")
if device.type == "cuda":
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. 超参数配置
# ==========================================
BATCH_SIZE = 128
PEAK_LR = 1e-3
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-2         # AdamW 原生权重衰减
EPOCHS = 10
WARMUP_RATIO = 0.1

# ==========================================
# 3. 数据集与增强 (MNIST)
# ==========================================
transform_train = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=True)

TOTAL_STEPS = len(train_loader) * EPOCHS
WARMUP_STEPS = int(TOTAL_STEPS * WARMUP_RATIO)

print(f"训练集样本量: {len(train_dataset)} | 测试集样本量: {len(test_dataset)}")
print(f"每个 Epoch 步数: {len(train_loader)} | 总步数: {TOTAL_STEPS} | Warmup 步数: {WARMUP_STEPS}")

# ==========================================
# 4. 网络架构：2 层 256 隐藏层 (Linear -> BN -> ReLU -> Dropout)
# ==========================================
class UltimateMNISTNet(nn.Module):
    def __init__(self, in_features=784, hidden_features=256, num_classes=10, dropout_p=0.2):
        super(UltimateMNISTNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            # 第 1 层隐藏层 (784 -> 256)
            nn.Linear(in_features, hidden_features, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            # 第 2 层隐藏层 (256 -> 256)
            nn.Linear(hidden_features, hidden_features, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            # 输出层 (256 -> 10)
            nn.Linear(hidden_features, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

model = UltimateMNISTNet().to(device)

# ==========================================
# 5. 损失函数与优化器 (原生 AdamW weight_decay)
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)

# ==========================================
# 6. 学习率调度器：Warmup + 余弦退火
# ==========================================
def lr_lambda(current_step: int):
    if current_step < WARMUP_STEPS:
        return float(current_step) / float(max(1, WARMUP_STEPS))
    progress = float(current_step - WARMUP_STEPS) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr_ratio = MIN_LR / PEAK_LR
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# ==========================================
# 7. 训练与评估循环
# ==========================================
print("\n开始训练...")
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss /= total
    train_acc = correct / total * 100
    current_lr = optimizer.param_groups[0]['lr']

    # 验证测试集
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            test_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_loss /= test_total
    test_acc = test_correct / test_total * 100
    if test_acc > best_acc:
        best_acc = test_acc

    print(f"Epoch [{epoch:02d}/{EPOCHS}] | "
          f"LR: {current_lr:.6f} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

print(f"\n训练完成！最佳测试集准确率: {best_acc:.2f}%")
