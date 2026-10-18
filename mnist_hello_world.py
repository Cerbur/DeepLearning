import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 确保在 Windows 控制台输出中文不乱码
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==========================================
# 1. 硬件设备配置（优先使用 GPU 加速）
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")
if device.type == "cuda":
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. 超参数设置
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.1   # 用户指定学习率 0.1
EPOCHS = 5

# ==========================================
# 3. 准备 MNIST 数据集（60,000 张训练集，10,000 张测试集）
# ==========================================
# ToTensor() 将像素值 [0, 255] 缩放到 [0.0, 1.0] 的张量
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准化：(x - mean) / std
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"训练集样本数量: {len(train_dataset)}")
print(f"测试集样本数量: {len(test_dataset)}")

# ==========================================
# 4. 定义神经网络模型
# 输入 784 -> 隐藏层 128 -> ReLU -> 输出层 10 (输出 logits)
# ==========================================
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 隐藏层：784 输入，128 输出
        self.relu = nn.ReLU()           # 激活函数：ReLU
        self.fc2 = nn.Linear(128, 10)   # 输出层：128 输入，10 输出

    def forward(self, x):
        # 展平输入：(batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)                 # 返回 logits（未经过 softmax 的原始网络输出得分）
        return x

model = MNISTNet().to(device)

# ==========================================
# 5. 损失函数与优化器
# ==========================================
# nn.CrossEntropyLoss 会接收 logits 并自动结合 LogSoftmax 与 NLLLoss，具有更好的数值稳定性
criterion = nn.CrossEntropyLoss()

# 优化器：使用 PyTorch 标准的 SGD（随机梯度下降）
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 6. 训练循环（使用 SGD 优化器更新参数）
# ==========================================
print("\n开始训练...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 优化器梯度清零（也可以放在 loss.backward() 之前）
        optimizer.zero_grad()

        # 6.1 前向传播 (计算 logits)
        logits = model(images)

        # 6.2 计算损失
        loss = criterion(logits, labels)

        # 6.3 反向传播 (自动求导，计算各参数梯度)
        loss.backward()

        # 6.4 【原手写参数更新代码，现已按要求注释】
        # with torch.no_grad():
        #     for param in model.parameters():
        #         if param.grad is not None:
        #             # 原地更新参数: W = W - η * ∇W
        #             param.sub_(LEARNING_RATE * param.grad)
        #             # 清零梯度
        #             param.grad.zero_()

        # 改用 SGD 优化器更新参数（内部等价于遍历参数执行 param.data -= lr * param.grad）
        optimizer.step()

        # 累计统计数据
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total * 100

    # 7. 测试集验证评估
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

    val_loss = test_loss / test_total
    val_acc = test_correct / test_total * 100

    print(f"Epoch [{epoch}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%")

print("\n训练完成！")
