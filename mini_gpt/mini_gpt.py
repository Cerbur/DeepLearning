import math
import random
from collenctions import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)  # 限制 CPU 线程数，避免占用过多资源

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


# 模型参数
D_MODEL = 128  # 模型维度
NUM_HEADS = 8  # 注意力头数
D_FF = 512  # 前馈网络维度
NUM_LAYERS = 8  # Transformer 层数
MAX_SEQ_LEN = 128  # 最大序列长度
ROPE_BASE = 10000.0  # Rotary Positional Embedding 基数
DROPOUT_RATE = 0.1
MAX_TOKEN_FREQ = 5
character_counts = Counter("", join(all_poems))
kept_ characters = {
    char
    for char, freq in character_counts.items()
    if char in "，。" or freq >= MAX_TOKEN_FREQ
}

tokens = ["<BOS>", "<EOS>", "<UNK>"] + sorted(kept_characters)
tokens_ids = {token: idx for idx, token in enumerate(tokens)}
BOS = tokens_ids["<BOS>"]
EOS = tokens_ids["<EOS>"]
UNK = tokens_ids["<UNK>"]

# 训练参数
BATCH_SIZE = 64 # 训练批次大小
MAX_EPOCHS = 1000 # 最大训练轮数
MIN_EPOCHS = 10 # 最小训练轮数
LEARNING_RATE = 0.002 # 学习率
EARLY_STOPPING_PATIENCE = 10 # 早停策略的耐心值
EARLY_STOPPING_DELTA = 0.0001 # 早停策略的最小改进阈值
WEIGHT_DECAY = 0.01 # 权重衰减
LABEL_SMOOTHING = 0.1 # 标签平滑


# def encode(text):

# def decode(ids):


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        # 初始化嵌入层权重
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # train() 方法中启用 Dropout，eval() 方法中禁用 Dropout
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, token_ids):
        return self.dropout(self.embedding(token_ids))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, max_length, base=ROPE_BASE):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE 要求 head_dim 必须为偶数"
        
        # head_dim // 2 个“频率因子”，每个对应一对维度，生成对的矩阵
        pair_index = torch.arange(head_dim // 2, dtype=torch.float32)

        # θ_i = base^(-2*i / head_dim)，给 对矩阵用这个公式算出每个维度（对）的旋转角度
        theta = base ** (-2 * pair_index / head_dim)

        # 位置编码：position = [0, 1, 2, ..., max_length-1]，生成位置矩阵
        position = torch.arange(max_length, dtype=torch.float32)

        # angles[p, i] = position[p] * theta[i]，位置矩阵和对矩阵外积，得到每个位置的旋转角度，也就是每个位置，每个对的旋转角
        angles = torch.outer(position, theta)

        self.register_buffer("cos", torch.cos(angles))  # cos(θ)
        self.register_buffer("sin", torch.sin(angles))  # sin(θ)

    def forward(self, x):
        # x: [batch_size, num_heads, seq_len, head_dim]
        sequence_length = x.shape[-2] # 获取序列长度

        cos = self.cos[:sequence_length] # 截取对应长度的 cos
        sin = self.sin[:sequence_length] # 截取对应长度的 sin

        even = x[..., ::2]  # 偶数维度
        odd = x[..., 1::2]  # 奇数维度

        rotated_even = even * cos - odd * sin  # 偶数维度旋转
        rotated_odd = even * sin + odd * cos    # 奇数维度旋

        # torch.stack((even, odd), dim=-1)	[[e0,o0],[e1,o1],...]
        rotated_x = torch.stack((rotated_even, rotated_odd), dim=-1)  # 重新组合偶数和奇数维度

        # torch.stack(..., dim=-1).flatten(-2)	[e0,o0,e1,o1,...]
        return rotated_x.flatten(-2)  # 展平最后两个维度，返回旋转后的 x

def attention(q, k, v):
    # [B, H, L, D] [L,D]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # 计算注意力分数
    
    seq_len = q.shape[-2]

    # 创建上三角掩码，防止模型看到未来的 token
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)

    scores = scores.masked_fill(mask, float("-inf"))  # 将上三角部分置为负无穷，softmax 后为 0

    weights = torch.softmax(scores, dim=-1)  # 计算注意力权重

    return weights @ v  # 返回加权后的值


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):

     

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert D_MODEL % NUM_HEADS == 0, "D_MODEL 必须能被 NUM_HEADS 整除"
        self.head_dim = D_MODEL // NUM_HEADS

        # qkv_proj: 将输入 x 映射到 Q、K、V 三个矩阵，输出维度为 3 * D_MODEL
        self.qkv_proj = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)  # Q、K、V 的线性变换

        # out_proj: 将多头注意力的输出映射回 D_MODEL 维度
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)  # 输出线性变换

        # RoPE 位置编码
        self.rope = RotaryPositionalEmbedding(self.head_dim, MAX_SEQ_LEN)  # RoPE 位置编码

    def forward(self, x):
        unbatched = x.ndim == 2  # 判断输入是否为未批处理的单个序列
        if unbatched:
            x = x.unsqueeze(0)  # 如果是单个序列，增加 batch 维度


        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)  # [B, L, 3*D_MODEL]
        q, k, v = qkv.chunk(3, dim=-1)  # 将 qkv 分割为 Q、K、V 三个矩阵

        def split_heads(tensor):
            # 将 [B, L, D_MODEL] -> [B, L, H, head_dim] -> [B, H, L, head_dim]
            return tensor.reshape(batch_size, seq_len, NUM_HEADS, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        q = self.rope(q)
        k = self.rope(k)

        context = attention(q, k, v)  # 计算注意力输出

        # 将 [B, H, L, head_dim] -> [B, L, H, head_dim] -> [B, L, D_MODEL]
        joined = context.transpose(1, 2).contiguous().reshape(batch_size, seq_len, D_MODEL)  # 将多头注意力输出拼接回原维度
        output = self.out_proj(joined)  # 输出线性变换

        return output.squeeze(0) if unbatched else output  # 如果是单个序列，去掉 batch 维度



        

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps  # 避免除以零
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 计算均方根
        return self.weight * (x / rms)  # 归一化并缩放

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FF, bias=False)  # 门控线性变换
        self.up_proj = nn.Linear(D_MODEL, D_FF, bias=False)  # 上采样线性变换
        self.down_proj = nn.Linear(D_FF, D_MODEL, bias=False)  # 下采样线性变换
        self.activation = nn.SiLU()  # SiLU 激活函数

    def forward(self, x):
        return self.down_proj(self.activation(self.up_proj(x)) * self.gate_proj(x))  # SwiGLU FFN

class GPTBlock:
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(D_MODEL)  # 第一个归一化层
        self.attention = MultiHeadAttention()  # 多头自注意力
        self.norm2 = RMSNorm(D_MODEL)  # 第二个归一化层
        self.ffn = SwiGLU()  # 前馈网络
        self.dropout = nn.Dropout(DROPOUT_RATE)  # Dropout 层

    def forward(self, x):        
        x = x + self.dropout(self.attention(self.norm1(x)))  # 残差连接 + 注意力
        x = x + self.dropout(self.ffn(self.norm2(x)))  # �残差连接 + 前馈网络
        return x


class GPTDecoder(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([GPTBlock() for _ in range(num_layers)])  # 堆叠多个 GPTBlock
        self.norm = RMSNorm(D_MODEL)  # 最后的归一化层

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # 逐层前向传播
        return self.norm(x)  # 最后归一化输出

class MiniGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size)  # Token 嵌入层
        self.decoder = GPTDecoder(NUM_LAYERS)  # GPT 解码器
        self.output_proj = nn.Linear(D_MODEL, vocab_size, bias=False)  # 输出投影层

        self.output_proj.weight = self.embedding.embedding.weight  # 输出投影层权重共享 Token 嵌入层权重


    def forward(self, token_ids):
        x = self.embedding(token_ids)  # Token 嵌入
        x = self.decoder(x)  # GPT 解码器
        return self.output_proj(x)  # 输出投影


train_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
eval_criterion = nn.CrossEntropyLoss()

def mean_loss(model, sequences):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    total_poems = 0

    with torch.no_grad():  # 评估时不需要计算梯度
        for start in range(0, len(sequences), BATCH_SIZE):
            batch = torch.stack(sequences[start:start + BATCH_SIZE]).to(device)  # 获取一个批次的序列
            inputs = batch[:, :-1]  # 输入序列（去掉最后一个 token）
            labels = batch[:, 1:]  # 标签序列（去掉第一个 token）
            logits = model(inputs)  # 前向传播，获取 logits
            loss = eval_criterion(logits.reshape(-1, tokens), labels.reshape(-1))  # 计算损失
            total_loss += loss.item() * batch.size(0)  # 累加损失
            total_poems += batch.size(0)  # 累加样本数量

    return total_loss / total_poems if total_poems > 0 else 0.0

class EarlyStopping:
    def __init__(self, patience, min_delta, min_epochs):
        self.patience = patience  # 耐心值
        self.min_delta = min_delta  # 最小改进阈值
        self.min_epochs = min_epochs  # 最小训练轮数
        self.best_loss = float("inf")  # 初始化最佳损失为无穷大
        self.best_epoch = 0  # 初始化最佳轮数为 0
        self.best_model_state = None  # 初始化最佳模型状态为 None
        self.epochs_since_improvement = 0  # 初始化自上次改进以来的轮数为 0

    def step(self, valid_loss, model, epoch):
        improved = valid_loss < self.best_loss - self.min_delta  # 判断是否有改进
        if improved:
            self.best_loss = valid_loss  # 更新最佳损失
            self.best_epoch = epoch  # 更新最佳轮数
            self.best_model_state =  {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            self.epochs_since_improvement = 0  # 重置自上次改进以来的轮数
        elif epoch >= self.min_epochs:
            self.epochs_since_improvement += 1  # 增加自上次改进以来的轮数
        else:
            self.epochs_since_improvement = 0  # 如果还没到最小训练轮数，重置自上次改进以来的轮数

model = MiniGPT(vocab_size=len(tokens)).to(device)  # 初始化 MiniGPT 模型

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # AdamW 优化器

# 初始化早停策略
early_stopping = EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE, 
    min_delta=EARLY_STOPPING_DELTA, 
    min_epochs=MIN_EPOCHS
)

train_losses = []  # 训练损失列表
valid_losses = []  # 验证损失列表

def train_one_epoch(model, examples):
    model.train()  # 设置模型为训练模式
    random.shuffle(examples)  # 打乱训练数据
    running_objective = 0.0  # 累计损失
    seen_poems = 0  # 累计样本数量

    for start in range(0, len(examples), BATCH_SIZE):
        batch = torch.stack(examples[start:start + BATCH_SIZE]).to(device)  # 获取一个批次的序列
        inputs = batch[:, :-1]  # 输入序列（去掉最后一个 token）
        labels = batch[:, 1:]  # 标签序列（去掉第一个 token

        logits = model(inputs)  # 前向传播，获取 logits
        loss = train_criterion(logits.reshape(-1, len(tokens)), labels.reshape(-1))  # 计算损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数
        
        running_objective += loss.item() * batch.size(0)  # 累加损失
        seen_poems += batch.size(0)  # 累加样本数量

    return running_objective / seen_poems if seen_poems > 0 else 0.0  # 返回平均损失

for epoch in range(1, MAX_EPOCHS + 1):
    train_objective = train_one_epoch(model, train_examples)  # 训练一个 epoch
    train_loss = mean_loss(model, train_eval_examples)  # 计算训练集平均损失
    valid_loss = mean_loss(model, valid_eval_examples)  # 计算验证集平均损失
    train_losses.append(train_loss)  # 记录训练损失
    valid_losses.append(valid_loss)  # 记录验证损失

    current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
    print(f"Epoch :{epoch:4d}, Train Objective: {train_objective:.4f}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Learning Rate: {current_lr:.4f}")
    if early_stopping.step(valid_loss, model, epoch):
        print(f"Early stopping triggered at epoch {epoch}. Best epoch: {early_stopping.best_epoch}, Best loss: {early_stopping.best_loss:.4f}")
        break  # 提前停止训练
