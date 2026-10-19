import argparse
import csv
import math
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn

# 模型参数
D_MODEL = 96  # 模型维度
NUM_HEADS = 6  # 注意力头数
D_FF = 256  # 前馈网络维度
NUM_LAYERS = 4  # Transformer 层数
MAX_SEQ_LEN = 128  # 最大序列长度
ROPE_BASE = 10000.0  # Rotary Positional Embedding 基数
DROPOUT_RATE = 0.1
MAX_TOKEN_FREQ = 5
# 训练参数
BATCH_SIZE = 64 # 训练批次大小
MAX_EPOCHS = 1000 # 最大训练轮数
MIN_EPOCHS = 10 # 最小训练轮数
LEARNING_RATE = 0.002 # warmup 后的峰值学习率
WARMUP_RATIO = 0.05 # 前 5% 的训练步数线性预热
MIN_LR_RATIO = 0.1 # 余弦退火最低学习率 / 峰值学习率
EARLY_STOPPING_PATIENCE = 10 # 早停策略的耐心值
EARLY_STOPPING_DELTA = 0.0001 # 早停策略的最小改进阈值
WEIGHT_DECAY = 0.01 # 权重衰减
LABEL_SMOOTHING = 0.1 # 标签平滑
TRAIN_EVAL_SIZE = 1024 # 固定训练评估子集大小，0 表示完整训练集


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



        

class RMSNorm(nn.Module):
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

class GPTBlock(nn.Module):
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


# 数据、训练与推理（模型定义以上保持原计算逻辑）
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "notebook" / "data" / "five_char_quatrains.tsv"
DEFAULT_OUTPUT = BASE_DIR / "checkpoints"
MODEL_CONFIG = {name: globals()[name] for name in (
    "D_MODEL", "NUM_HEADS", "D_FF", "NUM_LAYERS", "MAX_SEQ_LEN",
    "ROPE_BASE", "DROPOUT_RATE",
)}


def load_poems(path):
    """只读取 text 列，去空白、过滤非标准五绝并去重，避免跨集合重复。"""
    poems, seen, skipped = [], set(), 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("TSV 必须包含 text 列")
        for row in reader:
            text = "".join((row.get("text") or "").split())
            if not re.fullmatch(r"(?:[^，。]{5}，[^，。]{5}。){2}", text):
                skipped += 1
                continue
            if text not in seen:
                seen.add(text)
                poems.append(text)
    if len(poems) < 2:
        raise ValueError("至少需要两首有效五言绝句才能划分训练集和验证集")
    print(f"有效且去重诗歌: {len(poems)}，格式不符跳过: {skipped}")
    return poems


class CharTokenizer:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        if self.tokens[:3] != ["<BOS>", "<EOS>", "<UNK>"]:
            raise ValueError("检查点中的特殊 token 不兼容")
        self.token_ids = {token: i for i, token in enumerate(self.tokens)}
        self.bos, self.eos, self.unk = 0, 1, 2

    @classmethod
    def build(cls, poems, min_freq=MAX_TOKEN_FREQ):
        counts = Counter("".join(poems))
        chars = {c for c, n in counts.items() if n >= min_freq} | set("，。")
        return cls(["<BOS>", "<EOS>", "<UNK>"] + sorted(chars))

    def encode(self, text, special_tokens=True):
        ids = [self.token_ids.get(c, self.unk) for c in text]
        return [self.bos] + ids + [self.eos] if special_tokens else ids

    def decode(self, ids):
        return "".join(self.tokens[i] for i in ids if i not in (self.bos, self.eos))


def make_sequences(poems, tokenizer):
    # 所有有效诗歌均为 24 字符，加 BOS/EOS 为 26，无需 PAD 或改变注意力。
    return torch.tensor([tokenizer.encode(poem) for poem in poems], dtype=torch.long)


def select_train_eval_indices(train_count, sample_size, seed, saved_indices=None):
    """固定抽样，不消耗全局随机状态；恢复时校验并复用原索引。"""
    if (type(train_count) is not int or train_count < 1
            or type(sample_size) is not int or sample_size < 0):
        raise ValueError("训练集须非空，train-eval-size 须为非负整数")
    count = min(sample_size, train_count) if sample_size else train_count
    if saved_indices is not None:
        if (not isinstance(saved_indices, list) or len(saved_indices) != count
                or any(type(i) is not int or not 0 <= i < train_count for i in saved_indices)
                or len(set(saved_indices)) != count):
            raise ValueError("检查点中的训练评估索引无效或与评估大小不匹配")
        return list(saved_indices)
    if count == train_count:
        return list(range(train_count))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(train_count, generator=generator, device="cpu")[:count].tolist()


class WarmupCosineScheduler:
    """按更新步数线性预热、单次余弦退火；构造准备首批，step 准备下一批。"""

    def __init__(self, optimizer, total_steps, warmup_ratio=WARMUP_RATIO,
                 min_lr_ratio=MIN_LR_RATIO):
        if (type(total_steps) is not int or total_steps < 1
                or not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1):
            raise ValueError("total_steps 须为正整数，warmup_ratio 须在 [0,1)")
        self.optimizer = optimizer
        # 小样本时尽量保留峰值、最低值两个余弦阶段位置；单步时直接用峰值。
        warmup_steps = min(max(0, total_steps - 2), math.ceil(total_steps * warmup_ratio))
        self.load_state_dict(dict(
            total_steps=total_steps, warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio, completed_steps=0,
            base_lrs=[group["lr"] for group in optimizer.param_groups]))

    @property
    def total_steps(self):
        return self._state["total_steps"]

    @property
    def warmup_steps(self):
        return self._state["warmup_steps"]

    @property
    def min_lr_ratio(self):
        return self._state["min_lr_ratio"]

    @property
    def completed_steps(self):
        return self._state["completed_steps"]

    @property
    def base_lrs(self):
        return list(self._state["base_lrs"])

    def _apply_lr(self):
        step = self.completed_steps
        if step >= self.total_steps:
            factor = self.min_lr_ratio
        elif step < self.warmup_steps:
            factor = (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps - 1)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)) / 2
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self):
        self._state["completed_steps"] += 1
        self._apply_lr()

    def get_last_lr(self):
        """返回已应用到 optimizer、供下一批使用的各参数组学习率。"""
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_info(self):
        """返回独立的日志快照，不向调用者暴露内部可变状态。"""
        phase = ("finished" if self.completed_steps >= self.total_steps else
                 "warmup" if self.completed_steps < self.warmup_steps else "cosine")
        return dict(total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                    completed_steps=self.completed_steps, phase=phase,
                    peak_lrs=self.base_lrs,
                    min_lrs=[lr * self.min_lr_ratio for lr in self.base_lrs],
                    current_lrs=self.get_last_lr())

    def state_dict(self):
        # 保持旧检查点字段不变，但不再要求状态字段等同于构造参数。
        return dict(self._state, base_lrs=self.base_lrs)

    def load_state_dict(self, state):
        """先校验完整状态，再恢复原曲线和进度并同步 optimizer 学习率。"""
        try:
            total_steps = state["total_steps"]
            warmup_steps = state["warmup_steps"]
            completed_steps = state["completed_steps"]
            min_lr_ratio = state["min_lr_ratio"]
            base_lrs = list(state["base_lrs"])
            valid = (
                type(total_steps) is int and total_steps >= 1
                and type(warmup_steps) is int and 0 <= warmup_steps < total_steps
                and type(completed_steps) is int and completed_steps >= 0
                and math.isfinite(min_lr_ratio) and 0 <= min_lr_ratio <= 1
                and len(base_lrs) == len(self.optimizer.param_groups)
                and all(math.isfinite(lr) and lr >= 0 for lr in base_lrs))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("学习率调度状态无效或不完整") from error
        if not valid:
            raise ValueError("学习率调度状态无效或参数组数量不匹配")
        self._state = dict(total_steps=total_steps, warmup_steps=warmup_steps,
                           completed_steps=completed_steps, min_lr_ratio=min_lr_ratio,
                           base_lrs=base_lrs)
        self._apply_lr()


def run_epoch(model, sequences, batch_size, device, optimizer=None,
              label_smoothing=0.0, scheduler=None):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(sequences)) if training else torch.arange(len(sequences))
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    with torch.set_grad_enabled(training):
        for start in range(0, len(order), batch_size):
            batch = sequences[order[start:start + batch_size]].to(device)
            inputs, labels = batch[:, :-1], batch[:, 1:]
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("出现非有限损失，请检查数据或降低学习率")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * len(batch)
    return total_loss / len(sequences)


def save_checkpoint(path, model, tokenizer, optimizer, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(
        format_version=1, model_config=MODEL_CONFIG,
        model_state=model.state_dict(), tokens=tokenizer.tokens,
        optimizer_state=optimizer.state_dict(),
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **metadata,
    )
    # 临时文件写完再替换，避免中断时破坏上一次检查点。
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到检查点 {path}，请先运行 train 训练模型")
    # 仅加载自己训练或可信来源的文件；torch.load 使用 pickle。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or checkpoint["model_config"] != MODEL_CONFIG:
        raise ValueError("检查点格式或模型参数不匹配，请使用相同的模型结构")
    tokenizer = CharTokenizer(checkpoint["tokens"])
    model = MiniGPT(len(tokenizer.tokens)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint


def train(args, device):
    if args.resume:
        model, tokenizer, checkpoint = load_checkpoint(args.resume, device)
        train_poems, valid_poems = checkpoint["train_poems"], checkpoint["valid_poems"]
        settings = checkpoint["train_settings"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
        bad_epochs, history = checkpoint["bad_epochs"], checkpoint["history"]
    else:
        poems = load_poems(args.data)
        random.Random(args.seed).shuffle(poems)
        if args.max_poems:
            poems = poems[:args.max_poems]
        valid_count = max(1, int(len(poems) * args.valid_ratio))
        valid_poems, train_poems = poems[:valid_count], poems[valid_count:]
        tokenizer = CharTokenizer.build(train_poems, args.min_freq)
        if not any(token not in ("<BOS>", "<EOS>", "<UNK>", "，", "。") for token in tokenizer.tokens):
            raise ValueError("正文词表为空，请降低 --min-freq（小样本建议设为 1）")
        model = MiniGPT(len(tokenizer.tokens)).to(device)
        settings = dict(batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                        label_smoothing=LABEL_SMOOTHING, patience=args.patience,
                        min_delta=EARLY_STOPPING_DELTA, min_epochs=args.min_epochs,
                        warmup_ratio=args.warmup_ratio, min_lr_ratio=args.min_lr_ratio,
                        train_eval_size=args.train_eval_size, train_eval_seed=args.seed)
        start_epoch, best_loss, best_epoch, bad_epochs, history = 1, float("inf"), 0, 0, []
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  weight_decay=settings["weight_decay"])
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    train_sequences = make_sequences(train_poems, tokenizer)
    valid_sequences = make_sequences(valid_poems, tokenizer)
    settings.setdefault("train_eval_size", args.train_eval_size)
    settings.setdefault("train_eval_seed", args.seed)
    train_eval_indices = select_train_eval_indices(
        len(train_sequences), settings["train_eval_size"], settings["train_eval_seed"],
        checkpoint.get("train_eval_indices") if args.resume else None)
    train_eval_sequences = train_sequences[train_eval_indices]
    if args.resume and "train_eval_indices" not in checkpoint:
        print("旧检查点无训练评估索引：建立固定子集；旧 train_loss 历史保留，新记录使用 train_eval_loss。")
    print(f"固定训练评估子集: {len(train_eval_sequences)}/{len(train_sequences)}（仍参与训练）")
    print(f"训练集: {len(train_poems)}，验证集: {len(valid_poems)}，词表: {len(tokenizer.tokens)}")
    print(f"检查点输出目录: {args.output_dir.resolve()}")
    if start_epoch > args.epochs:
        raise ValueError(f"--epochs 是总轮数，恢复训练时必须大于已有轮数 {start_epoch - 1}")
    settings.setdefault("warmup_ratio", args.warmup_ratio)
    settings.setdefault("min_lr_ratio", args.min_lr_ratio)
    planned_steps = (args.epochs - start_epoch + 1) * math.ceil(
        len(train_sequences) / settings["batch_size"])
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps=planned_steps,
        warmup_ratio=settings["warmup_ratio"], min_lr_ratio=settings["min_lr_ratio"])
    if args.resume:
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print("恢复已保存的学习率曲线；超出原计划步数后保持最低学习率。")
        else:
            print("旧检查点无调度状态：从当前学习率开始，为剩余训练建立 warmup + 余弦曲线。")
    info = scheduler.get_info()
    print(f"学习率策略: warmup={info['warmup_steps']} 步 | "
          f"total={info['total_steps']} 步 | peak={info['peak_lrs'][0]:.6g} | "
          f"min={info['min_lrs'][0]:.6g} | completed={info['completed_steps']} | "
          f"phase={info['phase']}")
    if args.resume and args.output_dir.resolve() != args.resume.resolve().parent:
        source_best = args.resume.resolve().parent / "best.pt"
        if not source_best.is_file():
            raise ValueError("恢复到新目录需要原目录的 best.pt；请保留原输出目录继续训练")
        best_checkpoint = torch.load(source_best, map_location="cpu", weights_only=False)
        if (best_checkpoint.get("epoch") != best_epoch
                or best_checkpoint.get("tokens") != tokenizer.tokens
                or best_checkpoint.get("best_loss") != best_loss):
            raise ValueError("原目录 best.pt 与恢复检查点不匹配，请使用原输出目录")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_best, args.output_dir / "best.pt")
    for epoch in range(start_epoch, args.epochs + 1):
        lr_start = scheduler.get_last_lr()[0]
        objective = run_epoch(model, train_sequences, settings["batch_size"], device,
                              optimizer, settings["label_smoothing"], scheduler)
        train_eval_loss = run_epoch(model, train_eval_sequences, settings["batch_size"], device)
        valid_loss = run_epoch(model, valid_sequences, settings["batch_size"], device)
        improved = valid_loss < best_loss - settings["min_delta"]
        if improved:
            best_loss, best_epoch, bad_epochs = valid_loss, epoch, 0
        elif epoch >= settings["min_epochs"]:
            bad_epochs += 1
        history.append(dict(epoch=epoch, objective=objective,
                            train_eval_loss=train_eval_loss, valid_loss=valid_loss,
                            lr_start=lr_start, lr_next=scheduler.get_last_lr()[0]))
        metadata = dict(epoch=epoch, best_loss=best_loss, best_epoch=best_epoch,
                        bad_epochs=bad_epochs, history=history, train_settings=settings,
                        train_poems=train_poems, valid_poems=valid_poems,
                        scheduler_state=scheduler.state_dict(),
                        train_eval_indices=train_eval_indices)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", model, tokenizer, optimizer, **metadata)
        save_checkpoint(args.output_dir / "last.pt", model, tokenizer, optimizer, **metadata)
        print(f"Epoch {epoch:4d} | objective={objective:.4f} | train_eval={train_eval_loss:.4f} | "
              f"valid={valid_loss:.4f} | best={best_loss:.4f} (epoch {best_epoch}) | "
              f"lr_start={lr_start:.6g} | lr_next={scheduler.get_last_lr()[0]:.6g}", flush=True)
        if epoch >= settings["min_epochs"] and bad_epochs >= settings["patience"]:
            print("触发早停，推理请使用 best.pt。")
            break


@torch.inference_mode()
def generate(model, tokenizer, prompt="", max_new_tokens=25, temperature=0.8,
             top_k=30, structured=True):
    """默认约束五绝字数和标点；free-form 模式由模型自行生成 EOS。"""
    if not math.isfinite(temperature) or temperature <= 0 or top_k < 0 or max_new_tokens < 1:
        raise ValueError("temperature 必须为正数，top_k >= 0，max_new_tokens >= 1")
    prompt = "".join(prompt.split())
    unknown = sorted(set(prompt) - set(tokenizer.token_ids))
    if unknown:
        raise ValueError(f"输入含词表外字符: {''.join(unknown)}，请更换开头（低频字被映射为 UNK）")
    if len(prompt) + 1 > MAX_SEQ_LEN:
        raise ValueError(f"输入过长，最多 {MAX_SEQ_LEN - 1} 个字符")
    punctuation = {5: "，", 11: "。", 17: "，", 23: "。"}
    if structured:
        if len(prompt) > 24:
            raise ValueError("五绝开头不能超过 24 个字符（含标点）")
        for position, char in enumerate(prompt):
            if (position in punctuation and char != punctuation[position]) or (
                position not in punctuation and char in "，。"
            ):
                raise ValueError("开头格式不符，应为每句五字，标点依次为 ，。，。")
    ids = [tokenizer.bos] + tokenizer.encode(prompt, special_tokens=False)
    device = next(model.parameters()).device
    model.eval()
    for _ in range(max_new_tokens):
        position = len(ids) - 1
        if len(ids) > MAX_SEQ_LEN:
            break
        if structured and position >= 24:
            break
        if structured and position in punctuation:
            next_id = tokenizer.token_ids[punctuation[position]]
        else:
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(inputs)[0, -1].clone() / temperature
            logits[[tokenizer.bos, tokenizer.unk]] = float("-inf")
            if structured:
                logits[[tokenizer.eos, tokenizer.token_ids["，"], tokenizer.token_ids["。"]]] = float("-inf")
            if top_k:
                values, indices = torch.topk(logits, min(top_k, logits.numel()))
                filtered = torch.full_like(logits, float("-inf"))
                filtered[indices] = values
                logits = filtered
            if not torch.isfinite(logits).any():
                raise ValueError("词表中没有可生成的字符，请降低 --min-freq 后重新训练")
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        if next_id == tokenizer.eos:
            break
        ids.append(next_id)
    return tokenizer.decode(ids)


def infer(args, device):
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    print(f"已加载第 {checkpoint['epoch']} 轮参数，验证损失 {checkpoint['history'][-1]['valid_loss']:.4f}")

    def show(prompt):
        for _ in range(args.num_samples):
            text = generate(model, tokenizer, prompt, args.max_new_tokens,
                            args.temperature, args.top_k, not args.free_form)
            print(text.replace("，", "，\n").replace("。", "。\n"))

    if args.interactive:
        print("输入诗歌开头（直接回车为随机作诗，输入 /quit 退出）")
        while True:
            try:
                prompt = input("开头> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip() == "/quit":
                break
            try:
                show(prompt)
            except ValueError as error:
                print(f"输入错误: {error}")
    else:
        show(args.prompt)


def build_parser():
    parser = argparse.ArgumentParser(description="MiniGPT 五言绝句训练与推理")
    commands = parser.add_subparsers(dest="command")
    training = commands.add_parser("train", help="训练；直接运行脚本也会进入训练")
    training.add_argument("--data", type=Path, default=DEFAULT_DATA)
    training.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    training.add_argument("--resume", type=Path, help="从 last.pt 恢复；复用保存的数据划分与训练参数")
    training.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="目标总轮数，不是新增轮数")
    training.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    training.add_argument("--lr", type=float, default=LEARNING_RATE, help="warmup 后的峰值学习率")
    training.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO,
                          help="线性预热占计划训练步数的比例 [0,1)，默认 0.05；0 禁用预热")
    training.add_argument("--min-lr-ratio", type=float, default=MIN_LR_RATIO,
                          help="最低学习率 / 峰值学习率 [0,1]，默认 0.1")
    training.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                          help="AdamW 权重衰减，默认 0.01；恢复训练时复用检查点中的值")
    training.add_argument("--train-eval-size", type=int, default=TRAIN_EVAL_SIZE,
                          help="固定训练评估子集大小（首），默认 1024；0 表示全量，恢复时复用保存值")
    training.add_argument("--valid-ratio", type=float, default=0.1)
    training.add_argument("--min-freq", type=int, default=MAX_TOKEN_FREQ)
    training.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    training.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    training.add_argument("--max-poems", type=int, default=0, help="仅用部分数据做冒烟测试，0 表示全部")
    inference = commands.add_parser("infer", help="加载参数生成诗歌")
    inference.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT / "best.pt")
    inference.add_argument("--prompt", default="春", help="诗歌开头，而非聊天指令；空串表示随机生成")
    inference.add_argument("--interactive", action="store_true")
    inference.add_argument("--num-samples", type=int, default=1)
    inference.add_argument("--temperature", type=float, default=0.8)
    inference.add_argument("--top-k", type=int, default=30, help="0 表示不做 top-k 截断")
    inference.add_argument("--max-new-tokens", type=int, default=25)
    inference.add_argument("--free-form", action="store_true", help="不约束五绝格式，由模型生成标点和结束符")
    for command in (training, inference):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv if argv else ["train"])
    if args.command == "train":
        if (args.epochs < 1 or args.batch_size < 1 or args.min_freq < 1
                or args.patience < 1 or args.min_epochs < 0 or args.train_eval_size < 0
                or not math.isfinite(args.lr) or args.lr <= 0
                or not math.isfinite(args.warmup_ratio) or not 0 <= args.warmup_ratio < 1
                or not math.isfinite(args.min_lr_ratio) or not 0 <= args.min_lr_ratio <= 1
                or not math.isfinite(args.weight_decay) or args.weight_decay < 0
                or not 0 < args.valid_ratio < 1
                or args.max_poems < 0 or args.max_poems == 1):
            parser.error("训练参数无效：轮数/批次/词频/耐心值须为正，train-eval-size 须非负，weight-decay 须非负且有限，warmup-ratio 在 [0,1)，min-lr-ratio 在 [0,1]，验证比例在 (0,1)，max-poems 为 0 或 >=2")
    else:
        if (args.num_samples < 1 or args.max_new_tokens < 1 or args.top_k < 0
                or not math.isfinite(args.temperature) or args.temperature <= 0):
            parser.error("推理参数无效：样本数/生成长度/温度须为正，top-k >= 0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请使用 --device cpu 或检查 PyTorch 环境")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"运行设备: {device}")
    if device.type == "cuda":
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    try:
        train(args, device) if args.command == "train" else infer(args, device)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
