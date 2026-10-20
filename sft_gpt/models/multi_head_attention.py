# import math  # 原手写注意力使用；SDPA 内部完成缩放
# import torch  # 原手写实现构造掩码和计算 softmax 时使用

from torch import nn
from torch.nn import functional as F

from ..config import D_MODEL, NUM_HEADS, MAX_SEQ_LEN
from .rotary_positional_embedding import RotaryPositionalEmbedding


# def attention(q, k, v):
#     # [B, H, L, D] [L,D]
#     scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # 计算注意力分数
#
#     seq_len = q.shape[-2]
#
#     # 创建上三角掩码，防止模型看到未来的 token
#     mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
#
#     scores = scores.masked_fill(mask, float("-inf"))  # 将上三角部分置为负无穷，softmax 后为 0
#
#     weights = torch.softmax(scores, dim=-1)  # 计算注意力权重
#
#     return weights @ v  # 返回加权后的值


def attention(q, k, v):
    # SDPA 等价于上方的 softmax(QK^T / sqrt(head_dim) + 因果掩码) @ V。
    # is_causal=True 禁止关注未来 token，无需每次手动构造 [L, L] 掩码。
    # 原实现没有注意力权重 dropout；保留 0.0，GPTBlock 的输出 dropout 不变。
    # PyTorch 根据设备、dtype 和形状自动选择可用的融合内核或数学实现。
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


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

        # q = self.rope(q)
        # k = self.rope(k)
        # RoPE 在显式 FP32 区域完成旋转；
        # 转回 V 的 dtype，使 SDPA 的 Q/K/V 类型一致，并支持低精度融合内核。
        q = self.rope(q).to(dtype=v.dtype)
        k = self.rope(k).to(dtype=v.dtype)

        context = attention(q, k, v)  # 计算注意力输出

        # 将 [B, H, L, head_dim] -> [B, L, H, head_dim] -> [B, L, D_MODEL]
        joined = context.transpose(1, 2).contiguous().reshape(batch_size, seq_len, D_MODEL)  # 将多头注意力输出拼接回原维度
        output = self.out_proj(joined)  # 输出线性变换

        return output.squeeze(0) if unbatched else output  # 如果是单个序列，去掉 batch 维度
