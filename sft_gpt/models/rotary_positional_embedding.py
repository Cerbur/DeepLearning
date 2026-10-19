import torch
from torch import nn

from ..config import ROPE_BASE


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
