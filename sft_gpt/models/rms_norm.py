import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps  # 避免除以零
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 计算均方根
        return self.weight * (x / rms)  # 归一化并缩放
