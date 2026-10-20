import torch
from torch import nn

from .precision import promote_for_stability


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps  # 避免除以零
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x):
        # 不依赖输入或 autocast 的隐式提升：平方、归约和缩放保留 FP32。
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = promote_for_stability(x)
            weight = promote_for_stability(self.weight)
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return weight * (x / rms)
