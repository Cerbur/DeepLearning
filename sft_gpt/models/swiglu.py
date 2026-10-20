import torch
from torch import nn

from .precision import promote_for_stability

from ..config import D_MODEL, D_FF


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FF, bias=False)  # 门控线性变换
        self.up_proj = nn.Linear(D_MODEL, D_FF, bias=False)  # 上采样线性变换
        self.down_proj = nn.Linear(D_FF, D_MODEL, bias=False)  # 下采样线性变换
        self.activation = nn.SiLU()  # SiLU 激活函数

    def forward(self, x):
        # 三个 Linear 由外层 BF16 autocast 加速；非线性和门控乘法使用 FP32。
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            gated = self.activation(promote_for_stability(up)) * promote_for_stability(gate)
        return self.down_proj(gated)
