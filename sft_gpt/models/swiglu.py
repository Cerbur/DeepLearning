from torch import nn

from ..config import D_MODEL, D_FF


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FF, bias=False)  # 门控线性变换
        self.up_proj = nn.Linear(D_MODEL, D_FF, bias=False)  # 上采样线性变换
        self.down_proj = nn.Linear(D_FF, D_MODEL, bias=False)  # 下采样线性变换
        self.activation = nn.SiLU()  # SiLU 激活函数

    def forward(self, x):
        return self.down_proj(self.activation(self.up_proj(x)) * self.gate_proj(x))  # SwiGLU FFN
