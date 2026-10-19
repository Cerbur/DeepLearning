from torch import nn

from ..config import D_MODEL
from .gpt_block import GPTBlock
from .rms_norm import RMSNorm


class GPTDecoder(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([GPTBlock() for _ in range(num_layers)])  # 堆叠多个 GPTBlock
        self.norm = RMSNorm(D_MODEL)  # 最后的归一化层

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # 逐层前向传播
        return self.norm(x)  # 最后归一化输出
