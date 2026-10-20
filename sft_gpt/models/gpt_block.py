import torch
from torch import nn

from ..config import D_MODEL, DROPOUT_RATE
from .multi_head_attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .swiglu import SwiGLU
from .precision import promote_for_stability


class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(D_MODEL)  # 第一个归一化层
        self.attention = MultiHeadAttention()  # 多头自注意力
        self.norm2 = RMSNorm(D_MODEL)  # 第二个归一化层
        self.ffn = SwiGLU()  # 前馈网络
        self.dropout = nn.Dropout(DROPOUT_RATE)  # Dropout 层

    def forward(self, x):
        x = promote_for_stability(x)
        attention_output = self.attention(self.norm1(x))
        # 分支可用 BF16；dropout 和残差累加显式保留 FP32 主干。
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x + self.dropout(promote_for_stability(attention_output))
        ffn_output = self.ffn(self.norm2(x))
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x + self.dropout(promote_for_stability(ffn_output))
        return x
