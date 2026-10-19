from torch import nn

from ..config import D_MODEL, DROPOUT_RATE
from .multi_head_attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .swiglu import SwiGLU


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
