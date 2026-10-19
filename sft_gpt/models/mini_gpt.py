from torch import nn

from ..config import D_MODEL, NUM_LAYERS
from .token_embedding import TokenEmbedding
from .gpt_decoder import GPTDecoder


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
