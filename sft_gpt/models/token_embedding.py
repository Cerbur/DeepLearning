from torch import nn

from ..config import D_MODEL, DROPOUT_RATE


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        # 初始化嵌入层权重
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # train() 方法中启用 Dropout，eval() 方法中禁用 Dropout
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, token_ids):
        return self.dropout(self.embedding(token_ids))
