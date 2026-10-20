"""模型常量保持原值；数据及输出路径集中配置。"""
from pathlib import Path

# 模型参数
D_MODEL = 288  # 模型维度
NUM_HEADS = 6  # 注意力头数
D_FF = 768  # 前馈网络维度
NUM_LAYERS = 8  # Transformer 层数
MAX_SEQ_LEN = 512  # 最大序列长度
ROPE_BASE = 10000.0  # Rotary Positional Embedding 基数
DROPOUT_RATE = 0.1
MAX_TOKEN_FREQ = 5
# 训练参数
BATCH_SIZE = 32 # 训练批次大小
MAX_EPOCHS = 1000 # 最大训练轮数
MIN_EPOCHS = 10 # 最小训练轮数
LEARNING_RATE = 0.002 # warmup 后的峰值学习率
WARMUP_RATIO = 0.05 # 前 5% 的训练步数线性预热
MIN_LR_RATIO = 0.1 # 余弦退火最低学习率 / 峰值学习率
EARLY_STOPPING_PATIENCE = 10 # 早停策略的耐心值
EARLY_STOPPING_DELTA = 0.0001 # 早停策略的最小改进阈值
WEIGHT_DECAY = 0.01 # 权重衰减
LABEL_SMOOTHING = 0.1 # 标签平滑
TRAIN_EVAL_SIZE = 1024 # 固定训练评估子集大小，0 表示完整训练集

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / ".data" / "five_char_quatrains.tsv"  # 旧五绝数据 API
DEFAULT_TRAIN_DATA = BASE_DIR / ".data" / "corpus" / "train"
DEFAULT_EVAL_DATA = BASE_DIR / ".data" / "corpus" / "eval"
DEFAULT_OUTPUT = BASE_DIR / "checkpoints" / "pre_train"
DEFAULT_VOCAB = BASE_DIR / ".data" / "tokenizer" / "vocab.json"  # 旧字符词表
DEFAULT_TEXT_VOCAB = BASE_DIR / ".data" / "tokenizer" / "text_vocab.json"
DEFAULT_TOKENIZER_MIN_FREQ = 2
DEFAULT_VOCAB_SIZE = 8192
MODEL_CONFIG = {name: globals()[name] for name in (
    "D_MODEL", "NUM_HEADS", "D_FF", "NUM_LAYERS", "MAX_SEQ_LEN",
    "ROPE_BASE", "DROPOUT_RATE",
)}
