"""模型常量保持原值；数据及输出路径集中配置。"""
from pathlib import Path

# 模型参数
D_MODEL = 768  # 模型维度
NUM_HEADS = 12  # 注意力头数
D_FF = 2048  # 前馈网络维度
NUM_LAYERS = 13  # Transformer 层数
MAX_SEQ_LEN = 512  # 最大序列长度
ROPE_BASE = 10000.0  # Rotary Positional Embedding 基数
DROPOUT_RATE = 0.1
MAX_TOKEN_FREQ = 5
# 训练参数
BATCH_SIZE = 4 # micro-batch；有效 batch = 此值 * GRAD_ACCUM_STEPS
GRAD_ACCUM_STEPS = 16
MAX_STEPS = 10000 # 新实验默认优化器更新预算；0 为只受 epochs 限制
EVAL_BATCH_SIZE = 4
SAVE_EVERY_STEPS = 250
EVAL_EVERY_STEPS = 250
MAX_EPOCHS = 1000 # 最大训练轮数
MIN_EPOCHS = 10 # 最小训练轮数
LEARNING_RATE = 0.0003 # warmup 后的峰值学习率
WARMUP_RATIO = 0.05 # 前 5% 的训练步数线性预热
MIN_LR_RATIO = 0.1 # 余弦退火最低学习率 / 峰值学习率
EARLY_STOPPING_PATIENCE = 10 # 早停策略的耐心值
EARLY_STOPPING_DELTA = 0.0001 # 早停策略的最小改进阈值
WEIGHT_DECAY = 0.1 # 仅矩阵权重衰减，norm/1D 参数不衰减
LABEL_SMOOTHING = 0.0 # 标准 causal LM objective
ADAM_BETA2 = 0.95
TRAIN_EVAL_SIZE = 1024 # 固定训练评估子集大小，0 表示完整训练集

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / ".data" / "five_char_quatrains.tsv"  # 旧五绝数据 API
DEFAULT_TRAIN_DATA = BASE_DIR / ".data" / "corpus" / "train"
DEFAULT_EVAL_DATA = BASE_DIR / ".data" / "corpus" / "eval"
DEFAULT_OUTPUT = BASE_DIR / "checkpoints" / "pre_train" / "lccc"
DEFAULT_VOCAB = BASE_DIR / ".data" / "tokenizer" / "vocab.json"  # 旧字符词表
DEFAULT_TEXT_VOCAB = BASE_DIR / ".data" / "tokenizer" / "text_vocab.json"
DEFAULT_TOKENIZER_MIN_FREQ = 2
DEFAULT_VOCAB_SIZE = 8192
MODEL_CONFIG = {name: globals()[name] for name in (
    "D_MODEL", "NUM_HEADS", "D_FF", "NUM_LAYERS", "MAX_SEQ_LEN",
    "ROPE_BASE", "DROPOUT_RATE",
)}
