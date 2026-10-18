import sys
import random
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 确保在 Windows 控制台输出中文不乱码
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==========================================
# 1. 硬件设备配置（优先使用 GPU 加速）
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")
if device.type == "cuda":
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")

# 固定随机种子，保证结果可复现
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================
# 2. 超参数设置
# ==========================================
EMBEDDING_DIM = 64          # 词嵌入维度 (Embedding Dimension)
HIDDEN_DIM = 128            # GRU 隐层状态维度 (Hidden Dimension)
BATCH_SIZE = 16             # 批次大小
LEARNING_RATE = 0.003       # 学习率
EPOCHS = 60                 # 训练轮数
TEACHER_FORCING_RATIO = 0.5  # 训练时使用真实标签作为下一步输入的概率

# 4个标准特殊标记
PAD_TOKEN = "<pad>"  # 补齐长度用
UNK_TOKEN = "<unk>"  # 未登录词 / 词表中不存在的词
SOS_TOKEN = "<sos>"  # 句子起始符 (Start Of Sentence)
EOS_TOKEN = "<eos>"  # 句子结束符 (End Of Sentence)

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# ==========================================
# 3. 准备英文-中文平行语料 (Toy Dataset)
# ==========================================
# 入门演示使用经典的双语平行短句
RAW_DATA = [
    ("hello", "你好"),
    ("hello world", "你好世界"),
    ("good morning", "早上好"),
    ("good afternoon", "下午好"),
    ("good night", "晚安"),
    ("how are you", "你好吗"),
    ("i am fine", "我很好"),
    ("thank you", "谢谢你"),
    ("thanks", "谢谢"),
    ("you are welcome", "不客气"),
    ("i am a student", "我是学生"),
    ("i am a teacher", "我是老师"),
    ("he is a doctor", "他是医生"),
    ("she is a nurse", "她是护士"),
    ("i love programming", "我爱编程"),
    ("i like deep learning", "我喜欢深度学习"),
    ("i love china", "我爱中国"),
    ("i like reading", "我喜欢读书"),
    ("i like apples", "我喜欢苹果"),
    ("i like dogs", "我喜欢狗"),
    ("i love cats", "我爱猫"),
    ("what is your name", "你叫什么名字"),
    ("my name is tom", "我的名字是汤姆"),
    ("where are you from", "你来自哪里"),
    ("i am from beijing", "我来自北京"),
    ("see you tomorrow", "明天见"),
    ("goodbye", "再见"),
    ("have a nice day", "祝你有美好的一天"),
    ("can you help me", "你能帮我吗"),
    ("no problem", "没问题"),
    ("i am hungry", "我饿了"),
    ("i want to eat", "我想吃东西"),
    ("i am tired", "我累了"),
    ("it is cold today", "今天很冷"),
    ("it is sunny today", "今天天气晴朗"),
    ("welcome to china", "欢迎来到中国"),
    ("this is a book", "这是一本书"),
    ("that is a pen", "那是一支笔"),
    ("time is money", "时间就是金钱"),
    ("practice makes perfect", "熟能生巧"),
]

# ==========================================
# 4. 分词器与词表构建 (Tokenizer & Vocab)
# ==========================================
def tokenize_en(text: str):
    """英文简易分词：转小写，提取单词与标点"""
    text = text.lower().strip()
    return re.findall(r"\w+|[^\w\s]", text)

def tokenize_zh(text: str):
    """中文分词：采用字符级 (Character-level) 切分，无需额外第三方依赖且极其轻量高效"""
    return [char for char in text.strip() if not char.isspace()]

class Vocabulary:
    """词表管理类：维护 token <-> id 的双向映射"""
    def __init__(self, name: str):
        self.name = name
        self.token2idx = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
        }
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.count = 4

    def build_vocab(self, tokenized_sentences):
        for tokens in tokenized_sentences:
            for token in tokens:
                if token not in self.token2idx:
                    self.token2idx[token] = self.count
                    self.idx2token[self.count] = token
                    self.count += 1

    def encode(self, tokens, add_sos=False, add_eos=False):
        """将 token 序列转化为 id 列表"""
        indices = []
        if add_sos:
            indices.append(SOS_IDX)
        for token in tokens:
            indices.append(self.token2idx.get(token, UNK_IDX))
        if add_eos:
            indices.append(EOS_IDX)
        return indices

    def decode(self, indices):
        """将 id 列表还原为 token 序列（自动跳过 pad 与 sos，遇到 eos 截断）"""
        tokens = []
        for idx in indices:
            if idx in (SOS_IDX, PAD_IDX):
                continue
            if idx == EOS_IDX:
                break
            tokens.append(self.idx2token.get(idx, UNK_TOKEN))
        return tokens

    def __len__(self):
        return len(self.token2idx)

# 构建源语言(英文)与目标语言(中文)词表
src_sentences = [tokenize_en(en) for en, _ in RAW_DATA]
tgt_sentences = [tokenize_zh(zh) for _, zh in RAW_DATA]

src_vocab = Vocabulary("English")
tgt_vocab = Vocabulary("Chinese")
src_vocab.build_vocab(src_sentences)
tgt_vocab.build_vocab(tgt_sentences)

print(f"英文词表大小: {len(src_vocab)} | 中文词表大小: {len(tgt_vocab)}")

# ==========================================
# 5. 数据集与动态填充 DataLoader
# ==========================================
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.samples = []
        for en, zh in data:
            src_tokens = tokenize_en(en)
            tgt_tokens = tokenize_zh(zh)
            # Encoder 端：输入尾部附带 <eos>
            src_ids = src_vocab.encode(src_tokens, add_sos=False, add_eos=True)
            # Decoder 端：输入包含 <sos>，结束包含 <eos>
            tgt_ids = tgt_vocab.encode(tgt_tokens, add_sos=True, add_eos=True)
            self.samples.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """对同一个 batch 内长短不一的句子用 PAD_IDX 进行右侧对齐填充"""
    src_list, tgt_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded

dataset = TranslationDataset(RAW_DATA, src_vocab, tgt_vocab)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ==========================================
# 6. 网络架构定义：Seq2Seq (Embedding + GRU)
# ==========================================
class Encoder(nn.Module):
    """
    编码器 (Encoder)：
    输入: 英文序列 IDs -> (batch_size, src_len)
    过程:
      1. Embedding 层映射为密集向量 -> (batch_size, src_len, emb_dim)
      2. pack_padded_sequence 忽略末尾的 <pad>，确保获取的是句子实际结尾处的隐藏状态
      3. GRU 逐时步计算更新隐状态
    输出:
      最终步的隐状态 hidden -> (1, batch_size, hidden_dim)，作为整个句子的语义上下文向量 (Context Vector)
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        # src: (batch_size, src_len)
        embedded = self.embedding(src)  # (batch_size, src_len, emb_dim)
        # 统计每个样本的实际长度（非 PAD 数量）
        lengths = (src != PAD_IDX).sum(dim=1).clamp(min=1).cpu()
        # 压包：使 GRU 自动忽略 padding 部分，保证 hidden 是实际句尾的状态
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed_embedded)
        return hidden

class Decoder(nn.Module):
    """
    解码器 (Decoder)：
    输入: 单个目标词 ID -> (batch_size, 1) 以及上一步的隐状态 hidden -> (1, batch_size, hidden_dim)
    过程:
      1. Embedding 映射 -> (batch_size, 1, emb_dim)
      2. GRU 结合上一步隐状态计算得到新隐状态与 output -> (batch_size, 1, hidden_dim)
      3. Linear 投影到目标语言词表大小 -> (batch_size, vocab_size)
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden):
        # input_token: (batch_size, 1)
        # hidden: (1, batch_size, hidden_dim)
        embedded = self.embedding(input_token)   # (batch_size, 1, emb_dim)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))  # (batch_size, vocab_size)
        return prediction, hidden

class Seq2Seq(nn.Module):
    """
    Seq2Seq 组合模型：
    - 将源语言送入 Encoder 获得语义上下文向量 Context Vector
    - 解码器以 Context Vector 作为初始隐藏状态，从 <sos> 开始逐词解码
    - 训练时引入 Teacher Forcing：以一定概率用真实标签替代预测值作为下一时刻输入，加速收敛
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = len(tgt_vocab)

        # 预先分配张量保存每一时刻的输出 Logits
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # 1. 编码器编码，得到最终隐状态
        hidden = self.encoder(src)

        # 2. 解码器从第一个词开始（即 <sos> 标记）
        dec_input = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)

        # 3. 循环解码每一个时刻 t (从位置 1 到 tgt_len-1)
        for t in range(1, tgt_len):
            prediction, hidden = self.decoder(dec_input, hidden)
            outputs[:, t, :] = prediction

            # 决定下一时刻的输入是真实 token 还是当前预测的最大概率 token
            is_teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(dim=1).unsqueeze(1)
            dec_input = tgt[:, t].unsqueeze(1) if is_teacher_force else top1

        return outputs

# 实例化模型并迁移至目标设备 (GPU / CPU)
encoder = Encoder(len(src_vocab), EMBEDDING_DIM, HIDDEN_DIM)
decoder = Decoder(len(tgt_vocab), EMBEDDING_DIM, HIDDEN_DIM)
model = Seq2Seq(encoder, decoder, device).to(device)

print(f"模型参数总量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ==========================================
# 7. 损失函数与优化器
# ==========================================
# 使用交叉熵损失函数，忽略 <pad> 填充符位置
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
# 使用 AdamW 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 8. 训练循环
# ==========================================
print("\n开始训练 Seq2Seq 翻译模型...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for src_batch, tgt_batch in train_loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        optimizer.zero_grad()

        # 前向传播 (计算 logits)
        output = model(src_batch, tgt_batch, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

        # 目标与预测均切片跳过第一个位置 <sos>，展平后计算 CrossEntropyLoss
        output_dim = output.shape[-1]
        loss_output = output[:, 1:].reshape(-1, output_dim)
        loss_target = tgt_batch[:, 1:].reshape(-1)

        loss = criterion(loss_output, loss_target)

        # 反向传播与梯度裁剪（防止 RNN 梯度爆炸）
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - 平均损失 (Loss): {avg_loss:.4f}")

# ==========================================
# 9. 贪婪解码推理函数 (Greedy Decoding)
# ==========================================
def translate(model, sentence: str, max_len: int = 25):
    """
    推理函数：
    1. 输入英文句子分词并转为 Tensor
    2. Encoder 提取整句语义特征 (hidden)
    3. Decoder 从 <sos> 开始自回归逐步生成中文字，直至遇到 <eos> 或达到最大长度
    """
    model.eval()
    tokens = tokenize_en(sentence)
    src_ids = src_vocab.encode(tokens, add_sos=False, add_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, src_len)

    with torch.no_grad():
        hidden = model.encoder(src_tensor)
        dec_input = torch.tensor([[SOS_IDX]], dtype=torch.long).to(device)  # (1, 1)

        translated_ids = []
        for _ in range(max_len):
            prediction, hidden = model.decoder(dec_input, hidden)
            pred_id = prediction.argmax(dim=1).item()

            if pred_id == EOS_IDX:
                break
            translated_ids.append(pred_id)
            dec_input = torch.tensor([[pred_id]], dtype=torch.long).to(device)

    # 还原为中文字符
    chars = tgt_vocab.decode(translated_ids)
    return "".join(chars)

# ==========================================
# 10. 翻译效果展示
# ==========================================
print("\n================ 翻译测试结果 ================")
demo_sentences = [
    "Hello world",
    "Good morning",
    "How are you",
    "I like deep learning",
    "I love programming",
    "I like apples",
    "I love china",
    "Time is money",
    "Practice makes perfect",
    "What is your name",
    "See you tomorrow",
]

for s in demo_sentences:
    res = translate(model, s)
    print(f"EN: {s:<26} -> ZH: {res}")
