# SFT GPT：通用文本预训练工程

当前实现的是 **pre train（自回归预训练）**：普通文本、日常交流语料和对话记录都可以参与训练。不再默认筛选五言绝句，也不默认按诗词格式生成。**这不是 assistant-only SFT，也不是已经具备日常聊天能力的模型。**

## 1. 保持不变的模型与环境

九个模型实现保持原样：MiniGPT、GPTDecoder、GPTBlock、SwiGLU、RMSNorm、MultiHeadAttention、attention、RotaryPositionalEmbedding、TokenEmbedding。仅扩展模型之外的数据、分词、训练损失补齐处理、检查点和推理入口。源码与 AST、前向结果和梯度对照见测试。

原模型常量仍为 D_MODEL=96、NUM_HEADS=6、D_FF=256、NUM_LAYERS=4、MAX_SEQ_LEN=128、ROPE_BASE=10000、DROPOUT_RATE=0.1。`--seq-len` 只决定数据打包长度，不能超过 128，也不改变模型结构。

所有命令在仓库根目录 `E:\Code\DeepLearning` 执行。沿用现有 uv 环境，**没有安装或升级依赖**。

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv_cache"
uv run --project sft_gpt --no-sync python -m sft_gpt --help
```

本机为 Python 3.12.15、PyTorch 2.14.1+cu132、CUDA Runtime 13.2。环境在 `sft_gpt/.venv`，旧 `mini_gpt/.venv` 是兼容 Windows 启动器的 Junction；不要删除此连接及仓库根 `.python`。IDE 使用 `sft_gpt/.venv/Scripts/python.exe`。PowerShell 不允许未签名激活脚本时，直接运行 uv 即可，无需修改系统策略。后续依赖安装继续遵循 AGENTS 的 Conda 优先规范，不混改这两套环境。

## 2. 通用语料格式

`--data` 接受单个文件或目录；目录按相对路径排序递归读取 `.txt/.jsonl/.json/.csv/.tsv`。**请只传语料专用目录，不要把 tokenizer JSON、检查点输出或整个项目根目录混进去。** UTF-8，可带 BOM；不做大小写转换、繁简转换、Unicode 归一化或全局去空白。

### TXT：空行分隔文档

```text
今天下班以后，想做一顿简单的晚饭。
可以煮面，也可以炒青菜。

Learning a little every day helps.
```

保留非空行的前后空格及原始换行（包括 CRLF），空白行只用于分隔文档。如果需要把一整段多行对话作为一个文档，建议用 JSONL messages，避免空行把对话拆开。

### JSONL：每行一条文本或完整对话

```jsonl
{"text":"今天有点累，想早点休息。"}
{"text":"Hello world! 中文和 English 都可以，包含 emoji 😊。"}
{"text":"保留两行：\n第一行和第二行。"}
{"messages":[{"role":"system","content":"自然、礼貌地交流。"},{"role":"user","content":"晚上吃什么？"},{"role":"assistant","content":"你想吃清淡一点，还是热乎的饭菜？"}]}
```

messages 支持 `system/user/assistant` 与字符串 content，整段对话作为一个文档，按固定模板转为：

```text
system: 自然、礼貌地交流。
user: 晚上吃什么？
assistant: 你想吃清淡一点，还是热乎的饭菜？
```

这只是预训练文本模板，role 是普通文本，不是特殊控制 token。所有文本位置都参与 next-token loss，**没有只对 assistant 内容计算损失**。消息内容中写出的 `<BOS>` 等拼写不会被解析成控制 token。

JSON 接受记录列表或单条记录；CSV/TSV 默认读取 `text` 列，`--text-field content` 可指定字段名。坏格式会报文件路径和行号，不会悄悄跳过。仅删除纯空记录和完全相同的文档，不做近重复检测；近重复、同源文章或同一人的多段记录应在数据准备阶段按来源分组隔离。至少需要两个独立文档。

## 3. 默认 tokenizer：高频字符 + UTF-8 字节回退

默认 `--tokenizer hybrid`，不是 BPE，也不是逐词分词：

| 内容 | 编码方式 |
|---|---|
| 训练集中的高频中文及其他非 ASCII 字符 | 一个字符一个 token |
| ASCII、空格、换行、制表符 | 固定字节 token |
| 生僻字、验证集才出现的新字符、未学到的 emoji | UTF-8 字节序列，不映射 UNK |

BOS=0、EOS=1、UNK=2、PAD=3；ID 4..259 为完整 256 字节，之后是学习到的字符。默认 `--min-freq 2`、`--vocab-size 8192`（**上限，不保证凑满**，包含 260 个基础项）。高频字符按频次挑选，同频按字符排序；最终按字符排序分配 ID。词表加载绝不排序重建。

相比纯字节方案，常见中文通常仍用 1 token；低频中文需 3 个字节 token，多数 emoji 需 4 个。相比旧字符词表，新验证字符不再全部折叠成 UNK。英文目前仍是字节级，**不声称达到成熟 BPE 的压缩效率**。对合法 Unicode 文本，编码后解码保留原文，包括组合字符、空白和 emoji。

### 划分与持久化

1. 先按完整文档去重、固定种子打乱，按 `--valid-ratio` 划分训练／验证。整段 messages 不会跨集合拆分。
2. **只用训练集统计高频字符**，不让验证集决定词表。验证集中没见过的字符由固定 UTF-8 字节覆盖。
3. 新 hybrid 默认词表保存在 `.data/tokenizer/text_vocab.json`，不覆盖旧 `.data/tokenizer/vocab.json`。
4. 输出目录也保存 `vocab.json`；.pt 内嵌完整 tokenizer 类型、版本和有序词表，是模型编码方式的依据。
5. `--vocab` 只读复用指定词表，不重建、不回写；新全量训练可能覆盖默认 text_vocab.json，保留实验时请复制词表或指定独立输出目录。
6. JSON 严格校验类型、固定项、重复 token、版本、重复字段和非有限数字，保存用临时文件、fsync、原子替换；不存在或损坏不会静默建表。

```python
from sft_gpt import HybridTokenizer, load_tokenizer

# train_texts 是已划分出的训练文本，不是完整含验证集的语料。
train_texts = ["你好，今天怎么样？", "你好！今天很好。"]
tokenizer = HybridTokenizer.build(train_texts, min_freq=2, max_vocab_size=8192)
text = "未见字符𠮷 + emoji🦊\n  Hello"
ids = tokenizer.encode(text)
assert tokenizer.unk not in ids
assert tokenizer.decode(ids) == text
path = tokenizer.save()  # .data/tokenizer/text_vocab.json
restored = load_tokenizer(path)
assert restored.encode(text) == ids
```

旧 `CharTokenizer` API 和 JSON 保持兼容；`--tokenizer char` 可显式使用（仍可能出现 UNK）。**不能把旧模型直接换一份 hybrid 词表继续推理或续训**：ID、词表大小和词义不同，必须新建预训练实验。

## 4. 通用变长序列与 loss

每个文档编码为 BOS + 内容 + EOS。**训练与验证分别打包**成连续 token 流，窗口是 `seq_len+1`，步长为 seq_len，边界共享一个 token；输入与标签错开一位，流中每个可预测目标只出现一次。长文不会只留下前 128 token，尾部也保留。

- 同一 split 内可把不同文档拼到一个块，BOS/EOS 标识边界。因没有修改原 attention，**没有文档间 attention 隔离**，这是普通 packed causal-LM 预训练，不是逐对话独立 attention。
- 最后不足一块时仅右侧补齐为 -100。送 Embedding 前把负数换为安全 ID，标签保留 -100，CrossEntropyLoss 忽略补齐位置。有效 token 在补齐之前，原因果注意力无需新增 padding mask。
- 含补齐数据的平均 loss 按有效预测 token 数加权，不让短尾块与完整块等权。旧无 padding 五绝模式保留原统计与训练轨迹。
- 日志显示文档数、序列数、词表类型、有效 token、padding 和 UNK 比例。
- checkpoint 保存已划分文本、tokenizer、序列模式、seq_len、评估索引、优化器、调度器和 RNG。恢复时沿用这些设置，不用新的 CLI 参数重新切分或重建词表。

当前读取、去重和编码在内存中完成，检查点也保存文本以保证恢复一致；适合小模型实验，**不是十亿 token 级流式数据管线**。纯文本长度仍按 token 计，不按汉字数计。

## 5. uv 预训练命令

命令保留大小写敏感的 **preTrain**。默认语料路径仍是旧五绝 TSV，避免替换你的本地数据；要训练日常语料请显式传 `--data`。不传子命令也进入 preTrain。

```powershell
# 先用附带的极小语料跑通链路，不代表可训练出聊天能力
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --data sft_gpt/examples/pretrain_texts.jsonl --epochs 2 --batch-size 4 --seq-len 64 --output-dir sft_gpt/.data/text_smoke

# 正式使用自己的文本/对话语料目录
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --data sft_gpt/.data/corpus --tokenizer hybrid --vocab-size 8192 --min-freq 2 --seq-len 128 --output-dir sft_gpt/checkpoints/pre_train/daily

# 只读复用先前冻结的词表
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --data sft_gpt/.data/corpus --vocab sft_gpt/.data/tokenizer/text_vocab.json --output-dir sft_gpt/checkpoints/pre_train/daily_fixed

# 恢复：epochs 是目标总轮数；数据、序列配置、词表及训练设置取检查点
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --resume sft_gpt/checkpoints/pre_train/daily/last.pt --epochs 200 --output-dir sft_gpt/checkpoints/pre_train/daily
```

默认输出仍是 `checkpoints/pre_train/`，明确不是 SFT 产物，文件为 best.pt / last.pt / vocab.json，.pt 含 `training_stage="pre_train"`。保留原有 AdamW、warmup + 余弦退火、梯度裁剪、标签平滑、固定训练评估子集、早停行为。`--max-documents` 限制文档数便于冒烟，旧 `--max-poems` 是同义别名。

没有自动预留测试集；正式基准应准备独立测试文档，不用于建表或调参。更换 tokenizer、语料或打包策略后 loss 口径会变，不应直接和旧五绝实验的数值比较。

## 6. 普通文本推理与旧检查点

默认保留 prompt 空白、换行、emoji，不强制五字、不强加中文标点。

```powershell
uv run --project sft_gpt --no-sync python -m sft_gpt infer --checkpoint sft_gpt/.data/text_smoke/best.pt --prompt "你好，今天" --max-new-tokens 64

# 交互式前缀续写，/quit 退出；没有自动累积多轮历史
uv run --project sft_gpt --no-sync python -m sft_gpt infer --interactive --checkpoint sft_gpt/checkpoints/pre_train/daily/best.pt
```

推理每次只送最后 MAX_SEQ_LEN=128 token 给模型（滑动窗口，**没有扩大上下文**）；输出包含原 prompt。hybrid 生成时屏蔽非法 UTF-8 字节组合、BOS/UNK/PAD，末尾预算不足时不开始无法完成的多字节字符，避免截断乱码。`--max-new-tokens` 是 token 数，不是字符数。

兼容旧格式：无 tokenizer 描述的 version=1 检查点仍按 CharTokenizer 恢复，保留 ID；新字符检查点是 version=1 加显式 tokenizer，hybrid 检查点是 version=2。所有结构按 strict=True 加载。只能加载自己生成或可信的 .pt（torch.load 使用 pickle）。

旧实验不移动、不重写，恢复时自动保留旧逐首编码。要重跑原五绝流程或强制五绝格式，需明确指定 `--poetry`：

```powershell
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --poetry --min-freq 5 --data sft_gpt/.data/five_char_quatrains.tsv --output-dir sft_gpt/checkpoints/pre_train/poetry
uv run --project sft_gpt --no-sync python -m sft_gpt infer --poetry --checkpoint mini_gpt/checkpoints/best.pt --prompt 春
```

`--free-form` 为兼容旧命令保留，普通文本续写已是默认。诗词格式约束仅接受旧字符词表，不接受 hybrid。

## 7. 测试与后续方向

```powershell
uv run --project sft_gpt --no-sync python -m unittest discover -s sft_gpt/tests -t . -v
uv run --project sft_gpt --no-sync python -m unittest discover -s mini_gpt -p "test_*.py" -v
uv lock --project sft_gpt --check --offline
```

覆盖：模型原样保持、Unicode 随机无损往返、字节词表及持久化、UTF-8 生成约束、语料多格式与完整对话划分、packing 标签无遗漏、padding loss、词表仅训练集统计、通用预训练中断精确恢复、旧检查点和旧五绝轨迹。每个类单独文件。

本次验证：新工程 **78 项测试**、原工程 **21 项测试**全部通过；uv 锁文件离线检查通过。RTX 4070 SUPER 上用 12 条示例文档完成 2 轮预训练（11 条训练、1 条验证），得到 301-token hybrid 词表，训练和验证 UNK 均为 0；并验证普通文本生成和旧诗词检查点推理。此词表只来自极小样例，不是正式日常语料词表，示例输出不代表聊天质量。旧字符词表仍为 3432 项，未被新默认词表覆盖。

日常聊天能力还取决于语料规模与质量、预训练效果和后续 SFT。当前模型上下文仅 128 token，架构和容量没有因 tokenizer 改造而扩大；后续可另行设计更长上下文和 assistant-only SFT，但不在本次偷偷改模型。附带语料只用于功能演示。
