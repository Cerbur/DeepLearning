# SFT GPT：通用文本预训练工程

当前实现的是 **pre train（自回归预训练）**：普通文本、日常交流语料和对话记录都可以参与训练。不再默认筛选五言绝句，也不默认按诗词格式生成。**这不是 assistant-only SFT，也不是已经具备日常聊天能力的模型。**

## 1. 保持不变的模型与环境

沿用用户设定的约 0.1B 模型结构、参数形状、初始化和权重共享（8192 词表时 98,324,736 个唯一参数）。训练优化不再改动模型常量；可选 activation checkpointing 只改变激活保存/重算策略。Attention 使用 SDPA；RMSNorm、RoPE、残差及 SwiGLU 增加显式混合精度边界，不再要求这些模块的 forward 与旧源码逐字相同。未改动定义仍做源码与 AST 对照，模型初始化、FP32 前向结果及梯度另有回归测试。

当前模型配置见 [config.py](config.py)：D_MODEL=768、NUM_HEADS=12、D_FF=2048、NUM_LAYERS=13、MAX_SEQ_LEN=512、ROPE_BASE=10000、DROPOUT_RATE=0.1。数据流程和混合精度边界调整不修改这些模型常量或网络结构；`--seq-len` 只决定数据打包长度，不能超过 MAX_SEQ_LEN。

除语料下载／离线整理命令在项目目录 `E:\Code\DeepLearning\sft_gpt` 执行外，下述 uv 训练、推理与测试命令仍在父目录 `E:\Code\DeepLearning` 执行。沿用现有 uv 环境，**没有安装或升级依赖**。

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv_cache"
uv run --project sft_gpt --no-sync python -m sft_gpt --help
```

本机为 Python 3.12.15、PyTorch 2.14.1+cu132、CUDA Runtime 13.2。环境在 `sft_gpt/.venv`，旧 `mini_gpt/.venv` 是兼容 Windows 启动器的 Junction；不要删除此连接及仓库根 `.python`。IDE 使用 `sft_gpt/.venv/Scripts/python.exe`。PowerShell 不允许未签名激活脚本时，直接运行 uv 即可，无需修改系统策略。后续依赖安装继续遵循 AGENTS 的 Conda 优先规范，不混改这两套环境。

## 2. 通用语料格式

`--train-data` 和 `--eval-data` 分别接受单个文件或目录；`--data` 保留为训练输入的兼容别名。目录递归扫描 `.txt/.jsonl/.json/.csv/.tsv`，并发读取（每个集合默认 `--data-workers 8`，训练和验证也同时加载），按相对路径顺序合并，保证结果可复现。**请只传语料专用目录，不要把 tokenizer JSON、检查点输出或整个项目根目录混进去。** UTF-8，可带 BOM；不做大小写转换、繁简转换或全局去空白。默认使用下述中文／英文字符白名单清洗。

### 默认字符清洗（下载与加载共用）

仅保留：中文汉字（含繁体、CJK 扩展汉字及〇）、ASCII 英文字母 A–Z/a–z、ASCII 数字 0–9、ASCII 标点/常用符号、Unicode 标点，以及空格、Tab、CR/LF。日文假名、韩文、俄文、希腊文、阿拉伯文等其他文字、重音字母、全角英数、emoji 和其他符号连续片段替换为一个空格，避免把相邻英文直接粘连。清洗后只有标点／空白、没有中英数字正文的记录丢弃；去重、训练／验证隔离和建词表都在清洗之后。对话的每条 content 先清洗，任何一条清洗后无正文则跳过整段对话，不把 role 标签误当正文。

这是**字符级约束，不是语义语言识别**：中日共享汉字无法仅靠字符范围区分。下载时仍清理多余空行并去掉文档首尾空白；加载自定义文本时保留其余合法空白。tokenizer 本身仍支持完整 Unicode，旧检查点不重新清洗或改写。

### 默认 CDial-GPT LCCC-base 下载、划分与分片

在项目目录运行：

```powershell
# 默认下载 LCCC-base，清洗完整对话、去重、拆分并分片落盘
./.venv/Scripts/python.exe -m corpus

# 可调整有效去重完整对话总数上限；默认52k不是全量数据集的随机样本
./.venv/Scripts/python.exe -m corpus --target-docs 100000

# 本地 JSON/JSONL，可选 .gz 压缩；仅离线读取，不联网
./.venv/Scripts/python.exe -m corpus --input path/to/dialogs.jsonl.gz
```

默认使用 Python 标准库 `gzip` 流式读取 [LCCC-base train 压缩 JSONL](<https://hf-mirror.com/datasets/silver/lccc/resolve/main/lccc_base_train.jsonl.gz>)，该地址由 [thu-coai/lccc 官方 loader](<https://hf-mirror.com/datasets/thu-coai/lccc/raw/main/lccc.py>) 引用，不加载 CDial-GPT 模型权重。上游项目见 [CDial-GPT GitHub](<https://github.com/thu-coai/CDial-GPT>)；Hugging Face 来源见 [thu-coai/lccc](<https://huggingface.co/datasets/thu-coai/lccc>) 与实际数据仓库 [silver/lccc](<https://huggingface.co/datasets/silver/lccc>)。

默认最多收集 `--target-docs 52000` 条有效、去重的**完整对话**（`TARGET_DOCS=52000`）：按上游顺序读取，达到上限即停止，因此这是经过过滤的**前缀子集，不是从完整数据集随机抽样**。`--target-docs` 控制训练加验证的**总量**，不是训练集条数。之后用 `--seed 42` 和默认 `EVAL_RATIO=2000/52000=1/26`（约 **3.846%**），从收集到的上游 **train** 完整对话中在本地划分 eval，**不使用官方 valid/test**，同一对话不会跨集合拆开。`--eval-ratio` 仍可显式调整，例如 `--eval-ratio 0.05`。

默认输出到项目内 `.data/corpus/train` 和 `.data/corpus/eval`，两边各从 `lccc_part_001.jsonl` 开始命名，**每个文件最多 `--max-docs-per-file 5000` 条**。收集满默认 52,000 条时，训练为 **50,000 条 / 10 个文件**（每文件 5,000 条），验证为 **2,000 条 / 1 个文件**。不限定训练／验证的分片数；只有尾文件可不足上限，不生成空文件。其他数据量按各集合实际条数向上取整计算文件数；即使只有两个有效对话也可生成非空 train/eval。两个集合没有完全相同的文档。下载、解析或写入失败不会用半成品替换已有语料；成功发布时按 manifest 删除登记的旧生成文件，包括原 Wiki 分片，减少分片数时不会残留多余旧片，不删除未登记的其他文件。模块导入不下载。

LCCC 记录支持 `list[str]` 或 `{"dialog": list[str]}`，每段至少 2 轮。**保留完整对话及原句顺序，各句用换行连接，不添加 `user:` / `assistant:` 或其他角色标签**；BOS/EOS 由既有训练打包流程处理，不写成正文。任一轮清洗后为空／无正文，则丢弃**整段**，不删除该轮后继续拼接。默认 `--min-doc-len 2` 只计算各轮清洗后正文总长度，不计分隔换行。默认 `--tokenization word` 适配该来源的词级空格预分词：按 token 合并中文及标点边界（如 `我 今天 想 吃 火锅` → `我今天想吃火锅`、`你 好 啊` → `你好啊`），仅相邻 ASCII 英数字 token 之间保留一个空格，因而 `Hello world`、`1 2 3` 不会误合并。字符级预分词版本须显式指定 `--tokenization character`，才会合并所有 token（如 `h e l l o` → `hello`）；不凭 token 长度猜测英文／数字语义。旧 `text` / `completion` 记录仍兼容。JSONL 每行可以直接放一段 `list[str]`；普通 JSON 的多段对话须使用外层数组（如 `[ ["你好", "你好呀"], ["晚安", "好梦"] ]`），单段可用 `{"dialog": ["你好", "你好呀"]}` 包装，不要把顶层裸字符串数组误作记录集合。

LCCC-base 相比 Wiki 更适合本项目的闲聊语料方向，但存在噪声与质量限制；52k 前缀子集及当前小模型均**不保证聊天效果**。

下载与处理实现集中在 [corpus](corpus/)：
- [download.py](corpus/download.py)：下载／本地读取、去重拆分、按单文件条数上限分片和安全发布。
- [cleaning.py](corpus/cleaning.py)：下载与训练加载共用的中英字符清洗。
- [__main__.py](corpus/__main__.py)：`python -m corpus` 入口。

下载和离线整理统一使用 `python -m corpus`（本地输入加 `--input`），清洗 API 使用 `sft_gpt.corpus.cleaning.clean_corpus_text`。冗余的旧根目录转发脚本及单独的 repartition 包装入口已移除，不再保留多套入口。源码目录与输出目录 `.data/corpus` 分开；离线整理不删除原始输入。**不再保留旧 Wiki 语料**：LCCC 沿用 `.data/corpus/train` 和 `.data/corpus/eval`，成功发布会依据 manifest 清理旧 owned Wiki 分片；先前临时导出的 `.data/lccc` 目录移除，不再作为训练输入。旧检查点仍保留，不会因切换默认语料而迁移或删除。

### TXT：空行分隔文档

```text
今天下班以后，想做一顿简单的晚饭。
可以煮面，也可以炒青菜。

Learning a little every day helps.
```

保留非空行的前后空格及原始换行（包括 CRLF），空白行只用于分隔文档。如果需要把一整段多行对话作为一个文档，可用 JSONL `text` 保存正文换行，避免空行把对话拆开；自定义有角色语料也可以选择 `messages`。

### JSONL：每行一条文本或完整对话

```jsonl
{"text":"今天有点累，想早点休息。"}
{"text":"Hello world! 中文和 English 都可以，包含 emoji 😊。"}
{"text":"保留两行：\n第一行和第二行。"}
{"messages":[{"role":"system","content":"自然、礼貌地交流。"},{"role":"user","content":"晚上吃什么？"},{"role":"assistant","content":"你想吃清淡一点，还是热乎的饭菜？"}]}
```

默认 LCCC 导出为 `text`，内容只是按原顺序换行连接的各句正文，例如：

```jsonl
{"text":"晚上吃什么？\n你想吃清淡一点，还是热乎的饭菜？"}
```

`messages` 是**自定义语料的可选格式，不是默认 LCCC 格式**。它支持 `system/user/assistant` 与字符串 content，整段对话作为一个文档，按固定模板转为：

```text
system: 自然、礼貌地交流。
user: 晚上吃什么？
assistant: 你想吃清淡一点，还是热乎的饭菜？
```

以上带 role 的模板仅适用于自定义 `messages`，role 是普通文本，不是特殊控制 token；默认 LCCC 不加 role。所有文本位置都参与 next-token loss，**没有只对 assistant 内容计算损失**。消息内容中写出的 `<BOS>` 等拼写不会被解析成控制 token。

JSON 接受记录列表或单条记录；CSV/TSV 默认读取 `text` 列，`--text-field content` 可指定字段名。默认过滤不支持的文件、空文档、坏 JSONL 行、异常记录类型、非法消息、表格列数异常、二进制控制字符和无效 Unicode，并汇总警告（附路径/行号示例）。无法解析的 JSON 文件、坏表头、损坏编码或不可读文件整文件跳过；CSV 语法损坏也整文件跳过，不猜测记录边界。Python API `load_texts(..., strict=True)` 可用于立即报错排查。集合内按全文去重；训练集与验证集重叠的全文从训练集删除，验证集优先，删除后为空则报错。只传训练输入至少需两个有效文档；两个独立输入各至少一个。**不做近重复检测**；近重复、同源文章或同一人的多段记录应在准备阶段按来源分组隔离。

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

1. 从零训练（不传 `--resume` / `--init-from`）时，两个输入都不传则直接加载 `.data/corpus/train` 和 `.data/corpus/eval`；只显式传 `--train-data`（或 `--data`）时，两个默认目录均不使用，按 `--valid-ratio`（默认 0.1）从该输入拆分验证集。两个输入都指定时保留输入划分，不二次拆分；仅指定 `--eval-data` 报错，不隐式混入默认训练目录。按完整文档去重并用固定种子打乱，整段 messages 不会跨集合拆分。
2. 从零建表时，**只用训练集统计高频字符**，不让验证集决定词表。验证集中没见过的字符由固定 UTF-8 字节覆盖；`--init-from` 则精确复用源 tokenizer，不用新语料重新统计建表。
3. 从零新建的 hybrid 默认词表保存在 `.data/tokenizer/text_vocab.json`，不覆盖旧 `.data/tokenizer/vocab.json`。
4. 输出目录也保存 `vocab.json`；.pt 内嵌完整 tokenizer 类型、版本和有序词表，是模型编码方式的依据。
5. `--vocab` 只读复用指定词表，不重建、不回写；从零建表可能覆盖默认 text_vocab.json，保留实验时请另存词表（独立输出目录本身不隔离这个全局默认路径）。`--init-from` 不重建、也不覆盖全局默认词表；若同时显式传 `--vocab`，其 tokenizer 类型和完整 token ID 映射必须与源检查点一致。
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

旧 `CharTokenizer` API 和 JSON 保持兼容；`--tokenizer char` 可显式使用（仍可能出现 UNK）。**不能把旧模型直接换一份 hybrid 词表继续推理或续训**：ID、词表大小和词义不同，更换词表必须从零训练；仅换语料可用 `--init-from` 继承权重并冻结原词表，不能借此把 char 改成 hybrid。

## 4. 通用变长序列与 loss

每个文档编码为 BOS + 内容 + EOS。**训练与验证分别打包**成连续 token 流，窗口是 `seq_len+1`，步长为 seq_len，边界共享一个 token；输入与标签错开一位，流中每个可预测目标只出现一次。长文不会只留下第一个窗口，尾部也保留。

- 同一 split 内可把不同文档拼到一个块，BOS/EOS 标识边界。因没有修改原 attention，**没有文档间 attention 隔离**，这是普通 packed causal-LM 预训练，不是逐对话独立 attention。
- 最后不足一块时仅右侧补齐为 -100。送 Embedding 前把负数换为安全 ID，标签保留 -100，CrossEntropyLoss 忽略补齐位置。有效 token 在补齐之前，原因果注意力无需新增 padding mask。
- 含补齐数据的平均 loss 按有效预测 token 数加权，不让短尾块与完整块等权。旧无 padding 五绝模式保留原统计与训练轨迹。
- 日志显示文档数、序列数、词表类型、有效 token、padding 和 UNK 比例。
- checkpoint 保存已划分文本、tokenizer、序列模式、seq_len、评估索引、优化器、调度器和 RNG。恢复时沿用这些设置，不用新的 CLI 参数重新切分或重建词表。

当前读取、去重和编码在内存中完成，检查点也保存文本以保证恢复一致；适合小模型实验，**不是十亿 token 级流式数据管线**。纯文本长度仍按 token 计，不按汉字数计。

## 5. uv 预训练命令

命令保留大小写敏感的 **preTrain**。不传子命令也进入 preTrain，默认加载项目内 `.data/corpus/train` 和 `.data/corpus/eval`。默认目录不存在或过滤后为空会明确失败，不会自动下载或回退到旧 Wiki／五绝数据。

**切换语料有两条路径：从零训练并新建词表，或用 `--init-from PATH` 继承兼容检查点的权重和冻结词表，开始新阶段。** 从零切换到 LCCC 时不传旧 Wiki 的 `--resume` / `--vocab`；继承旧模型则见下方继续预训练示例。旧 Wiki checkpoint 内含旧训练／验证文本，`--resume` 恢复这些语料，不会读取新的默认 LCCC 目录；它禁止同时传 `--train-data`、`--data` 或 `--eval-data`，换语料会提示改用 `--init-from`。下方普通恢复示例只恢复其对应的 LCCC 阶段。模型架构和全 token 自回归预训练目标不变，不是 assistant-only SFT，也不加载 CDial-GPT 权重。

```powershell
# 先用附带的极小语料跑通链路，不代表可训练出聊天能力
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --data sft_gpt/examples/pretrain_texts.jsonl --epochs 2 --batch-size 4 --seq-len 64 --output-dir sft_gpt/.data/text_smoke

# 使用默认已拆分的 LCCC-base 目录，从头训练并新建词表
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --tokenizer hybrid --vocab-size 8192 --min-freq 2 --seq-len 128 --output-dir sft_gpt/checkpoints/pre_train/lccc

# 自定义两个独立目录（也可以传文件），不重新拆分
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --train-data sft_gpt/.data/corpus/train --eval-data sft_gpt/.data/corpus/eval --data-workers 8

# 仅自定义训练输入时，按训练CLI的valid-ratio拆出10%验证；不是下载器的默认eval比例
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --train-data sft_gpt/examples --valid-ratio 0.1

# 从零训练时只读复用先前的 LCCC 词表，不继承模型权重；继承旧模型须用下方 init-from
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --vocab sft_gpt/.data/tokenizer/text_vocab.json --output-dir sft_gpt/checkpoints/pre_train/lccc_fixed

# 仅恢复本次新建的 LCCC 实验；epochs 是目标总轮数，数据等设置取检查点
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --resume sft_gpt/checkpoints/pre_train/lccc/last.pt --epochs 200 --output-dir sft_gpt/checkpoints/pre_train/lccc
```

新训练默认输出改为项目内 `checkpoints/pre_train/lccc`；`--resume` 未显式指定 `--output-dir` 时沿用被恢复检查点所在目录，避免旧 Wiki 续训产物默认写进 LCCC 实验。明确不是 SFT 产物，文件为 best.pt / last.pt / vocab.json，.pt 含 `training_stage="pre_train"`。使用 AdamW、按优化器更新步数的 warmup + 余弦退火、梯度裁剪、可配置标签平滑及固定训练评估子集。新实验默认使用下述 0.1B 配方；旧 checkpoint 恢复不套用新默认值。`--max-documents` 限制训练与验证的文档总数便于冒烟，两个独立输入时按其实际比例预留、每个集合至少一篇；仅训练输入时先限量再按 `--valid-ratio` 拆分。旧 `--max-poems` 是同义别名。

没有自动预留测试集；正式基准应准备独立测试文档，不用于建表或调参。更换 tokenizer、语料或打包策略后 loss 口径会变，不应直接和旧五绝实验的数值比较。

### 换语料继续预训练：`--init-from` 与 `--resume`

- `--init-from PATH` 与 `--resume` **互斥**。前者只继承源模型权重及 tokenizer 的精确 token ID，后者恢复同一阶段的训练状态和已保存文本；`--vocab` 单独使用只复用词表，不加载模型权重。
- init 必须**显式指定 `--train-data`（或 `--data`）和 `--output-dir`**，不隐式使用默认训练目录。可加 `--eval-data` 传入独立固定验证集；也可只给训练输入，由 `--valid-ratio` 拆分。输出目录必须不存在或为空，且不能是源 checkpoint 所在目录；每个新阶段使用新目录，避免覆盖已有实验。
- 源架构必须与当前 [config.py](<config.py>) 一致，所有模型权重按原形状加载；**不能借此缩小已训练模型**。词表冻结指 token ID 映射不变，不是冻结 embedding 权重。不能更换 tokenizer 或重建词表；显式 `--vocab` 仅允许与源 tokenizer 完全一致。init 不重建、不覆盖全局默认词表。
- 新阶段采用本次 CLI 的训练参数（未传项使用新训练默认值），**不继承**源 optimizer、scheduler、global_step、trained_tokens、history、best、early-stop、轮内 cursor 或 RNG。重新初始化优化器、学习率曲线和随机状态；`--epochs`、`--max-steps`、`--max-tokens` 都是**新阶段预算**，不扣减源阶段已消耗的轮数／步数／tokens。
- checkpoint 保存来源 `initialization` 元数据，并保存**本阶段新语料**及其划分。新阶段中断后用本阶段的 `last.pt` 加 `--resume` 恢复，不要再次对源检查点执行 init（那会再开一个阶段）。resume 不允许同时传训练／验证输入，包括 `--data` 别名；如需再次换语料，改用 init 和另一个新输出目录。
- **没有自动混合旧新语料，也不检查新语料／新验证集是否泄漏源阶段的旧训练数据。** 新旧语料混合、按来源隔离和独立固定验证集均由用户准备；当前输入之间的全文去重不能代替跨阶段泄漏检查。更换语料会改变 loss 分布，不宜把新旧阶段 best/loss 直接当作同一基准。
- **仅加载自己生成或可信来源的 checkpoint**：`torch.load` 使用 pickle。降低 LR 是可选调参，不是 init 默认自动降 LR；例如可根据验证结果主动把 `--lr 3e-4` 改成 `--lr 1e-4`，不保证一定更好。

以下 PowerShell 命令仍从父目录 `E:\Code\DeepLearning` 执行，直接使用现有虚拟环境，**不安装依赖**。示例源为已训练的兼容 0.1B LCCC 模型；请先准备好对应语料，并确认两个新阶段输出目录不存在或为空。两个 init 示例是不同阶段方案，不会自动混合彼此的数据。

```powershell
# 方案 A：显式给 train/eval，保留用户准备的固定验证集；输出到全新目录
./sft_gpt/.venv/Scripts/python.exe -m sft_gpt preTrain --init-from sft_gpt/checkpoints/pre_train/lccc_100m/last.pt --train-data sft_gpt/.data/corpus/train --eval-data sft_gpt/.data/corpus/eval --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 16 --seq-len 512 --lr 3e-4 --epochs 10 --max-steps 2000 --max-tokens 0 --output-dir sft_gpt/checkpoints/pre_train/lccc_stage2_fixed

# 方案 B：只有训练输入（data 是 train-data 别名），从中拆出 10% 验证；另用新目录
./sft_gpt/.venv/Scripts/python.exe -m sft_gpt preTrain --init-from sft_gpt/checkpoints/pre_train/lccc_100m/last.pt --data sft_gpt/examples/pretrain_texts.jsonl --valid-ratio 0.1 --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 16 --seq-len 64 --lr 3e-4 --epochs 2 --max-steps 100 --max-tokens 0 --output-dir sft_gpt/checkpoints/pre_train/text_stage2_split

# 方案 A 阶段内中断恢复：用本阶段 last.pt，不再传 init-from、train-data/data 或 eval-data
./sft_gpt/.venv/Scripts/python.exe -m sft_gpt preTrain --resume sft_gpt/checkpoints/pre_train/lccc_stage2_fixed/last.pt --device cuda
```

### 0.1B 训练默认值与梯度累积

| 项目 | 新实验默认值 |
|---|---|
| micro-batch / 梯度累积 | 4 / 16（有效 batch 64，512 长度时约 32768 tokens/update） |
| 峰值 LR / warmup / 最低 LR 比例 | 3e-4 / 5% 更新步数 / 0.1 |
| AdamW betas / weight decay | (0.9, 0.95) / 矩阵 0.1；norm、1D、scalar 为 0 |
| label smoothing / clip norm | 0.0 / 1.0 |
| 最大更新数 / token 软预算 | 10000 / 0（不限）；epochs=1000 仅作额外上限 |
| 评估 batch / 步级验证 / 步级保存 | 4 / 每250更新 / 每250更新 |
| 步级验证子集 | 固定抽样，最多64个评估batch；0表示全量 |
| activation checkpointing / fused AdamW | 默认关闭，显式 opt-in |

- --batch-size 是单次前向 micro-batch，不是有效 batch。梯度累积按整个窗口的有效预测 token 数归一化；尾窗口、不等大 micro-batch、-100 padding 都正确处理，不会固定除以16而缩小尾批梯度。每个微批立即反向释放图，不保留16份前向图。裁剪、optimizer.step 与 scheduler.step 只在窗口结束执行。设 --grad-accum-steps 1 可禁用累积。
- 更新数按 ceil(ceil(序列数 / micro-batch) / accum) 计算，每轮尾窗口更新。--max-steps 是整个实验更新上限，不是每轮上限；--max-tokens 是整个实验有效预测 token 软上限，开始的窗口会完成，可能超过预算最多一个窗口。二者与 epochs 任意一个先达到即停止。token 预算对应的 LR 总步数使用平均 tokens/update 估计，padding/尾窗可能使最后更新数与估计略有不同。
- 步级验证只作诊断，不与全量 loss 混用；每轮/预算结束仍完整验证，best 和早停只依据完整验证；预算截断的半轮记录标记 epoch_complete=false，只有完整 epoch 才累计/重置早停 patience。--eval-every-steps 0 / --save-every-steps 0 可禁用步级操作。计时日志的 tokens/s 含评估/步级保存耗时，不是纯 kernel 基准；同时记录峰值 allocated 显存和裁剪前梯度范数，非有限梯度直接报错，避免污染优化器状态。
- --gradient-checkpointing 开启逐 block 非重入重计算，保留 dropout RNG 和 FP32 稳定区域；只在训练有梯度时生效，参数/state_dict 不变。显存不足也可用 batch=2、accum=32 保持有效 batch64。不要通过全模型 .bfloat16() 绕过精度边界。
- --fused-adamw 显式启用 CUDA fused AdamW，不支持时报错，不自动回退。共享 embedding/output 参数仅进入优化器一次；--no-optimizer-decay-groups 可使用旧单组策略。

以下命令均在父目录执行，不安装依赖：

```powershell
# 当前 0.1B 配方，新实验；不要覆盖原有实验目录
./sft_gpt/.venv/Scripts/python.exe -m sft_gpt preTrain --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 16 --seq-len 512 --lr 3e-4 --label-smoothing 0 --max-steps 10000 --output-dir sft_gpt/checkpoints/pre_train/lccc_100m

# 如显存不足，新训练在上面命令加 --gradient-checkpointing；或用 --batch-size 2 --grad-accum-steps 32

# 轮内/轮末恢复：自动沿用保存的 batch、累积、optimizer、checkpointing、词表与数据
./sft_gpt/.venv/Scripts/python.exe -m sft_gpt preTrain --resume sft_gpt/checkpoints/pre_train/lccc_100m/last.pt --device cuda

# 扩展总预算（保留原 LR 曲线，超原计划后保持最低学习率，不自动重新 warmup）
./sft_gpt/.venv/Scripts/python.exe -m sft_gpt preTrain --resume sft_gpt/checkpoints/pre_train/lccc_100m/last.pt --max-steps 20000 --device cuda
```

检查点新增 global_step、trained_tokens、epoch_complete、轮内排列/游标和累计 loss，仅在优化器更新边界保存，不保存半窗口梯度。恢复轮内游标不会重新洗牌；达到预算后再次恢复须扩展预算。没有这些字段的旧检查点按轮末恢复，默认 accum=1、单组AdamW、betas=(0.9,0.999)，仍要求模型结构匹配。老检查点没有累计 token 时，日志按已完成 epoch 和当前有效 token 数估算历史 token。新 CLI 的 batch/lr/累积等不会覆盖恢复设置；只有显式预算与精度允许覆盖。恢复到新输出目录时会校验并复制已有 best；首轮首次完整验证前的步级检查点尚无 best 时允许直接恢复。

**本次回归验证：** 项目现有环境共 **215 项测试通过**；全套运行时用进程内 32维/2层夹具避免大量 GB 级临时检查点，另以真实 98M 结构执行下述 GPU 更新验证。新增测试覆盖累积/尾窗/padding、大 batch 数值等价、非有限梯度保护、checkpointing/RNG、轮内中断精确恢复、预算与旧检查点兼容。没有修改磁盘上的模型结构常量，也未安装依赖。

**本机冒烟验证：** 4070 SUPER、项目 PyTorch 2.14.1+cu132、实际 98,324,736 参数、BF16、micro-batch=4、seq_len=512、accum=16，每种模式执行1次更新（32768预测tokens）。关闭/开启重计算的峰值 allocated 分别约 **2.68 / 1.87 GiB**，reserved 约 **2.77 / 1.98 GiB**；合成数据 loss 均有限、参数和最终梯度均为FP32。未预热的单次检查不是长期吞吐基准，也未启动正式语料训练。

**范围说明：** 本次不改变用户的 dropout=0.1、模型维度、tokenizer 或已有 BF16/SDPA。已降低 micro-batch 并在更新/评估前释放本批 logits，但仍使用完整词表投影与 FP32 CE，不声称已实现 fused/chunked linear-CE。读取/packing 和检查点全文语料仍在内存，未改成 mmap/token shards；十亿 token 级管线应单独改造。没有新引入依赖。

### BF16 混合精度（默认自动开启）

训练精度策略控制算子计算精度，并在模型中显式隔离 FP32 数值敏感区域；不改模型结构或参数存储类型：

- 不传 `--precision`：新训练默认 `auto`，支持 BF16 的 CUDA（例如 RTX 4070 SUPER）启用 BF16，CPU 或不支持的设备保持 FP32。
- `--precision bf16`：强制 BF16；设备不支持时明确报错，不静默改用 FP16。
- `--precision fp32`：关闭本训练循环的 autocast，便于与旧训练对照。
- 训练 objective、固定训练子集评估、验证使用同一精度策略。仅模型前向放在 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 中；启用 BF16 时，交叉熵在上下文外显式接收 `logits.float()`；非 BF16 路径保留输入 logits 的精度，包括独立 FP64 数值参考。反向、梯度裁剪、AdamW 和调度器也在上下文外。不使用 GradScaler，不把整模型转换成 BF16，权重、最终参数梯度和优化器动量仍为 FP32；反向内部算子不保证全部为 FP32。训练循环隔离调用方外层 autocast，并禁用 BF16 权重转换缓存，防止缓存跨优化器更新复用旧权重。

BF16 策略启用时的显式边界（`auto` 解析为 BF16 或显式 `bf16` 均相同）：

| 部分 | 计算/输出精度 |
|---|---|
| QKV、attention 输出投影、SwiGLU 三个 Linear、词表投影 | BF16 计算路径与输出；参数本体仍 FP32 |
| SDPA 的 Q/K/V 与 attention 输出 | BF16；内部 softmax/归约交由稳定融合内核或 FP32 中间量的数学后端处理，不拆回低精度手写 softmax |
| Embedding 输出、残差主干、dropout、残差加法 | FP32 |
| RMSNorm 平方/均值/开方/缩放与输出 | 显式关闭 autocast，低精度输入先升 FP32 |
| RoPE 位置/频率/sin/cos 表、旋转计算与输出 | FP32；进入 SDPA 前 Q/K 才转回 V 的 BF16 类型 |
| SwiGLU 的 SiLU 与门控乘法 | 显式 FP32；down_proj 恢复 BF16 计算路径 |
| 交叉熵与 loss | 显式 FP32，位于 BF16 autocast 外 |
| 权重、最终参数梯度、AdamW 动量、完整训练检查点 | FP32，不进行 BF16 压缩导出 |
| token ID、标签、布尔 mask | 保持整数/布尔语义，不转换为浮点 |

数值敏感模块仅将 FP16/BF16 提升为 FP32，不降低独立 FP64 数值参考测试的精度。`fp32` 或 CPU 回退不启用 BF16；独立 `infer` 仍默认 FP32，本次没有增加 BF16 推理或 KV cache。边界调整可能改变 BF16 下的数值轨迹，已有 loss 历史不保证逐位一致。

历史显式精度边界验证（早于本节训练优化）：项目环境 PyTorch 2.14.1+cu132 全量 **162 项测试通过**；Conda `pytorch` 环境 PyTorch 2.0.1 / CUDA 11.8 的 **31 项相关测试通过**（新增 13 项边界测试，以及 18 项既有精度、SDPA 和模型保留回归）。两套环境均在 RTX 4070 SUPER 实际执行 CUDA BF16 用例，没有跳过。新增 [边界回归测试](<tests/test_precision_boundaries.py>) 通过全层 hooks、真实 SDPA/交叉熵与独立数值参考检查 BF16/FP32 边界、FP32 梯度/AdamW 状态、CPU 回退及 FP64 兼容。本次未重新跑性能基准，也未安装或升级依赖。

- checkpoint 的 `train_settings.precision` 保存请求策略；恢复默认复用，可显式覆盖。无该字段的旧检查点按 `auto` 处理；需要旧 FP32 数值轨迹时显式传 `--precision fp32`。切换精度/设备会影响 loss 和早停比较，不保证与旧历史逐位一致。

以下命令在父目录执行，沿用现有环境，不安装依赖：

```powershell
# 4070 SUPER 新训练，显式开启 BF16（不传 precision 也会自动启用）
./sft_gpt/.venv/Scripts/python.exe -m sft_gpt preTrain --device cuda --precision bf16

# 从现有 last.pt 续训：在原恢复命令后加 --precision bf16 即可

# 合成数据 A/B 基准；只测训练循环，不读写正式词表和检查点
./sft_gpt/.venv/Scripts/python.exe -m sft_gpt.examples.benchmark_precision --batch-size 16 --seq-len 128 --vocab-size 8192
# 可追加 --bf16-first 交换顺序，复核温度/功率漂移；输出同时记录 TF32 设置
```

历史版本实测（早于当前 SDPA 与显式精度边界调整，不能作为当前版本性能保证；RTX 4070 SUPER，现有项目环境 PyTorch 2.14.1+cu132，batch=16、vocab=8192，TF32=False、float32_matmul_precision=highest；每模式预热 3 步，3 次各 10 步取耗时中位数；再交换 FP32/BF16 顺序复测）：

| 序列长度 | BF16 / FP32 训练吞吐比（正序 / 反序） | FP32 峰值 allocated 显存 | BF16 峰值 allocated 显存 |
|---|---:|---:|---:|
| 512（训练默认） | 1.13 / 1.39 倍 | 3484–3487 MiB | 3197–3199 MiB |
| 128（短序列） | 0.93 / 0.88 倍 | 832–834 MiB | 约 718 MiB |

默认 512 长度的两次运行均有提速，峰值 allocated 显存减少约 **8%**；128 短序列最终复测则慢约 **8%–14%**，显存减少约 **14%**，此时可用 `--precision fp32` 对照。先前单次短序列测量曾显示 1.29 倍，后续正反顺序复测未复现，不能当作稳定收益。不同运行间绝对耗时波动明显（512 正序 FP32/BF16 每步 316/280 ms，反序 113/82 ms），这里只报告两次实测比值，不承诺固定加速比，也不是全语料整轮基准。

历史验证记录（不是当前改动的复测结果）：当时项目环境全量 **136 项测试通过**；另在 Conda `pytorch` 环境（PyTorch 2.0.1 / CUDA 11.8）运行新增的 **9 项精度测试全部通过**，均实际执行 CUDA BF16 测试，包含有限 loss/梯度、FP32 参数/优化器状态、调度步数、padding loss、AMP 检查点 CPU 加载续训、精度策略恢复及旧检查点缺字段兼容。未安装或升级任何依赖。

BF16 通常能减少激活显存并加速矩阵运算，但参数、梯度、AdamW 状态仍占 FP32 显存；不能保证整体显存减半或固定 3–5 倍提速。模型尺寸、序列长度、数据准备、SDPA 后端和显式 FP32 边界都会影响收益。基准输出预热后多次测量的中位耗时、token/s 和 PyTorch 峰值 allocated 显存（不是 nvidia-smi 的总占用）；真实语料的整轮速度另受编码、评估和保存检查点影响。已启动的训练进程需要停止并从检查点重新启动才能使用新代码。

## 6. 普通文本推理与旧检查点

默认保留 prompt 空白、换行、emoji，不强制五字、不强加中文标点。默认 LCCC 是无角色的普通自回归文本，用自然文本和换行作续写前缀，例如 PowerShell 双引号字符串 ``"你好，今天怎么样？`n"``（其中反引号 + n 展开为换行），不添加 `user:` / `assistant:`。

```powershell
uv run --project sft_gpt --no-sync python -m sft_gpt infer --checkpoint sft_gpt/checkpoints/pre_train/lccc/best.pt --prompt "你好，今天怎么样？`n" --max-new-tokens 64

# 交互式前缀续写，/quit 退出；没有自动累积多轮历史
uv run --project sft_gpt --no-sync python -m sft_gpt infer --interactive --checkpoint sft_gpt/checkpoints/pre_train/lccc/best.pt
```

推理每次只送最后 MAX_SEQ_LEN=512 token 给模型（滑动窗口，**没有扩大上下文**）；输出包含原 prompt。hybrid 生成时屏蔽非法 UTF-8 字节组合、BOS/UNK/PAD，末尾预算不足时不开始无法完成的多字节字符，避免截断乱码。`--max-new-tokens` 是 token 数，不是字符数。

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

历史验证记录（并非本轮重跑结果）：新工程 **78 项测试**、原工程 **21 项测试**全部通过；uv 锁文件离线检查通过。RTX 4070 SUPER 上用 12 条示例文档完成 2 轮预训练（11 条训练、1 条验证），得到 301-token hybrid 词表，训练和验证 UNK 均为 0；并验证普通文本生成和旧诗词检查点推理。此词表只来自极小样例，不是正式日常语料词表，示例输出不代表聊天质量。旧字符词表仍为 3432 项，未被新默认词表覆盖。

本轮 LCCC-base 迁移验证：已真实下载并在 `.data/corpus` 导出 **50,000 条训练对话（10 片）+ 2,000 条验证对话（1 片）**；默认加载复核两集合各自全文唯一、交集为 0，不添加 user/assistant 角色标签。旧 Wiki 分片已按 manifest 清除，临时 `.data/lccc` 已删除，旧模型 checkpoint 保留。项目现有环境完整 **176 项测试通过**；RTX 4070 SUPER 使用默认新语料抽取 32 条对话完成 1 轮 CUDA BF16 训练（31 条训练、1 条验证，训练／验证 UNK 均为 0），验证读取、建表、打包、反向和检查点写入链路。该冒烟实验不是正式全量训练，也不代表聊天质量评测；未安装或升级依赖。

日常聊天能力还取决于语料规模与质量、预训练效果和后续 SFT。当前模型上下文为 512 token，架构和容量没有因 tokenizer 改造而扩大；后续可另行设计更长上下文和 assistant-only SFT，但不在本次偷偷改模型。附带语料只用于功能演示。
