# SFT GPT：通用文本预训练工程

当前实现的是 **pre train（自回归预训练）**：普通文本、日常交流语料和对话记录都可以参与训练。不再默认筛选五言绝句，也不默认按诗词格式生成。**这不是 assistant-only SFT，也不是已经具备日常聊天能力的模型。**

## 1. 保持不变的模型与环境

模型结构、参数形状、初始化和权重共享保持不变。Attention 使用 SDPA；RMSNorm、RoPE、残差及 SwiGLU 增加显式混合精度边界，不再要求这些模块的 forward 与旧源码逐字相同。未改动定义仍做源码与 AST 对照，模型初始化、FP32 前向结果及梯度另有回归测试。

当前模型配置见 [config.py](config.py)：D_MODEL=288、NUM_HEADS=6、D_FF=768、NUM_LAYERS=8、MAX_SEQ_LEN=512、ROPE_BASE=10000、DROPOUT_RATE=0.1。数据流程和混合精度边界调整不修改这些模型常量或网络结构；`--seq-len` 只决定数据打包长度，不能超过 MAX_SEQ_LEN。

所有命令在仓库根目录 `E:\Code\DeepLearning` 执行。沿用现有 uv 环境，**没有安装或升级依赖**。

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

### Wiki 下载、划分与分片

在项目目录运行：

```powershell
# 一次完成下载、去重、按文档拆分、分片落盘
./.venv/Scripts/python.exe -m corpus

# 已有旧版根目录原始分片时可离线整理；也可将 --input 换成原始 JSON/JSONL 文件
./.venv/Scripts/python.exe -m corpus --input .data/corpus
```

默认最多收集 50,000 条有效去重文档，`--eval-ratio 0.1`、`--seed 42`，**每个文件最多 `--max-docs-per-file 5000` 条**，不限定训练／验证的分片数。先按完整文档拆分 train/eval，再分别按上限连续写文件；只有尾文件可不足上限，不生成空文件。50,000 条时训练为 45,000 条 / **9 个文件**、验证为 5,000 条 / **1 个文件**。其他数据量按各集合实际条数向上取整计算文件数；即使只有两个有效文档也可生成非空 train/eval。两个集合没有完全相同的文档。下载、解析或写入失败不会用半成品替换已有语料；仅清理 manifest 登记的旧生成文件，减少分片数时不会残留多余旧片。模块导入不下载。

下载与处理实现集中在 [corpus](corpus/)：
- [download.py](corpus/download.py)：下载／本地读取、去重拆分、按单文件条数上限分片和安全发布。
- [cleaning.py](corpus/cleaning.py)：下载与训练加载共用的中英字符清洗。
- [__main__.py](corpus/__main__.py)：`python -m corpus` 入口。

下载和离线整理统一使用 `python -m corpus`（本地输入加 `--input`），清洗 API 使用 `sft_gpt.corpus.cleaning.clean_corpus_text`。冗余的旧根目录转发脚本及单独的 repartition 包装入口已移除，不再保留多套入口。源码目录与输出目录 `.data/corpus` 分开；离线整理不删除原始输入，默认训练只读 train/eval。

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

1. 两个输入都不传时，直接加载 `.data/corpus/train` 和 `.data/corpus/eval`；只显式传 `--train-data`（或 `--data`）时，两个默认目录均不使用，按 `--valid-ratio`（默认 0.1）从该输入拆分验证集。两个输入都指定时保留输入划分，不二次拆分；仅指定 `--eval-data` 报错，不隐式混入默认训练目录。按完整文档去重并用固定种子打乱，整段 messages 不会跨集合拆分。
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

每个文档编码为 BOS + 内容 + EOS。**训练与验证分别打包**成连续 token 流，窗口是 `seq_len+1`，步长为 seq_len，边界共享一个 token；输入与标签错开一位，流中每个可预测目标只出现一次。长文不会只留下第一个窗口，尾部也保留。

- 同一 split 内可把不同文档拼到一个块，BOS/EOS 标识边界。因没有修改原 attention，**没有文档间 attention 隔离**，这是普通 packed causal-LM 预训练，不是逐对话独立 attention。
- 最后不足一块时仅右侧补齐为 -100。送 Embedding 前把负数换为安全 ID，标签保留 -100，CrossEntropyLoss 忽略补齐位置。有效 token 在补齐之前，原因果注意力无需新增 padding mask。
- 含补齐数据的平均 loss 按有效预测 token 数加权，不让短尾块与完整块等权。旧无 padding 五绝模式保留原统计与训练轨迹。
- 日志显示文档数、序列数、词表类型、有效 token、padding 和 UNK 比例。
- checkpoint 保存已划分文本、tokenizer、序列模式、seq_len、评估索引、优化器、调度器和 RNG。恢复时沿用这些设置，不用新的 CLI 参数重新切分或重建词表。

当前读取、去重和编码在内存中完成，检查点也保存文本以保证恢复一致；适合小模型实验，**不是十亿 token 级流式数据管线**。纯文本长度仍按 token 计，不按汉字数计。

## 5. uv 预训练命令

命令保留大小写敏感的 **preTrain**。不传子命令也进入 preTrain，默认加载项目内 `.data/corpus/train` 和 `.data/corpus/eval`。默认目录不存在或过滤后为空会明确失败，不会自动下载或回退到旧五绝 TSV。

```powershell
# 先用附带的极小语料跑通链路，不代表可训练出聊天能力
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --data sft_gpt/examples/pretrain_texts.jsonl --epochs 2 --batch-size 4 --seq-len 64 --output-dir sft_gpt/.data/text_smoke

# 使用默认已拆分的 Wiki 目录
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --tokenizer hybrid --vocab-size 8192 --min-freq 2 --seq-len 128 --output-dir sft_gpt/checkpoints/pre_train/daily

# 自定义两个独立目录（也可以传文件），不重新拆分
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --train-data sft_gpt/.data/corpus/train --eval-data sft_gpt/.data/corpus/eval --data-workers 8

# 只提供训练目录，从该目录拆出 10% 验证；不使用任何默认语料目录
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --train-data sft_gpt/examples --valid-ratio 0.1

# 只读复用先前冻结的词表
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --vocab sft_gpt/.data/tokenizer/text_vocab.json --output-dir sft_gpt/checkpoints/pre_train/daily_fixed

# 恢复：epochs 是目标总轮数；数据、序列配置、词表及训练设置取检查点
uv run --project sft_gpt --no-sync python -m sft_gpt preTrain --resume sft_gpt/checkpoints/pre_train/daily/last.pt --epochs 200 --output-dir sft_gpt/checkpoints/pre_train/daily
```

默认输出仍是 `checkpoints/pre_train/`，明确不是 SFT 产物，文件为 best.pt / last.pt / vocab.json，.pt 含 `training_stage="pre_train"`。保留原有 AdamW、warmup + 余弦退火、梯度裁剪、标签平滑、固定训练评估子集、早停行为。`--max-documents` 限制训练与验证的文档总数便于冒烟，两个独立输入时按其实际比例预留、每个集合至少一篇；仅训练输入时先限量再按 `--valid-ratio` 拆分。旧 `--max-poems` 是同义别名。

没有自动预留测试集；正式基准应准备独立测试文档，不用于建表或调参。更换 tokenizer、语料或打包策略后 loss 口径会变，不应直接和旧五绝实验的数值比较。

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

本次显式精度边界验证：项目环境 PyTorch 2.14.1+cu132 全量 **162 项测试通过**；Conda `pytorch` 环境 PyTorch 2.0.1 / CUDA 11.8 的 **31 项相关测试通过**（新增 13 项边界测试，以及 18 项既有精度、SDPA 和模型保留回归）。两套环境均在 RTX 4070 SUPER 实际执行 CUDA BF16 用例，没有跳过。新增 [边界回归测试](<tests/test_precision_boundaries.py>) 通过全层 hooks、真实 SDPA/交叉熵与独立数值参考检查 BF16/FP32 边界、FP32 梯度/AdamW 状态、CPU 回退及 FP64 兼容。本次未重新跑性能基准，也未安装或升级依赖。

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

默认保留 prompt 空白、换行、emoji，不强制五字、不强加中文标点。

```powershell
uv run --project sft_gpt --no-sync python -m sft_gpt infer --checkpoint sft_gpt/.data/text_smoke/best.pt --prompt "你好，今天" --max-new-tokens 64

# 交互式前缀续写，/quit 退出；没有自动累积多轮历史
uv run --project sft_gpt --no-sync python -m sft_gpt infer --interactive --checkpoint sft_gpt/checkpoints/pre_train/daily/best.pt
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

本轮语料流程验证：项目全量 **143 项测试通过**，其中下载与分片 **26 项标准库离线测试**覆盖每文件条数上限、整除／余数边界、无空文件、小量文档、大验证集同样分片、旧多余分片清理、发布失败回滚和 UTF-8 分块边界。下载／清洗实现已集中到 corpus 包，冗余旧入口及独立 repartition 包装器已清理。已删除原来的旧语料目录，并用 `python -m corpus` 从远端真实重新下载生成 50,000 条有效去重文档：训练 45,000 条 / **9 个文件**，验证 5,000 条 / **1 个文件**，本次每个文件均为 **5,000 条**。逐文件检查和默认目录加载复核通过，集合间完全重复文档为 0，所有文档满足字符白名单及最短长度要求，manifest 与实际文件一致，无旧分片残留。

日常聊天能力还取决于语料规模与质量、预训练效果和后续 SFT。当前模型上下文为 512 token，架构和容量没有因 tokenizer 改造而扩大；后续可另行设计更长上下文和 assistant-only SFT，但不在本次偷偷改模型。附带语料只用于功能演示。
