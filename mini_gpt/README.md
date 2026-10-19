# MiniGPT 五言绝句训练与推理

保留原模型的 128 维、8 头、8 层、RoPE、RMSNorm、SwiGLU、残差和输入输出权重共享。仅修复原代码的导入/语法问题及 RMSNorm、GPTBlock 缺失的 nn.Module 继承，移除未使用的空 SelfAttention 占位。

## 环境与路径

使用项目已有的 uv 环境，不需要激活 Conda。下列命令均在项目根目录 `E:\Code\DeepLearning` 执行。

所有命令使用 `uv run --project mini_gpt --no-sync`，选择 `mini_gpt/.venv` 中现有的解释器和依赖；`--no-sync` 不会重新同步、安装或升级依赖。若环境尚未建立，这套命令不能代替环境安装。

可先检查实际使用的解释器、PyTorch 和 CUDA：

~~~powershell
uv run --project mini_gpt --no-sync python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
~~~

本次实测：uv 0.13.1、Python 3.12.15、PyTorch 2.14.1+cpu，CUDA 不可用。因此当前 uv 环境会自动使用 CPU，不能通过 `--device cuda` 开启 GPU。代码可用，但完整训练可能较慢；本次没有替换 PyTorch、安装依赖，也没有修改项目依赖配置或锁文件。

> 如果在受限环境遇到默认 uv 缓存目录不可写，可先在当前 PowerShell 会话指定工作区内的缓存目录，再执行下面的命令：
>
> ~~~powershell
> $env:UV_CACHE_DIR = "$PWD/.uv-cache"
> ~~~
>
> 这只更改缓存位置，不更改项目虚拟环境或依赖。

## 1. 训练

~~~powershell
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py train
~~~

不带子命令执行 `uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py` 也会进入默认训练。IDE 的 Run 请将解释器选为 `mini_gpt/.venv/Scripts/python.exe`。默认自动选择 GPU，最多 1000 轮，第 10 轮起累计未改善次数，连续 10 次未改善时早停。每轮打印完整训练的优化目标 objective（含标签平滑）、固定训练评估子集的 train_eval 和完整验证集的 valid（后二者均为 eval 模式、无标签平滑的交叉熵）。

可自定义参数：

~~~powershell
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py train --epochs 100 --batch-size 64 --lr 0.002
~~~

### 固定训练评估子集

默认从训练集固定抽取 1024 首，仅用于每轮结束后的训练损失评估；这些诗仍正常参与完整训练，不是额外留出的验证集。抽样使用独立 CPU 随机数生成器，不影响训练打乱或 dropout 的随机序列。每轮只评估同一批样本，避免重新抽样造成曲线抖动；验证集始终完整评估，最佳模型和早停仍只依据 valid。

- --train-eval-size 1024：默认；训练集不足 1024 首时评估全量。
- --train-eval-size 2048：增加估计稳定性，同时增加评估开销。
- --train-eval-size 0：恢复完整训练集评估。

检查点保存 train_eval_indices 及评估配置，恢复时复用原索引，不受本次 --seed 或 --train-eval-size 覆盖。旧检查点缺少这些信息时按本次设置建立一次固定子集。旧 history 的 train_loss（全量评估）不改写，新条目使用 train_eval_loss，避免把抽样指标伪装成同口径的历史全量指标。

例如 15066 首训练诗、batch-size 64 时，训练损失评估从每轮 236 批降到 16 批。只减少额外评估的计算，不减少优化器更新步数，也不保证总训练耗时按此比例下降；该改动本身不是防过拟合措施。

### 学习率：线性 warmup + 余弦退火

默认开启，按每次训练 batch 的优化器更新步进（验证不改变学习率）：

- `--lr 0.002`：预热结束后的峰值学习率。
- `--warmup-ratio 0.05`：前 5% 计划步数线性预热，从峰值 / 预热步数逐步升至峰值；设为 0 可禁用预热。
- `--min-lr-ratio 0.1`：随后单次余弦退火到峰值的 10%，默认最低值为 0.0002，不做周期重启。

计划步数 = 目标总轮数 × 每轮训练 batch 数（不足一批也计一步）。预热步数向上取整，小样本时尽量保留两个余弦阶段位置；只有一步时使用峰值。每轮记录并打印 `lr_start`（首批学习率）和 `lr_next`（下一批学习率）。早停仍有效，因此不保证一定走完整条曲线。默认 1000 轮对应约 50 轮预热，可根据实际训练计划缩短总轮数或降低 warmup 比例。

~~~powershell
python mini_gpt/mini_gpt.py train --epochs 100 --lr 0.002 --warmup-ratio 0.05 --min-lr-ratio 0.1
~~~

调度器接口：构造时传入 optimizer、total_steps、warmup_ratio 和 min_lr_ratio；warmup 步数换算及小样本边界处理均由类内部负责。每次 optimizer.step() 后调用 scheduler.step()；通过 get_last_lr() 查询已设置、供下一批使用的学习率，通过 get_info() 查询阶段（warmup/cosine/finished）、峰值、最低值和步数快照。state_dict()/load_state_dict() 对称保存、恢复状态，加载时先完整校验，再同步优化器学习率；仍兼容此前含 warmup_steps 的调度检查点。

默认数据路径相对脚本解析，不依赖当前工作目录；自定义 --data、--output-dir、--checkpoint 等相对路径则相对命令执行目录。

数据处理：读取 notebook/data/five_char_quatrains.tsv 的 text 列，去空白、检查四句五言及标点并去重；固定种子打乱，90% 训练、10% 验证。仅用训练集建立字符词表，低于 --min-freq（默认 5）的字映射为 UNK。每首编码为 BOS + 24 字符 + EOS，输入和标签错开一位预测下一个字符，无需 padding。

默认参数保存在脚本旁的 checkpoints 目录：

- best.pt：验证集损失改善时保存，用于推理。
- last.pt：每轮保存，用于恢复训练。

检查点包含模型 state_dict、词表、模型配置、优化器状态、训练配置、训练/验证文本划分、固定训练评估子集索引、轮数、损失历史和随机数状态。再次全新训练会覆盖同名文件，保留旧实验时请指定新的 --output-dir。

## 2. 恢复训练

~~~powershell
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py train --resume mini_gpt/checkpoints/last.pt --epochs 200
~~~

--epochs 表示目标总轮数，不是新增轮数。恢复时复用检查点内的词表、数据划分、batch-size、学习率、早停设置和优化器；不重新读取 TSV，其他数据及训练超参数选项不覆盖保存值。--device 和 --output-dir 仍可更改。建议继续使用原输出目录；更换输出目录时会校验并复制原目录的 best.pt，若原最佳检查点缺失或不匹配会明确报错。CPU/GPU 可切换，但跨设备不保证逐位一致。

新检查点同时保存学习率调度曲线与已完成步数，恢复时不会重新 warmup；命令行的学习率与调度比例不覆盖保存配置。增加 --epochs 不会拉伸原曲线，超过原计划步数后保持最低学习率，以避免学习率跳升。旧版固定学习率检查点仍可恢复：以检查点当前学习率为峰值，使用本次 warmup/min-lr 比例，在剩余轮数上建立新曲线，并打印迁移提示。

## 3. 输入开头进行推理

~~~powershell
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py infer --prompt "春"
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py infer --prompt "明月" --num-samples 3 --temperature 0.8 --top-k 30
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py infer --checkpoint mini_gpt/checkpoints/best.pt --prompt "春" --device cpu
~~~

这里的输入是“诗歌开头”，不是聊天指令或诗歌主题。默认强制四句五字和 ，。，。 标点；字由模型采样，格式约束不代表内容质量已达标。若要由模型自行生成标点和 EOS：

~~~powershell
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py infer --prompt "春" --free-form --max-new-tokens 40
~~~

--temperature 越低越保守；--top-k 0 不截断候选；--max-new-tokens 限制新增字符数（含标点）。词表外的开头会明确报错，可换常见字，或用更低的 --min-freq 重新训练。推理只需要检查点，无需 TSV。仅加载自己训练或可信来源的 pt 文件，torch.load 使用 pickle。

### 交互输入

~~~powershell
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py infer --interactive
~~~

输入开头并回车即可生成；空输入表示从 BOS 随机作诗；输入 /quit 退出。

## 4. 快速验证（不用于评估诗歌质量）

~~~powershell
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py train --epochs 1 --max-poems 64 --min-freq 1 --batch-size 16 --output-dir mini_gpt/smoke_checkpoints
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py infer --checkpoint mini_gpt/smoke_checkpoints/best.pt --prompt "春"
uv run --project mini_gpt --no-sync python mini_gpt/mini_gpt.py train --resume mini_gpt/smoke_checkpoints/last.pt --epochs 2 --output-dir mini_gpt/smoke_checkpoints
uv run --project mini_gpt --no-sync python -m unittest discover -s mini_gpt -p "test_*.py" -v
~~~

小样本/少轮训练仅验证管线，不能期待生成优质诗歌；正常使用先在完整数据上训练，再观察验证损失和实际生成质量。

## 5. 本次 uv 验证结果

- 64 首小样本训练 1 轮成功，保存最佳和最新检查点。
- 使用最佳检查点生成两首诗成功。
- 从最新检查点恢复到第 2 轮成功。
- 回归测试中，完整数据读取、编码解码、因果注意力和生成边界共 4 项通过。另 2 项涉及临时目录文件读写，被当前 Windows 沙箱权限阻止；完整测试未全部通过，不能据此宣称已完整验证。常规检查点目录的保存与加载已由上述训练/推理命令实际验证。
- 以上均在现有 uv CPU 环境中运行；没有进行完整数据的长时间训练，也没有验证此 uv 环境的 GPU 训练。
