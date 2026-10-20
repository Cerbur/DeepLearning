import argparse
import math
import random
import sys
from pathlib import Path

import torch

from .config import (DEFAULT_OUTPUT, MAX_EPOCHS, BATCH_SIZE,
                     LEARNING_RATE, WARMUP_RATIO, MIN_LR_RATIO, WEIGHT_DECAY,
                     TRAIN_EVAL_SIZE, MIN_EPOCHS, EARLY_STOPPING_PATIENCE,
                     MAX_SEQ_LEN, DEFAULT_TOKENIZER_MIN_FREQ, DEFAULT_VOCAB_SIZE)
from .data import resolve_data_paths
from .inference import infer
from .pre_training.trainer import pre_train


def build_parser():
    parser = argparse.ArgumentParser(description="MiniGPT 通用文本预训练（pre train）与推理")
    commands = parser.add_subparsers(dest="command")
    training = commands.add_parser("preTrain", help="预训练（pre train）；不传子命令时也进入预训练")
    training.add_argument("--train-data", "--data", dest="train_data", type=Path,
                          help="训练文件或目录；--data 是兼容别名。均未指定时使用 .data/corpus/train")
    training.add_argument("--eval-data", type=Path,
                          help="验证文件或目录；均未指定时使用 .data/corpus/eval；仅训练输入时自动拆分")
    training.add_argument("--data-workers", type=int, default=8,
                          help="每个集合并发读取文件数，默认 8；训练/验证也并发加载")
    training.add_argument("--text-field", default="text", help="表格或 JSON 的文本字段名")
    training.add_argument("--tokenizer", choices=("hybrid", "char"), default="hybrid",
                          help="默认高频字符 + UTF-8 字节回退；char 保留旧字符建表方式")
    training.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
                          help="hybrid 词表上限（含 260 个基础 token），不根据验证集建表")
    training.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN,
                          help="通用预训练序列长度，不能超过原模型 MAX_SEQ_LEN")
    training.add_argument("--poetry", action="store_true",
                          help="显式使用旧五绝数据筛选及逐首编码，便于旧实验对照")
    training.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                          help="预训练（pre train）检查点输出目录，默认 checkpoints/pre_train")
    training.add_argument("--vocab", type=Path,
                          help="只读加载已有 JSON tokenizer；不指定时只用训练集建表，旧词表不会被新默认词表覆盖")
    training.add_argument("--resume", type=Path, help="从预训练 last.pt 恢复；复用保存的数据划分与训练参数")
    training.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="目标总轮数，不是新增轮数")
    training.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    training.add_argument("--precision", choices=("auto", "bf16", "fp32"), default=None,
                          help="默认 auto：支持的 CUDA GPU 使用 BF16，否则 FP32；恢复时默认复用保存模式，可显式覆盖")
    training.add_argument("--lr", type=float, default=LEARNING_RATE, help="warmup 后的峰值学习率")
    training.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO,
                          help="线性预热占计划训练步数的比例 [0,1)，默认 0.05；0 禁用预热")
    training.add_argument("--min-lr-ratio", type=float, default=MIN_LR_RATIO,
                          help="最低学习率 / 峰值学习率 [0,1]，默认 0.1")
    training.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                          help="AdamW 权重衰减，默认 0.01；恢复训练时复用检查点中的值")
    training.add_argument("--train-eval-size", type=int, default=TRAIN_EVAL_SIZE,
                          help="固定训练评估子集大小（首），默认 1024；0 表示全量，恢复时复用保存值")
    training.add_argument("--valid-ratio", type=float, default=0.1,
                          help="仅训练输入时拆出验证集的比例，默认 0.1；已有独立验证输入时不重新拆分")
    training.add_argument("--min-freq", type=int, default=DEFAULT_TOKENIZER_MIN_FREQ,
                          help="高频字符阈值；hybrid 的低频字回退 UTF-8 字节，不丢为 UNK")
    training.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    training.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    training.add_argument("--max-documents", "--max-poems", dest="max_documents", type=int, default=0,
                          help="限制文档数做冒烟测试；0 为全量，旧 --max-poems 作为别名")
    inference = commands.add_parser("infer", help="加载预训练参数续写普通文本")
    inference.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT / "best.pt")
    inference.add_argument("--prompt", default="你好", help="文本前缀（保留空白和换行）；空串从 BOS 生成")
    inference.add_argument("--interactive", action="store_true")
    inference.add_argument("--num-samples", type=int, default=1)
    inference.add_argument("--temperature", type=float, default=0.8)
    inference.add_argument("--top-k", type=int, default=30, help="0 表示不做 top-k 截断")
    inference.add_argument("--max-new-tokens", type=int, default=64)
    generation_format = inference.add_mutually_exclusive_group()
    generation_format.add_argument("--poetry", action="store_true",
                                   help="仅旧字符词表可选：约束五绝格式；默认不做诗词约束")
    generation_format.add_argument("--free-form", action="store_true",
                                   help="兼容旧选项；普通文本续写已是默认行为")
    for command in (training, inference):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv if argv else ["preTrain"])
    if args.command == "preTrain":
        if (args.epochs < 1 or args.batch_size < 1 or args.min_freq < 1
                or args.patience < 1 or args.min_epochs < 0 or args.train_eval_size < 0
                or not math.isfinite(args.lr) or args.lr <= 0
                or not math.isfinite(args.warmup_ratio) or not 0 <= args.warmup_ratio < 1
                or not math.isfinite(args.min_lr_ratio) or not 0 <= args.min_lr_ratio <= 1
                or not math.isfinite(args.weight_decay) or args.weight_decay < 0
                or not 0 < args.valid_ratio < 1
                or args.max_documents < 0 or args.max_documents == 1
                or not 1 <= args.seq_len <= MAX_SEQ_LEN
                or args.vocab_size < 260 or not args.text_field or args.data_workers < 1):
            parser.error("训练参数无效：轮数/批次/词频/耐心值须为正，train-eval-size 须非负，weight-decay 须非负且有限，warmup-ratio 在 [0,1)，min-lr-ratio 在 [0,1]，验证比例在 (0,1)，max-documents 为 0 或 >=2，seq-len 在模型上限内，vocab-size >=260，data-workers 须为正整数")
        if not args.resume:
            try:
                args.train_data, args.eval_data = resolve_data_paths(args.train_data, args.eval_data)
            except ValueError as error:
                parser.error(str(error))
    else:
        if (args.num_samples < 1 or args.max_new_tokens < 1 or args.top_k < 0
                or not math.isfinite(args.temperature) or args.temperature <= 0):
            parser.error("推理参数无效：样本数/生成长度/温度须为正，top-k >= 0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请使用 --device cpu 或检查 PyTorch 环境")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"运行设备: {device}")
    if device.type == "cuda":
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    try:
        pre_train(args, device) if args.command == "preTrain" else infer(args, device)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
