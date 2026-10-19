import math
import random
import shutil

import torch

from ..config import DEFAULT_VOCAB, DEFAULT_TEXT_VOCAB, MAX_SEQ_LEN, EARLY_STOPPING_DELTA, LABEL_SMOOTHING
from ..data import (load_poems, load_texts, make_sequences, make_pretrain_sequences,
                    select_train_eval_indices)
from ..models import MiniGPT
from ..tokenization import (CharTokenizer, HybridTokenizer, load_tokenizer, tokenizer_spec)
from .checkpoint import load_checkpoint, save_checkpoint
from .epoch import run_epoch
from .warmup_cosine_scheduler import WarmupCosineScheduler


def pre_train(args, device):
    """运行自回归预训练（pre train），产出预训练检查点。"""
    if args.resume:
        model, tokenizer, checkpoint = load_checkpoint(args.resume, device)
        settings = dict(checkpoint["train_settings"])
        if "train_texts" in checkpoint:
            train_texts, valid_texts = checkpoint["train_texts"], checkpoint["valid_texts"]
        else:
            train_texts, valid_texts = checkpoint["train_poems"], checkpoint["valid_poems"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
        bad_epochs, history = checkpoint["bad_epochs"], checkpoint["history"]
    else:
        texts = load_poems(args.data) if args.poetry else load_texts(args.data, args.text_field)
        random.Random(args.seed).shuffle(texts)
        if args.max_documents:
            texts = texts[:args.max_documents]
        valid_count = max(1, int(len(texts) * args.valid_ratio))
        valid_texts, train_texts = texts[:valid_count], texts[valid_count:]
        if args.vocab is not None:
            tokenizer = load_tokenizer(args.vocab)
        elif args.poetry or args.tokenizer == "char":
            tokenizer = CharTokenizer.build(train_texts, args.min_freq)
        else:
            tokenizer = HybridTokenizer.build(train_texts, args.min_freq, args.vocab_size)
        if len(tokenizer.tokens) <= 3:
            raise ValueError("正文词表为空，请降低 --min-freq 或使用 hybrid tokenizer")
        model = MiniGPT(len(tokenizer.tokens)).to(device)
        settings = dict(batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                        label_smoothing=LABEL_SMOOTHING, patience=args.patience,
                        min_delta=EARLY_STOPPING_DELTA, min_epochs=args.min_epochs,
                        warmup_ratio=args.warmup_ratio, min_lr_ratio=args.min_lr_ratio,
                        train_eval_size=args.train_eval_size, train_eval_seed=args.seed,
                        sequence_mode="poetry" if args.poetry else "packed",
                        seq_len=args.seq_len)
        start_epoch, best_loss, best_epoch, bad_epochs, history = 1, float("inf"), 0, 0, []
    if args.resume and args.vocab is not None:
        if tokenizer_spec(load_tokenizer(args.vocab)) != tokenizer_spec(tokenizer):
            raise ValueError("外部词表与检查点 token ID 不一致；恢复训练必须复用原词表")
    sequence_mode = settings.get("sequence_mode", "poetry")  # 无此字段的旧 checkpoint
    if sequence_mode not in ("poetry", "packed"):
        raise ValueError("检查点 sequence_mode 不受支持")
    if sequence_mode == "poetry":
        if not isinstance(tokenizer, CharTokenizer) or not set("，。").issubset(tokenizer.token_ids):
            raise ValueError("旧五绝模式要求字符词表及中文标点 ， 和 。；不会自动改变 token ID")
    # 显式词表是只读输入；实验副本由 save_checkpoint 导出。
    if args.vocab is None:
        tokenizer.save(DEFAULT_TEXT_VOCAB if isinstance(tokenizer, HybridTokenizer) else DEFAULT_VOCAB)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  weight_decay=settings["weight_decay"])
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if sequence_mode == "poetry":
        train_sequences = make_sequences(train_texts, tokenizer)
        valid_sequences = make_sequences(valid_texts, tokenizer)
    else:
        train_sequences = make_pretrain_sequences(train_texts, tokenizer, settings["seq_len"])
        valid_sequences = make_pretrain_sequences(valid_texts, tokenizer, settings["seq_len"])
    settings.setdefault("train_eval_size", args.train_eval_size)
    settings.setdefault("train_eval_seed", args.seed)
    train_eval_indices = select_train_eval_indices(
        len(train_sequences), settings["train_eval_size"], settings["train_eval_seed"],
        checkpoint.get("train_eval_indices") if args.resume else None)
    train_eval_sequences = train_sequences[train_eval_indices]
    if args.resume and "train_eval_indices" not in checkpoint:
        print("旧检查点无训练评估索引：建立固定子集；旧 train_loss 历史保留，新记录使用 train_eval_loss。")
    print(f"固定训练评估子集: {len(train_eval_sequences)}/{len(train_sequences)}（仍参与训练）")
    print(f"训练文档: {len(train_texts)}，验证文档: {len(valid_texts)}，词表: {len(tokenizer.tokens)} "
          f"({type(tokenizer).__name__})，模式: {sequence_mode}")
    print(f"训练序列: {len(train_sequences)}，验证序列: {len(valid_sequences)}；"
          f"模型最大上下文仍为 {MAX_SEQ_LEN} token，未修改网络")
    for split_name, sequences in (("训练", train_sequences), ("验证", valid_sequences)):
        labels = sequences[:, 1:]
        count = int((labels != -100).sum())
        unknown = int((labels == tokenizer.unk).sum())
        print(f"{split_name}有效预测 token: {count}，padding={1 - count / labels.numel():.2%}，"
              f"UNK={unknown / count:.2%}")
    print(f"预训练（pre train）检查点输出目录: {args.output_dir.resolve()}")
    if start_epoch > args.epochs:
        raise ValueError(f"--epochs 是总轮数，恢复训练时必须大于已有轮数 {start_epoch - 1}")
    settings.setdefault("warmup_ratio", args.warmup_ratio)
    settings.setdefault("min_lr_ratio", args.min_lr_ratio)
    planned_steps = (args.epochs - start_epoch + 1) * math.ceil(
        len(train_sequences) / settings["batch_size"])
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps=planned_steps,
        warmup_ratio=settings["warmup_ratio"], min_lr_ratio=settings["min_lr_ratio"])
    if args.resume:
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print("恢复已保存的学习率曲线；超出原计划步数后保持最低学习率。")
        else:
            print("旧检查点无调度状态：从当前学习率开始，为剩余训练建立 warmup + 余弦曲线。")
    info = scheduler.get_info()
    print(f"学习率策略: warmup={info['warmup_steps']} 步 | "
          f"total={info['total_steps']} 步 | peak={info['peak_lrs'][0]:.6g} | "
          f"min={info['min_lrs'][0]:.6g} | completed={info['completed_steps']} | "
          f"phase={info['phase']}")
    if args.resume and args.output_dir.resolve() != args.resume.resolve().parent:
        source_best = args.resume.resolve().parent / "best.pt"
        if not source_best.is_file():
            raise ValueError("恢复到新目录需要原目录的 best.pt；请保留原输出目录继续训练")
        best_checkpoint = torch.load(source_best, map_location="cpu", weights_only=False)
        if (best_checkpoint.get("epoch") != best_epoch
                or best_checkpoint.get("tokens") != tokenizer.tokens
                or (best_checkpoint.get("tokenizer") is not None
                    and best_checkpoint["tokenizer"] != tokenizer_spec(tokenizer))
                or best_checkpoint.get("best_loss") != best_loss):
            raise ValueError("原目录 best.pt 与恢复检查点不匹配，请使用原输出目录")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_best, args.output_dir / "best.pt")
    for epoch in range(start_epoch, args.epochs + 1):
        lr_start = scheduler.get_last_lr()[0]
        objective = run_epoch(model, train_sequences, settings["batch_size"], device,
                              optimizer, settings["label_smoothing"], scheduler)
        train_eval_loss = run_epoch(model, train_eval_sequences, settings["batch_size"], device)
        valid_loss = run_epoch(model, valid_sequences, settings["batch_size"], device)
        improved = valid_loss < best_loss - settings["min_delta"]
        if improved:
            best_loss, best_epoch, bad_epochs = valid_loss, epoch, 0
        elif epoch >= settings["min_epochs"]:
            bad_epochs += 1
        history.append(dict(epoch=epoch, objective=objective,
                            train_eval_loss=train_eval_loss, valid_loss=valid_loss,
                            lr_start=lr_start, lr_next=scheduler.get_last_lr()[0]))
        metadata = dict(epoch=epoch, best_loss=best_loss, best_epoch=best_epoch,
                        bad_epochs=bad_epochs, history=history, train_settings=settings,
                        scheduler_state=scheduler.state_dict(),
                        train_eval_indices=train_eval_indices)
        if sequence_mode == "poetry":
            metadata.update(train_poems=train_texts, valid_poems=valid_texts)
        else:
            metadata.update(train_texts=train_texts, valid_texts=valid_texts)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", model, tokenizer, optimizer, **metadata)
        save_checkpoint(args.output_dir / "last.pt", model, tokenizer, optimizer, **metadata)
        print(f"Epoch {epoch:4d} | objective={objective:.4f} | train_eval={train_eval_loss:.4f} | "
              f"valid={valid_loss:.4f} | best={best_loss:.4f} (epoch {best_epoch}) | "
              f"lr_start={lr_start:.6g} | lr_next={scheduler.get_last_lr()[0]:.6g}", flush=True)
        if epoch >= settings["min_epochs"] and bad_epochs >= settings["patience"]:
            print("触发早停，推理请使用 best.pt。")
            break
