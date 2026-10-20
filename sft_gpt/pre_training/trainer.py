import math
import random
import shutil
import time
from pathlib import Path

import torch

from ..config import DEFAULT_VOCAB, DEFAULT_TEXT_VOCAB, MAX_SEQ_LEN, EARLY_STOPPING_DELTA, MAX_STEPS
from ..data import (load_train_eval_texts, make_sequences, make_pretrain_sequences,
                    select_train_eval_indices)
from ..models import MiniGPT
from ..tokenization import (CharTokenizer, HybridTokenizer, load_tokenizer, tokenizer_spec)
from .checkpoint import load_checkpoint, save_checkpoint
from .epoch import run_epoch
from .precision import resolve_precision
from .optimizer import build_optimizer
from .warmup_cosine_scheduler import WarmupCosineScheduler


def validate_training_sources(args):
    """Keep exact resume separate from a new, non-destructive data stage."""
    resume = getattr(args, "resume", None)
    init_from = getattr(args, "init_from", None)
    if resume is not None and init_from is not None:
        raise ValueError("--resume 与 --init-from 不能同时使用")
    if resume is not None and (getattr(args, "train_data", None) is not None
                               or getattr(args, "eval_data", None) is not None):
        raise ValueError("--resume 复用保存的数据，不能指定 --train-data/--data 或 --eval-data；换语料请使用 --init-from")
    if init_from is not None:
        if getattr(args, "train_data", None) is None:
            raise ValueError("--init-from 必须显式指定 --train-data（或 --data），不会隐式使用默认语料")
        if getattr(args, "output_dir", None) is None:
            raise ValueError("--init-from 必须显式指定新的 --output-dir")
        source = Path(init_from).resolve()
        output = Path(args.output_dir).resolve()
        if output == source.parent:
            raise ValueError("--init-from 的 --output-dir 不能是源检查点所在目录，请使用新的空目录")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("--init-from 的 --output-dir 必须不存在或为空目录，避免覆盖已有实验")


def pre_train(args, device):
    """运行自回归预训练（pre train），产出预训练检查点。"""
    validate_training_sources(args)
    init_from = getattr(args, "init_from", None)
    initialization = None
    requested_precision = getattr(args, "precision", None)
    if args.resume:
        model, tokenizer, checkpoint = load_checkpoint(args.resume, device)
        settings = dict(checkpoint["train_settings"])
        initialization = checkpoint.get("initialization")
        precision_policy = requested_precision or settings.get("precision", "auto")
        precision = resolve_precision(device, precision_policy)
        if "train_texts" in checkpoint:
            train_texts, valid_texts = checkpoint["train_texts"], checkpoint["valid_texts"]
        else:
            train_texts, valid_texts = checkpoint["train_poems"], checkpoint["valid_poems"]
        start_epoch = checkpoint["epoch"] + (1 if checkpoint.get("epoch_complete", True) else 0)
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
        bad_epochs, history = checkpoint["bad_epochs"], checkpoint["history"]
    else:
        precision_policy = requested_precision or "auto"
        precision = resolve_precision(device, precision_policy)
        if init_from is not None:
            model, tokenizer, source = load_checkpoint(init_from, device)
            if args.vocab is not None and tokenizer_spec(load_tokenizer(args.vocab)) != tokenizer_spec(tokenizer):
                raise ValueError("外部词表与初始化检查点 token ID 不一致；--init-from 必须复用原词表")
            initialization = dict(checkpoint=str(Path(init_from).resolve()),
                                  source_epoch=source.get("epoch"),
                                  source_global_step=source.get("global_step"),
                                  source_trained_tokens=source.get("trained_tokens"))
            # Retain weights/tokenizer only, not source texts, Adam tensors or RNG.
            del source
            print(f"继续预训练新阶段，初始化权重: {Path(init_from).resolve()}")
            print("复用原分词器；重新读取语料，重置优化器、学习率计划、预算计数、历史与早停。")
        train_texts, valid_texts = load_train_eval_texts(
            args.train_data, args.eval_data, text_field=args.text_field, seed=args.seed,
            valid_ratio=args.valid_ratio, max_documents=args.max_documents,
            workers=args.data_workers, poetry=args.poetry)
        if init_from is None:
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
                        label_smoothing=args.label_smoothing, patience=args.patience,
                        min_delta=EARLY_STOPPING_DELTA, min_epochs=args.min_epochs,
                        warmup_ratio=args.warmup_ratio, min_lr_ratio=args.min_lr_ratio,
                        train_eval_size=args.train_eval_size, train_eval_seed=args.seed,
                        sequence_mode="poetry" if args.poetry else "packed",
                        seq_len=args.seq_len, grad_accum_steps=args.grad_accum_steps,
                        max_steps=MAX_STEPS if args.max_steps is None else args.max_steps,
                        max_tokens=args.max_tokens or 0,
                        gradient_checkpointing=args.gradient_checkpointing,
                        optimizer_decay_groups=args.optimizer_decay_groups,
                        adam_betas=(0.9, args.adam_beta2), fused_adamw=args.fused_adamw,
                        eval_batch_size=args.eval_batch_size, eval_every_steps=args.eval_every_steps,
                        eval_max_batches=args.eval_max_batches, save_every_steps=args.save_every_steps)
        start_epoch, best_loss, best_epoch, bad_epochs, history = 1, float("inf"), 0, 0, []
    # 保存请求的策略，auto 恢复到不同硬件时可安全回退；旧检查点默认 auto。
    settings["precision"] = precision_policy
    print(f"训练/评估精度: {precision}（策略: {precision_policy}；模型权重保持 FP32）")
    if args.resume and history:
        print("恢复提示：更改精度或设备可能影响 loss 和早停比较；精确对照请保持相同设置。")
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
    if args.vocab is None and initialization is None:
        tokenizer.save(DEFAULT_TEXT_VOCAB if isinstance(tokenizer, HybridTokenizer) else DEFAULT_VOCAB)
    # Missing fields mean the legacy recipe, never today's larger-model defaults.
    legacy_defaults = dict(grad_accum_steps=1, max_steps=0, max_tokens=0,
                           gradient_checkpointing=False, optimizer_decay_groups=False,
                           adam_betas=(0.9, 0.999), fused_adamw=False,
                           eval_batch_size=settings["batch_size"], eval_every_steps=0,
                           eval_max_batches=0, save_every_steps=0)
    for name, value in legacy_defaults.items():
        settings.setdefault(name, value)
    if args.resume:
        for name in ("max_steps", "max_tokens"):
            override = getattr(args, name, None)
            if override is not None:
                settings[name] = override
        print("恢复沿用保存的 micro-batch、累积、优化器和评估设置；仅预算/精度可显式覆盖。")
    if hasattr(model, "decoder"):
        model.decoder.gradient_checkpointing = settings["gradient_checkpointing"]
    elif settings["gradient_checkpointing"]:
        raise ValueError("该模型没有可启用 gradient checkpointing 的 decoder")
    optimizer = build_optimizer(model, settings["lr"], settings["weight_decay"],
                                decay_groups=settings["optimizer_decay_groups"],
                                betas=tuple(settings["adam_betas"]),
                                fused=settings["fused_adamw"], device=device)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
            # CPU runs also save CUDA RNG on CUDA-capable hosts. Restore it even
            # when this run trains on CPU, rather than retaining the new CLI seed.
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
    micro_batches = math.ceil(len(train_sequences) / settings["batch_size"])
    updates_per_epoch = math.ceil(micro_batches / settings["grad_accum_steps"])
    global_step = (checkpoint.get("global_step", (checkpoint.get("scheduler_state") or {}).get(
        "completed_steps", checkpoint["epoch"] * updates_per_epoch)) if args.resume else 0)
    train_token_count = int((train_sequences[:, 1:] != -100).sum())
    trained_tokens = checkpoint.get("trained_tokens", checkpoint["epoch"] * train_token_count) if args.resume else 0
    epoch_progress = checkpoint.get("epoch_progress", {}) if args.resume and not checkpoint.get("epoch_complete", True) else {}
    planned_steps = (args.epochs - start_epoch + 1) * updates_per_epoch
    if epoch_progress:
        planned_steps -= epoch_progress["updates"]
    if settings["max_steps"]:
        planned_steps = min(planned_steps, settings["max_steps"] - global_step)
    if settings["max_tokens"]:
        # Token stopping is exact at update boundaries; this is a scheduling estimate
        # because shuffled padding and epoch tails change actual tokens per update.
        mean_tokens = train_token_count / updates_per_epoch
        planned_steps = min(planned_steps, math.ceil((settings["max_tokens"] - trained_tokens) / mean_tokens))
    if planned_steps < 1:
        raise ValueError("训练预算已完成；继续训练请显式增加 --max-steps / --max-tokens（0 取消限制）")
    effective_batch = settings["batch_size"] * settings["grad_accum_steps"]
    print(f"micro-batch={settings['batch_size']} | grad_accum={settings['grad_accum_steps']} | "
          f"effective batch={effective_batch} | 每轮更新={updates_per_epoch} | "
          f"label_smoothing={settings['label_smoothing']} | checkpointing={settings['gradient_checkpointing']}")
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
    # A first-epoch step checkpoint may precede the first full validation:
    # best_epoch=0 legitimately means there is no best.pt to copy yet.
    if args.resume and best_epoch > 0 and args.output_dir.resolve() != args.resume.resolve().parent:
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
        del best_checkpoint
    # Step-level subsets are fixed and diagnostic only: best/early stopping always
    # use a full validation pass, so subset/full losses are never mixed.
    subset_size = settings["eval_max_batches"] * settings["eval_batch_size"]
    subset_indices = select_train_eval_indices(len(valid_sequences), subset_size,
                                               settings["train_eval_seed"] + 1)
    valid_step_sequences = valid_sequences[subset_indices]
    last_step_eval = checkpoint.get("last_step_eval") if args.resume else None
    if args.resume:
        # Model/Adam state has been loaded; do not retain another CPU copy for
        # the entire run. Texts, cursor and settings have their own references.
        del checkpoint

    def budget_finished():
        return ((settings["max_steps"] and global_step >= settings["max_steps"])
                or (settings["max_tokens"] and trained_tokens >= settings["max_tokens"]))

    for epoch in range(start_epoch, args.epochs + 1):
        progress = epoch_progress
        epoch_progress = {}
        lr_start = progress.get("lr_start", scheduler.get_last_lr()[0])
        base_updates = global_step - progress.get("updates", 0)
        base_tokens = trained_tokens - progress.get("tokens", 0)
        started = time.perf_counter()
        call_start_tokens = trained_tokens
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def metadata(complete):
            result = dict(epoch=epoch, best_loss=best_loss, best_epoch=best_epoch,
                          bad_epochs=bad_epochs, history=history, train_settings=settings,
                          scheduler_state=scheduler.state_dict(), train_eval_indices=train_eval_indices,
                          global_step=global_step, trained_tokens=trained_tokens,
                          epoch_complete=complete, epoch_progress={} if complete else progress,
                          last_step_eval=last_step_eval)
            if initialization is not None:
                result["initialization"] = initialization
            if sequence_mode == "poetry":
                result.update(train_poems=train_texts, valid_poems=valid_texts)
            else:
                result.update(train_texts=train_texts, valid_texts=valid_texts)
            return result

        def after_update(state):
            nonlocal global_step, trained_tokens, last_step_eval
            state["lr_start"] = lr_start
            global_step = base_updates + state["updates"]
            trained_tokens = base_tokens + state["tokens"]
            # The final boundary is evaluated/saved once below, after epoch history.
            if state["next_batch"] >= micro_batches or budget_finished():
                return
            interval = settings["eval_every_steps"]
            if interval and global_step % interval == 0:
                # Nested validation must not leave the outer training loop in eval mode.
                training = model.training
                try:
                    value = run_epoch(model, valid_step_sequences, settings["eval_batch_size"],
                                      device, precision=precision)
                finally:
                    model.train(training)
                last_step_eval = dict(global_step=global_step, valid_subset_loss=value)
                print(f"Step {global_step} | valid_subset={value:.4f} | tokens={trained_tokens}", flush=True)
            interval = settings["save_every_steps"]
            if interval and global_step % interval == 0:
                save_checkpoint(args.output_dir / "last.pt", model, tokenizer, optimizer, **metadata(False))

        remaining_steps = settings["max_steps"] - global_step if settings["max_steps"] else None
        remaining_tokens = settings["max_tokens"] - trained_tokens if settings["max_tokens"] else None
        objective = run_epoch(model, train_sequences, settings["batch_size"], device,
                              optimizer, settings["label_smoothing"], scheduler, precision=precision,
                              grad_accum_steps=settings["grad_accum_steps"],
                              max_updates=remaining_steps, max_tokens=remaining_tokens,
                              progress=progress, on_update=after_update)
        complete = progress["next_batch"] >= micro_batches
        train_eval_loss = run_epoch(model, train_eval_sequences, settings["eval_batch_size"],
                                    device, precision=precision)
        valid_loss = run_epoch(model, valid_sequences, settings["eval_batch_size"],
                               device, precision=precision)
        improved = valid_loss < best_loss - settings["min_delta"]
        if improved:
            best_loss, best_epoch = valid_loss, epoch
            if complete:
                bad_epochs = 0
        elif complete and epoch >= settings["min_epochs"]:
            bad_epochs += 1
        record = dict(epoch=epoch, objective=objective, train_eval_loss=train_eval_loss,
                      valid_loss=valid_loss, lr_start=lr_start, lr_next=scheduler.get_last_lr()[0])
        if not complete:
            record.update(epoch_complete=False, global_step=global_step)
        history.append(record)
        # Keep runtime metrics out of reproducible checkpoint history.
        elapsed = time.perf_counter() - started
        rate = (trained_tokens - call_start_tokens) / max(elapsed, 1e-9)
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0
        if improved:
            save_checkpoint(args.output_dir / "best.pt", model, tokenizer, optimizer, **metadata(complete))
        save_checkpoint(args.output_dir / "last.pt", model, tokenizer, optimizer, **metadata(complete))
        print(f"Epoch {epoch:4d} | step={global_step} | objective={objective:.4f} | "
              f"train_eval={train_eval_loss:.4f} | valid={valid_loss:.4f} | "
              f"best={best_loss:.4f} (epoch {best_epoch}) | tokens={trained_tokens} | "
              f"tokens/s={rate:.0f} (含评估) | peak_allocated={peak:.2f} GiB | "
              f"grad_norm={progress.get('last_grad_norm', 0.0):.4f} | "
              f"lr_start={lr_start:.6g} | lr_next={scheduler.get_last_lr()[0]:.6g}", flush=True)
        if budget_finished():
            print("达到训练更新步/token预算；已保存更新边界，可显式扩展预算后恢复。")
            break
        if epoch >= settings["min_epochs"] and bad_epochs >= settings["patience"]:
            print("触发早停，推理请使用 best.pt。")
            break
