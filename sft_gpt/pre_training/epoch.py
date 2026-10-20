from contextlib import nullcontext

import torch
from torch import nn

from .precision import resolve_precision


def run_epoch(model, sequences, batch_size, device, optimizer=None,
              label_smoothing=0.0, scheduler=None, precision="auto"):
    precision = resolve_precision(device, precision)
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(sequences)) if training else torch.arange(len(sequences))
    total_loss = 0.0
    total_weight = 0
    padded = bool((sequences == -100).any())
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    # 屏蔽调用方外层 AMP，确保显式 FP32 和反向传播不会继承 autocast。
    with torch.autocast(device_type=torch.device(device).type, enabled=False), \
            torch.set_grad_enabled(training):
        for start in range(0, len(order), batch_size):
            batch = sequences[order[start:start + batch_size]].to(device)
            # -100 只在右侧补齐；忽略其标签，不能送负数给 Embedding。
            inputs, labels = batch[:, :-1].clamp_min(0), batch[:, 1:]
            valid_tokens = int((labels != -100).sum())
            if valid_tokens == 0:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            # 仅前向和 loss 启用 autocast；CE 自动以 FP32 计算，反向在上下文外。
            # BF16 不需要 FP16 的 GradScaler，参数/梯度及 AdamW 状态仍为 FP32。
            # 外层上下文会延长 autocast 缓存寿命；禁用权重缓存，避免 step 后复用旧权重。
            context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)
                       if precision == "bf16" else nullcontext())
            with context:
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("出现非有限损失，请检查数据或降低学习率")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            weight = valid_tokens if padded else len(batch)
            total_loss += loss.item() * weight
            total_weight += weight
    if total_weight == 0:
        raise ValueError("没有可用于预测的有效 token")
    return total_loss / total_weight
