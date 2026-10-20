"""Token-normalized micro-batch training with update-boundary resumability."""
import math
from contextlib import nullcontext

import torch
from torch import nn

from .precision import resolve_precision


def run_epoch(model, sequences, batch_size, device, optimizer=None,
              label_smoothing=0.0, scheduler=None, precision="auto", *,
              grad_accum_steps=1, max_updates=None, max_tokens=None,
              progress=None, on_update=None):
    """Return the running epoch mean, optionally stopping at an optimizer boundary.

    Accumulation windows include up to grad_accum_steps micro-batches. Each
    backward is weighted by its valid target count / the entire window count,
    including short final batches and -100 padding. No activation graph survives
    a micro-batch. progress contains the CPU permutation, next micro-batch cursor,
    loss totals and completed updates/tokens; save it only from on_update.
    max_updates/max_tokens are budgets for THIS call, not lifetime counters.
    The token budget is soft: a started update window is always completed.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    if type(grad_accum_steps) is not int or grad_accum_steps < 1:
        raise ValueError("grad_accum_steps 必须为正整数")
    for name, value in (("max_updates", max_updates), ("max_tokens", max_tokens)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} 必须为正整数或 None")
    precision = resolve_precision(device, precision)
    training = optimizer is not None
    model.train(training)
    state = progress if progress is not None else {}
    batches = math.ceil(len(sequences) / batch_size)
    if not state:
        state.update(order=torch.randperm(len(sequences)) if training else torch.arange(len(sequences)),
                     next_batch=0, total_loss=0.0, total_weight=0, updates=0, tokens=0)
    order = state["order"]
    cursor = state["next_batch"]
    if (not isinstance(order, torch.Tensor) or order.device.type != "cpu"
            or order.dtype != torch.long or order.ndim != 1 or len(order) != len(sequences)
            or type(cursor) is not int or not 0 <= cursor <= batches):
        raise ValueError("epoch progress 的数据顺序或微批游标无效")
    if len(order) and (int(order.min()) < 0 or int(order.max()) >= len(sequences)
                       or len(torch.unique(order)) != len(order)):
        raise ValueError("epoch progress 的数据顺序必须是完整排列")
    start_updates, start_tokens = state["updates"], state["tokens"]
    padded = bool((sequences == -100).any())
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    window_size = grad_accum_steps if training else 1
    # Isolate callers' outer autocast; backward, clipping and updates stay outside AMP.
    with torch.autocast(device_type=torch.device(device).type, enabled=False), \
            torch.set_grad_enabled(training):
        while cursor < batches:
            end = min(cursor + window_size, batches)
            # Count on the source (normally CPU), without retaining GPU activations.
            window_indices = order[cursor * batch_size:min(end * batch_size, len(order))]
            counts = (sequences[window_indices, 1:] != -100).sum(dim=1)
            window_tokens = int(counts.sum())
            if not window_tokens:
                cursor = end
                state["next_batch"] = cursor
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            window_loss, window_weight = 0.0, 0
            for micro in range(cursor, end):
                offset = (micro - cursor) * batch_size
                valid_tokens = int(counts[offset:offset + batch_size].sum())
                if not valid_tokens:
                    continue
                batch = sequences[order[micro * batch_size:(micro + 1) * batch_size]].to(device)
                inputs, labels = batch[:, :-1].clamp_min(0), batch[:, 1:]
                context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)
                           if precision == "bf16" else nullcontext())
                with context:
                    logits = model(inputs)
                loss_logits = logits.float() if precision == "bf16" else logits
                loss = criterion(loss_logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError("出现非有限损失，请检查数据或降低学习率")
                if training:
                    # Preserve the original arithmetic when accumulation is disabled.
                    backward_loss = loss if valid_tokens == window_tokens else loss * (valid_tokens / window_tokens)
                    backward_loss.backward()
                weight = valid_tokens if padded else len(batch)
                window_loss += loss.item() * weight
                window_weight += weight
                # Release even the final logits before optimizer/evaluation/checkpoint callbacks.
                del logits, loss_logits, loss
                if training:
                    del backward_loss
            if training:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                state["updates"] += 1
                state["last_grad_norm"] = float(grad_norm)
            # Commit statistics only after a successful whole update window.
            state["total_loss"] += window_loss
            state["total_weight"] += window_weight
            state["tokens"] += window_tokens
            cursor = end
            state["next_batch"] = cursor
            if training and on_update is not None:
                on_update(state)
                model.train(training)
            if ((max_updates is not None and state["updates"] - start_updates >= max_updates)
                    or (max_tokens is not None and state["tokens"] - start_tokens >= max_tokens)):
                break
    if state["total_weight"] == 0:
        raise ValueError("没有可用于预测的有效 token")
    return state["total_loss"] / state["total_weight"]
