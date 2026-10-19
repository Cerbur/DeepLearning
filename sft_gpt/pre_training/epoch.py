import torch
from torch import nn


def run_epoch(model, sequences, batch_size, device, optimizer=None,
              label_smoothing=0.0, scheduler=None):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(sequences)) if training else torch.arange(len(sequences))
    total_loss = 0.0
    total_weight = 0
    padded = bool((sequences == -100).any())
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    with torch.set_grad_enabled(training):
        for start in range(0, len(order), batch_size):
            batch = sequences[order[start:start + batch_size]].to(device)
            # -100 只在右侧补齐；忽略其标签，不能送负数给 Embedding。
            inputs, labels = batch[:, :-1].clamp_min(0), batch[:, 1:]
            valid_tokens = int((labels != -100).sum())
            if valid_tokens == 0:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
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
