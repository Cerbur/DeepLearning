import math

from ..config import WARMUP_RATIO, MIN_LR_RATIO


class WarmupCosineScheduler:
    """按更新步数线性预热、单次余弦退火；构造准备首批，step 准备下一批。"""

    def __init__(self, optimizer, total_steps, warmup_ratio=WARMUP_RATIO,
                 min_lr_ratio=MIN_LR_RATIO):
        if (type(total_steps) is not int or total_steps < 1
                or not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1):
            raise ValueError("total_steps 须为正整数，warmup_ratio 须在 [0,1)")
        self.optimizer = optimizer
        # 小样本时尽量保留峰值、最低值两个余弦阶段位置；单步时直接用峰值。
        warmup_steps = min(max(0, total_steps - 2), math.ceil(total_steps * warmup_ratio))
        self.load_state_dict(dict(
            total_steps=total_steps, warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio, completed_steps=0,
            base_lrs=[group["lr"] for group in optimizer.param_groups]))

    @property
    def total_steps(self):
        return self._state["total_steps"]

    @property
    def warmup_steps(self):
        return self._state["warmup_steps"]

    @property
    def min_lr_ratio(self):
        return self._state["min_lr_ratio"]

    @property
    def completed_steps(self):
        return self._state["completed_steps"]

    @property
    def base_lrs(self):
        return list(self._state["base_lrs"])

    def _apply_lr(self):
        step = self.completed_steps
        if step >= self.total_steps:
            factor = self.min_lr_ratio
        elif step < self.warmup_steps:
            factor = (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps - 1)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)) / 2
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self):
        self._state["completed_steps"] += 1
        self._apply_lr()

    def get_last_lr(self):
        """返回已应用到 optimizer、供下一批使用的各参数组学习率。"""
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_info(self):
        """返回独立的日志快照，不向调用者暴露内部可变状态。"""
        phase = ("finished" if self.completed_steps >= self.total_steps else
                 "warmup" if self.completed_steps < self.warmup_steps else "cosine")
        return dict(total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                    completed_steps=self.completed_steps, phase=phase,
                    peak_lrs=self.base_lrs,
                    min_lrs=[lr * self.min_lr_ratio for lr in self.base_lrs],
                    current_lrs=self.get_last_lr())

    def state_dict(self):
        # 保持旧检查点字段不变，但不再要求状态字段等同于构造参数。
        return dict(self._state, base_lrs=self.base_lrs)

    def load_state_dict(self, state):
        """先校验完整状态，再恢复原曲线和进度并同步 optimizer 学习率。"""
        try:
            total_steps = state["total_steps"]
            warmup_steps = state["warmup_steps"]
            completed_steps = state["completed_steps"]
            min_lr_ratio = state["min_lr_ratio"]
            base_lrs = list(state["base_lrs"])
            valid = (
                type(total_steps) is int and total_steps >= 1
                and type(warmup_steps) is int and 0 <= warmup_steps < total_steps
                and type(completed_steps) is int and completed_steps >= 0
                and math.isfinite(min_lr_ratio) and 0 <= min_lr_ratio <= 1
                and len(base_lrs) == len(self.optimizer.param_groups)
                and all(math.isfinite(lr) and lr >= 0 for lr in base_lrs))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("学习率调度状态无效或不完整") from error
        if not valid:
            raise ValueError("学习率调度状态无效或参数组数量不匹配")
        self._state = dict(total_steps=total_steps, warmup_steps=warmup_steps,
                           completed_steps=completed_steps, min_lr_ratio=min_lr_ratio,
                           base_lrs=base_lrs)
        self._apply_lr()
