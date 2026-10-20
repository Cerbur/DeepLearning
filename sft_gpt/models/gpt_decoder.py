import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import D_MODEL
from .gpt_block import GPTBlock
from .rms_norm import RMSNorm


class GPTDecoder(nn.Module):
    # Runtime-only switch: no new parameters/buffers or initialization changes.
    gradient_checkpointing = False

    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([GPTBlock() for _ in range(num_layers)])  # 堆叠多个 GPTBlock
        self.norm = RMSNorm(D_MODEL)  # 最后的归一化层

    def forward(self, x):
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            if checkpointing:
                # PyTorch 2.0 non-reentrant recomputation must not reuse cached
                # autocast weight casts from the original forward. Preserve the
                # surrounding enabled/dtype state and the blocks' FP32 islands.
                autocast_enabled = (torch.is_autocast_enabled() if x.is_cuda
                                    else torch.is_autocast_cpu_enabled())
                with torch.autocast(device_type=x.device.type, enabled=autocast_enabled,
                                    cache_enabled=False):
                    x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=True)
            else:
                x = layer(x)  # 逐层前向传播
        return self.norm(x)  # 最后归一化输出
