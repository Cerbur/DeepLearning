"""AdamW construction with deterministic decay groups and explicit fused opt-in."""
import inspect

import torch


def build_optimizer(model, lr, weight_decay, *, decay_groups=True,
                    betas=(0.9, 0.95), fused=False, device=None):
    """Build AdamW without moving parameters or silently changing hyperparameters.

    Matrix/tensor parameters (ndim >= 2) receive weight decay; vectors and
    scalars do not. Shared Parameters appear once, in first-occurrence order.
    ``decay_groups=False`` retains one group in ``model.parameters()`` order,
    including frozen parameters, for loading legacy optimizer state dicts.
    Legacy callers must explicitly pass their original betas, e.g. (0.9, 0.999).

    ``fused=True`` is strict opt-in: all parameters must already be floating
    CUDA tensors, CUDA must be available, and this PyTorch must expose fused
    AdamW. ``device`` optionally validates the requested execution device; it
    never moves the model. Unsupported fused configurations raise rather than
    falling back to a different optimizer. Backend errors at step() propagate.
    """
    parameters = []
    seen = set()
    for parameter in model.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            parameters.append(parameter)

    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if fused:
        if device is not None and torch.device(device).type != "cuda":
            raise ValueError("fused AdamW requires a CUDA device; no CPU fallback is performed")
        if not torch.cuda.is_available():
            raise ValueError("fused AdamW requires an available CUDA runtime")
        if any(parameter.device.type != "cuda" or not parameter.is_floating_point()
               for parameter in parameters):
            raise ValueError("fused AdamW requires all parameters to be floating CUDA tensors")
        if not supports_fused:
            raise RuntimeError("This PyTorch version does not support fused AdamW")

    if decay_groups:
        groups = [
            {"params": [p for p in parameters if p.ndim >= 2],
             "weight_decay": weight_decay},
            {"params": [p for p in parameters if p.ndim < 2],
             "weight_decay": 0.0},
        ]
    else:
        groups = [{"params": parameters, "weight_decay": weight_decay}]

    # Omit the keyword unless explicitly opted in: legacy AdamW stores
    # fused=None, not False, in param_groups/state_dict metadata.
    kwargs = {"fused": True} if fused else {}
    try:
        return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay,
                                 betas=betas, **kwargs)
    except (TypeError, RuntimeError) as error:
        if fused:
            raise RuntimeError("Unable to construct fused AdamW; no fallback is performed") from error
        raise
