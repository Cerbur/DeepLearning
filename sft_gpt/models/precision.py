"""显式 FP32 数值稳定边界；线性层继续由调用方的 autocast 选择精度。"""
import torch


def promote_for_stability(tensor):
    """低精度浮点升到 FP32；不降低显式 FP64 数值参考的精度。"""
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor
