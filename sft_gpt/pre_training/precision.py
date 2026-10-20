"""CUDA BF16 autocast 策略；不转换模型权重或优化器状态。"""
import torch


def resolve_precision(device, precision="auto"):
    """auto 在支持 BF16 的 CUDA 上启用，否则使用 FP32；显式 BF16 严格检查。"""
    if precision not in ("auto", "bf16", "fp32"):
        raise ValueError("precision 必须为 auto、bf16 或 fp32")
    device = torch.device(device)
    if precision == "fp32":
        return "fp32"
    supported = False
    if device.type == "cuda" and torch.cuda.is_available():
        # is_bf16_supported 查询当前设备；兼容 cuda:1 等非默认设备。
        with torch.cuda.device(device):
            supported = torch.cuda.is_bf16_supported()
    if precision == "bf16" and not supported:
        raise ValueError("--precision bf16 需要支持 BF16 的 CUDA GPU；请使用 auto 或 fp32")
    return "bf16" if supported else "fp32"
