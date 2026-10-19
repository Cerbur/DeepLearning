import random
from pathlib import Path

import torch

from ..config import MODEL_CONFIG
from ..models import MiniGPT
from ..tokenization import (CharTokenizer, tokenizer_spec, tokenizer_from_dict)


def save_checkpoint(path, model, tokenizer, optimizer, **metadata):
    """保存预训练状态及自描述 tokenizer；兼容旧 version=1 字符检查点。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON 可独立加载；检查点仍保留 tokens，以兼容旧格式并绑定模型词表。
    tokenizer.save(path.parent / "vocab.json")
    checkpoint = dict(
        format_version=1 if isinstance(tokenizer, CharTokenizer) else 2, model_config=MODEL_CONFIG,
        model_state=model.state_dict(), tokens=tokenizer.tokens,
        optimizer_state=optimizer.state_dict(),
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **metadata,
    )
    # 标记为预训练产物；不改变旧有模型、优化器及训练状态字段。
    checkpoint["training_stage"] = "pre_train"
    checkpoint["tokenizer"] = tokenizer_spec(tokenizer)
    # 临时文件写完再替换，避免中断时破坏上一次检查点。
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到检查点 {path}，请先运行 preTrain 预训练模型")
    # 仅加载自己训练或可信来源的文件；torch.load 使用 pickle。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") not in (1, 2) or checkpoint["model_config"] != MODEL_CONFIG:
        raise ValueError("检查点格式或模型参数不匹配，请使用相同的模型结构")
    if "tokenizer" in checkpoint:
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
        if tokenizer.tokens != checkpoint["tokens"]:
            raise ValueError("检查点 tokenizer 与 tokens 的 ID 顺序不一致")
    elif checkpoint["format_version"] == 1:
        tokenizer = CharTokenizer(checkpoint["tokens"])
    else:
        raise ValueError("通用文本检查点缺少 tokenizer 描述，不能猜测编码方式")
    model = MiniGPT(len(tokenizer.tokens)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint
