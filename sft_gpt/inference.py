import math

import torch

from .config import MAX_SEQ_LEN
from .pre_training.checkpoint import load_checkpoint
from .tokenization import CharTokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompt="", max_new_tokens=25, temperature=0.8,
             top_k=30, structured=False):
    """默认保留文本原样续写；structured=True 仅用于旧字符五绝模型。"""
    if not math.isfinite(temperature) or temperature <= 0 or top_k < 0 or max_new_tokens < 1:
        raise ValueError("temperature 必须为正数，top_k >= 0，max_new_tokens >= 1")
    if structured:
        prompt = "".join(prompt.split())  # 旧五绝模式保留原有去空白顺序
    if isinstance(tokenizer, CharTokenizer):
        unknown = sorted(set(prompt) - set(tokenizer.token_ids))
        if unknown:
            raise ValueError(f"输入含旧字符词表外字符: {''.join(unknown)}；换用通用 tokenizer 需重新预训练")
    punctuation = {5: "，", 11: "。", 17: "，", 23: "。"}
    if structured:
        if not isinstance(tokenizer, CharTokenizer):
            raise ValueError("五绝格式约束只支持旧字符 tokenizer；通用文本请不传 --poetry")
        if not set("，。").issubset(tokenizer.token_ids):
            raise ValueError("结构化五绝生成要求词表包含中文标点 ， 和 。")
        if len(prompt) > 24:
            raise ValueError("五绝开头不能超过 24 个字符（含标点）")
        for position, char in enumerate(prompt):
            if (position in punctuation and char != punctuation[position]) or (
                position not in punctuation and char in "，。"
            ):
                raise ValueError("开头格式不符，应为每句五字，标点依次为 ，。，。")
    ids = [tokenizer.bos] + tokenizer.encode(prompt, special_tokens=False)
    device = next(model.parameters()).device
    model.eval()
    for _ in range(max_new_tokens):
        position = len(ids) - 1
        if structured and position >= 24:
            break
        if structured and position in punctuation:
            next_id = tokenizer.token_ids[punctuation[position]]
        else:
            inputs = torch.tensor([ids[-MAX_SEQ_LEN:]], dtype=torch.long, device=device)
            logits = model(inputs)[0, -1].clone() / temperature
            logits[[tokenizer.bos, tokenizer.unk]] = float("-inf")
            if hasattr(tokenizer, "valid_next_ids"):
                allowed = tokenizer.valid_next_ids(ids)
                remaining = max_new_tokens - _ - 1
                if remaining < 3:
                    # 不在预算末尾开始无法写完的 UTF-8 多字节字符。
                    allowed = [token_id for token_id in allowed
                               if token_id >= 260 or token_id == tokenizer.eos
                               or tokenizer.pending_byte_count(ids + [token_id]) <= remaining]
                valid = torch.zeros_like(logits, dtype=torch.bool)
                valid[allowed] = True
                logits = logits.masked_fill(~valid, float("-inf"))
            if structured:
                logits[[tokenizer.eos, tokenizer.token_ids["，"], tokenizer.token_ids["。"]]] = float("-inf")
            if top_k:
                values, indices = torch.topk(logits, min(top_k, logits.numel()))
                filtered = torch.full_like(logits, float("-inf"))
                filtered[indices] = values
                logits = filtered
            if not torch.isfinite(logits).any():
                raise ValueError("词表中没有可生成的字符，请降低 --min-freq 后重新训练")
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        if next_id == tokenizer.eos:
            break
        ids.append(next_id)
    return tokenizer.decode(ids)


def infer(args, device):
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    print(f"已加载第 {checkpoint['epoch']} 轮参数，验证损失 {checkpoint['history'][-1]['valid_loss']:.4f}")

    def show(prompt):
        for _ in range(args.num_samples):
            text = generate(model, tokenizer, prompt, args.max_new_tokens,
                            args.temperature, args.top_k, args.poetry)
            print(text.replace("，", "，\n").replace("。", "。\n") if args.poetry else text)

    if args.interactive:
        print("输入文本前缀（回车为随机续写，/quit 退出；这是预训练模型续写，不是多轮聊天 SFT）")
        while True:
            try:
                prompt = input("文本> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip() == "/quit":
                break
            try:
                show(prompt)
            except ValueError as error:
                print(f"输入错误: {error}")
    else:
        show(args.prompt)
