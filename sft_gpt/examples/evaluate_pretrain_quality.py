"""Reproducible read-only checkpoint evaluation using the production sampler."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sft_gpt.inference import generate
from sft_gpt.pre_training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pre_train/best.pt")
    parser.add_argument("--output", default="evaluation/pretrain_quality_results.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    prompts = [
        ("greeting", "你好"),
        ("everyday", "今天的天气很好，我决定"),
        ("story", "小猫躲在门后，听见脚步声越来越近，"),
        ("news", "据报道，近年来新能源汽车的发展"),
        ("explanation", "植物进行光合作用时，需要"),
        ("science", "水在标准大气压下的沸点是"),
        ("causality", "因为昨天晚上下了一场大雨，所以今天早上"),
        ("list", "节约用水可以从以下几个方面做起：\n第一，"),
        ("consistency", "小明把红色的球放进盒子，把蓝色的球放在桌上。他打开盒子，看到了"),
        ("fact", "中国的首都是"),
        ("arithmetic_probe", "1+1="),
        ("english_probe", "The sun rises in the"),
        ("code_probe", "def add(a, b):\n    return"),
        ("chat_probe", "user: 你好，请介绍一下你自己。\nassistant: "),
    ]
    validation = checkpoint.get("valid_texts", checkpoint.get("valid_poems", []))
    indices = random.Random(20261020).sample(range(len(validation)), min(4, len(validation)))
    for index in indices:
        prompts.append((f"heldout_prefix_{index}", validation[index][:60]))
    tests = [(category, prompt, seed, 64) for category, prompt in prompts for seed in (42, 123)]
    tests.extend([
        ("long_story", "周末，小明和朋友一起去公园。他们刚走到湖边，就发现", 42, 192),
        ("long_explanation", "良好的睡眠对身体健康非常重要。首先，", 42, 192),
        ("long_news", "近年来，人工智能技术在各个领域得到广泛应用。", 42, 192),
    ])
    result = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": checkpoint["epoch"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_settings": checkpoint["train_settings"],
        "tokenizer": type(tokenizer).__name__,
        "vocab_size": len(tokenizer.tokens),
        "history_tail": checkpoint.get("history", [])[-12:],
        "sampling": {"device": "cpu", "precision": "fp32", "temperature": 0.8, "top_k": 30, "structured": False, "seeds": [42, 123]},
        "samples": [],
        "notes": ["Production generate() is used unchanged; sampled IDs are observed to record stop reason.", "These are informal qualitative probes, not a standardized benchmark.", "English/code/arithmetic/chat are additional probes, not mandatory general-Chinese-pretraining pass criteria.", "heldout_prefix uses validation documents saved in the checkpoint; exact train/eval document deduplication does not rule out near duplicates."],
    }
    actual_multinomial = torch.multinomial
    for category, prompt, seed, budget in tests:
        torch.manual_seed(seed)
        sampled = []
        def traced(*a, **kw):
            value = actual_multinomial(*a, **kw)
            sampled.append(value.item())
            return value
        start = time.monotonic()
        with patch.object(torch, "multinomial", side_effect=traced):
            text = generate(model, tokenizer, prompt, budget, temperature=0.8, top_k=30, structured=False)
        continuation = text[len(prompt):]
        ids = sampled[:-1] if sampled and sampled[-1] == tokenizer.eos else sampled
        grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
        counts = Counter(grams)
        row = {"category": category, "prompt": prompt, "seed": seed, "budget": budget, "continuation": continuation,
               "stop_reason": "eos" if sampled and sampled[-1] == tokenizer.eos else "length",
               "new_tokens": len(ids), "sampling_steps": len(sampled),
               "repeated_4gram_fraction": 1-len(counts)/len(grams) if grams else 0,
               "max_4gram_occurrences": max(counts.values(), default=0),
               "seconds": time.monotonic()-start}
        result["samples"].append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    stops = Counter(row["stop_reason"] for row in result["samples"])
    result["summary"] = {"sample_count": len(tests), "stop_reasons": dict(stops),
                         "total_generated_tokens": sum(row["new_tokens"] for row in result["samples"]),
                         "samples_with_4gram_repeated_at_least_4_times": sum(row["max_4gram_occurrences"]>=4 for row in result["samples"])}
    history = checkpoint.get("history", [])
    if history:
        result["summary"]["checkpoint_valid_perplexity"] = math.exp(history[-1]["valid_loss"])
        result["summary"]["checkpoint_train_eval_perplexity"] = math.exp(history[-1]["train_eval_loss"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("SUMMARY", json.dumps(result["summary"], ensure_ascii=True), flush=True)
    print("OUTPUT", output.resolve(), flush=True)


if __name__ == "__main__":
    main()
