"""从父目录运行：python -m sft_gpt.examples.benchmark_precision。
只使用合成 token，不加载/覆盖正式语料、词表或检查点。
"""
import argparse
import gc
import json
import statistics
import time

import torch

from sft_gpt.config import MAX_SEQ_LEN
from sft_gpt.models import MiniGPT
from sft_gpt.pre_training.epoch import run_epoch
from sft_gpt.pre_training.precision import resolve_precision


def main():
    parser = argparse.ArgumentParser(description="相同模型/合成数据上的 FP32 与 BF16 训练对照")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bf16-first", action="store_true", help="交换测量顺序，复核温度/功率漂移")
    args = parser.parse_args()
    if (min(args.batch_size, args.steps, args.warmup_steps, args.repeats) < 1
            or args.vocab_size < 4 or not 1 <= args.seq_len <= MAX_SEQ_LEN):
        parser.error("批次/步数/重复数须为正，vocab-size >=4，seq-len 不得超过模型上限")
    resolve_precision("cuda", "bf16")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    initial = MiniGPT(args.vocab_size).state_dict()
    data = torch.randint(args.vocab_size, (args.batch_size * args.steps, args.seq_len + 1))
    warmup = data[:args.batch_size].repeat(args.warmup_steps, 1)
    results = {}
    for precision in (("bf16", "fp32") if args.bf16_first else ("fp32", "bf16")):
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(123)
        model = MiniGPT(args.vocab_size).cuda()
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        run_epoch(model, warmup, args.batch_size, "cuda", optimizer, 0.1, precision=precision)
        times, peaks = [], []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss = run_epoch(model, data, args.batch_size, "cuda", optimizer, 0.1, precision=precision)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            peaks.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
        seconds = statistics.median(times)
        results[precision] = dict(seconds=seconds, ms_per_step=seconds * 1000 / args.steps,
                                  tokens_per_second=args.batch_size * args.steps * args.seq_len / seconds,
                                  peak_allocated_mib=max(peaks), last_objective=loss)
        del model, optimizer
    print(json.dumps(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                          matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                          float32_matmul_precision=torch.get_float32_matmul_precision(),
                          config=vars(args), results=results,
                          speedup=results["fp32"]["seconds"] / results["bf16"]["seconds"],
                          peak_memory_reduction=1 - results["bf16"]["peak_allocated_mib"]
                          / results["fp32"]["peak_allocated_mib"]), indent=2))


if __name__ == "__main__":
    main()
