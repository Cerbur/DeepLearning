"""混合精度策略、CPU 等价性、CUDA 数值及检查点恢复回归。"""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from sft_gpt.cli import build_parser, main
from sft_gpt.models import MiniGPT
from sft_gpt.pre_training.checkpoint import load_checkpoint, save_checkpoint
from sft_gpt.pre_training.epoch import run_epoch
from sft_gpt.pre_training.trainer import pre_train
from sft_gpt.tokenization import CharTokenizer
from sft_gpt.pre_training.precision import resolve_precision


class PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cpu_auto_and_fp32_do_not_probe_cuda(self):
        with mock.patch("torch.cuda.is_available") as probe:
            self.assertEqual(resolve_precision("cpu"), "fp32")
            self.assertEqual(resolve_precision("cuda", "fp32"), "fp32")
            probe.assert_not_called()
        with self.assertRaisesRegex(ValueError, "BF16"):
            resolve_precision("cpu", "bf16")
        with self.assertRaisesRegex(ValueError, "precision"):
            resolve_precision("cpu", "fp16")

    def test_cuda_auto_detection_and_explicit_bf16_rejection(self):
        for available, supported in ((False, False), (True, False), (True, True)):
            with self.subTest(available=available, supported=supported), \
                    mock.patch("torch.cuda.is_available", return_value=available), \
                    mock.patch("torch.cuda.device") as context, \
                    mock.patch("torch.cuda.is_bf16_supported", return_value=supported):
                expected = "bf16" if available and supported else "fp32"
                self.assertEqual(resolve_precision("cuda:1"), expected)
                if available:
                    context.assert_called_with(torch.device("cuda:1"))
                else:
                    context.assert_not_called()
                if expected == "bf16":
                    self.assertEqual(resolve_precision("cuda:1", "bf16"), expected)
                else:
                    with self.assertRaisesRegex(ValueError, "BF16"):
                        resolve_precision("cuda:1", "bf16")

    def test_cli_precision_choices(self):
        parser = build_parser()
        self.assertIsNone(parser.parse_args(["preTrain"]).precision)
        for precision in ("auto", "bf16", "fp32"):
            self.assertEqual(parser.parse_args(["preTrain", "--precision", precision]).precision, precision)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["preTrain", "--precision", "fp16"])

    def test_cpu_auto_matches_fp32_exactly(self):
        torch.manual_seed(123)
        first = MiniGPT(32)
        second = copy.deepcopy(first)
        sequences = torch.tensor([[0, 5, 6, 7, 1], [0, 8, 1, -100, -100]])
        results = []
        for model, precision in ((first, "auto"), (second, "fp32")):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            torch.manual_seed(456)
            with mock.patch("torch.autocast", wraps=torch.autocast) as autocast:
                results.append(run_epoch(model, sequences, 1, "cpu", optimizer, 0.1, precision=precision))
                autocast.assert_called_once_with(device_type="cpu", enabled=False)
        self.assertEqual(*results)
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_cpu_fp32_and_auto_override_outer_autocast(self):
        model = MiniGPT(32)
        sequences = torch.tensor([[0, 5, 6, 7, 1]])
        dtypes = []
        handle = model.register_forward_hook(lambda module, args, output: dtypes.append(output.dtype))
        try:
            for precision in ("fp32", "auto"):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    run_epoch(model, sequences, 1, "cpu", precision=precision)
            self.assertEqual(dtypes, [torch.float32, torch.float32])
        finally:
            handle.remove()

    def test_invalid_bf16_request_fails_before_loading_data(self):
        args = build_parser().parse_args(["preTrain", "--precision", "bf16"])
        with mock.patch("sft_gpt.pre_training.trainer.load_train_eval_texts") as load, \
                self.assertRaisesRegex(ValueError, "BF16"):
            pre_train(args, torch.device("cpu"))
        load.assert_not_called()

    def test_checkpoint_policy_reuse_override_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "texts.txt"
            data.write_text("hello world\n\nhi there", encoding="utf-8")
            output = root / "out"
            base = ["preTrain", "--device", "cpu", "--output-dir", str(output)]
            with mock.patch("sft_gpt.pre_training.trainer.DEFAULT_TEXT_VOCAB", root / "vocab.json"), \
                    mock.patch("builtins.print"):
                main(base + ["--data", str(data), "--seq-len", "8", "--epochs", "1", "--precision", "fp32"])
                path = output / "last.pt"
                state = torch.load(path, weights_only=False)
                self.assertEqual(state["train_settings"]["precision"], "fp32")
                main(base + ["--resume", str(path), "--epochs", "2"])
                state = torch.load(path, weights_only=False)
                self.assertEqual(state["train_settings"]["precision"], "fp32")
                main(base + ["--resume", str(path), "--epochs", "3", "--precision", "auto"])
                state = torch.load(path, weights_only=False)
                self.assertEqual(state["train_settings"]["precision"], "auto")
                state["train_settings"].pop("precision")
                torch.save(state, path)
                main(base + ["--resume", str(path), "--epochs", "4"])
                self.assertEqual(torch.load(path, weights_only=False)["train_settings"]["precision"], "auto")

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
    def test_cuda_bf16_train_eval_fp32_state_and_scheduler(self):
        torch.manual_seed(123)
        model = MiniGPT(32).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = mock.Mock()
        sequences = torch.tensor([[0, 5, 6, 7, 1], [0, 8, 1, -100, -100]])
        forwards, backwards, losses = [], [], []
        original_ce = torch.nn.functional.cross_entropy

        def ce(*args, **kwargs):
            result = original_ce(*args, **kwargs)
            losses.append(result.dtype)
            return result

        handle = model.register_forward_hook(lambda module, args, output: forwards.append(output.dtype))
        grad_handle = model.output_proj.weight.register_hook(
            lambda grad: backwards.append(torch.is_autocast_enabled()))
        before = model.output_proj.weight.detach().clone()
        try:
            with mock.patch("torch.nn.functional.cross_entropy", side_effect=ce), \
                    torch.autocast("cuda", dtype=torch.bfloat16):
                loss = run_epoch(model, sequences, 1, "cuda", optimizer, 0.1, scheduler, precision="auto")
                valid = run_epoch(model, sequences, 1, "cuda", precision="bf16")
            self.assertTrue(math.isfinite(loss) and math.isfinite(valid))
            self.assertEqual(forwards, [torch.bfloat16] * 4)
            self.assertEqual(losses, [torch.float32] * 4)
            self.assertEqual(backwards, [False, False])
            self.assertEqual(scheduler.step.call_count, 2)
            self.assertFalse(torch.equal(before, model.output_proj.weight))
            for parameter in model.parameters():
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertEqual(parameter.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            for state in optimizer.state.values():
                self.assertEqual(state["exp_avg"].dtype, torch.float32)
                self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                batch = sequences.cuda()
                expected = original_ce(model(batch[:, :-1].clamp_min(0)).reshape(-1, 32),
                                       batch[:, 1:].reshape(-1)).item()
            self.assertTrue(math.isclose(valid, expected, rel_tol=5e-3, abs_tol=5e-3))
            forwards.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                run_epoch(model, sequences, 2, "cuda", precision="fp32")
            self.assertEqual(forwards, [torch.float32])
            # AMP 产物仍可按原格式在 CPU FP32 加载并继续训练，无 scaler 状态。
            tokenizer = CharTokenizer(["<BOS>", "<EOS>", "<UNK>"] + list("abcdefghijklmnopqrstuvwxyz012"))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "amp.pt"
                save_checkpoint(path, model, tokenizer, optimizer, train_settings={"precision": "bf16"})
                restored, _, state = load_checkpoint(path, torch.device("cpu"))
                self.assertNotIn("scaler_state", state)
                self.assertEqual(state["train_settings"]["precision"], "bf16")
                self.assertIs(restored.output_proj.weight, restored.embedding.embedding.weight)
                cpu_optimizer = torch.optim.AdamW(restored.parameters())
                cpu_optimizer.load_state_dict(state["optimizer_state"])
                self.assertTrue(math.isfinite(run_epoch(restored, sequences, 2, "cpu",
                                                        cpu_optimizer, precision="fp32")))
        finally:
            handle.remove()
            grad_handle.remove()

    def test_empty_labels_and_nonfinite_loss_remain_guarded(self):
        model = MiniGPT(32)
        with self.assertRaisesRegex(ValueError, "有效 token"):
            run_epoch(model, torch.tensor([[0, -100, -100]]), 1, "cpu")
        with mock.patch.object(model, "forward", return_value=torch.full((1, 2, 32), float("nan"))), \
                self.assertRaisesRegex(RuntimeError, "非有限损失"):
            run_epoch(model, torch.tensor([[0, 1, 2]]), 1, "cpu")
