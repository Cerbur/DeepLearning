"""AMP boundaries, not an unfused reimplementation of SDPA internals.

CUDA tests exercise the real PyTorch 2.0.1 autocast/SDPA/AdamW path. CPU
low-precision tensors are used only for standalone normalization/RoPE tests;
run_epoch on CPU must remain FP32 even inside outer BF16 autocast.
"""
import math
import unittest
from contextlib import contextmanager
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from sft_gpt.models import (
    GPTBlock, GPTDecoder, MiniGPT, MultiHeadAttention, RMSNorm,
    RotaryPositionalEmbedding, SwiGLU, TokenEmbedding,
)
from sft_gpt.models import multi_head_attention as mha_module
from sft_gpt.models.precision import promote_for_stability
from sft_gpt.pre_training.epoch import run_epoch


CUDA_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@contextmanager
def record_boundaries(model):
    """Observe module boundaries without replacing their computation."""
    records = {}
    handles = []
    observed = (nn.Linear, nn.Embedding, nn.Dropout, nn.SiLU, MiniGPT, TokenEmbedding,
                GPTBlock, GPTDecoder, RMSNorm, RotaryPositionalEmbedding,
                SwiGLU, MultiHeadAttention)

    def hook_for(name):
        def hook(module, args, output):
            records.setdefault(name, []).append((args[0].dtype, output.dtype))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, observed):
            handles.append(module.register_forward_hook(hook_for(name)))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


class PrecisionBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self, device):
        # Initialisation is deterministic without changing another suite's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(541)
            return MiniGPT(32).to(device)

    def assert_parameter_and_adam_dtypes(self, model, optimizer):
        self.assertTrue(optimizer.state, "an actual optimizer update is required")
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(parameter.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                state = optimizer.state[parameter]
                for key in ("exp_avg", "exp_avg_sq"):
                    self.assertIn(key, state)
                for key, value in state.items():
                    if torch.is_tensor(value) and value.is_floating_point():
                        self.assertEqual(value.dtype, torch.float32, key)
                        self.assertTrue(torch.isfinite(value).all().item(), key)

    def assert_model_boundaries(self, model, records, mixed):
        low = torch.bfloat16 if mixed else torch.float32
        self.assertTrue(records)
        modules = dict(model.named_modules())
        for name, module in modules.items():
            if name not in records:
                continue
            if isinstance(module, MiniGPT):
                expected = (torch.int64, low)
            elif isinstance(module, (nn.Embedding, TokenEmbedding)):
                expected = (torch.int64, torch.float32)
            elif isinstance(module, nn.Linear):
                # Only attention context arriving at out_proj is BF16.
                # FFN SiLU/gating are FP32 before the autocast down projection.
                input_dtype = low if name.endswith("attention.out_proj") else torch.float32
                expected = (input_dtype, low)
            elif isinstance(module, RotaryPositionalEmbedding):
                expected = (low, torch.float32)
                self.assertEqual(module.cos.dtype, torch.float32)
                self.assertEqual(module.sin.dtype, torch.float32)
            elif isinstance(module, (SwiGLU, MultiHeadAttention)):
                expected = (torch.float32, low)
            else:
                # Embedding dropout, RMSNorm, SiLU, residual dropout, block
                # outputs and the final decoder norm stay on the FP32 path.
                expected = (torch.float32, torch.float32)
            with self.subTest(module=name):
                count = 2 if isinstance(module, RotaryPositionalEmbedding) else 1
                if isinstance(module, nn.Dropout) and name.endswith(".dropout"):
                    # GPTBlock shares one dropout module across both residuals.
                    if isinstance(modules.get(name.rsplit(".", 1)[0]), GPTBlock):
                        count = 2
                self.assertEqual(records[name], [expected] * count)
        # Prevent a vacuous pass if hooks stop reaching whole model stages.
        for name, module in modules.items():
            if isinstance(module, (nn.Linear, nn.Embedding, RMSNorm, GPTBlock,
                                   SwiGLU, RotaryPositionalEmbedding, nn.SiLU)):
                self.assertIn(name, records)

    def check_epoch_boundaries(self, device, precision, mixed, outer_autocast=False):
        model = self.make_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        sequences = torch.tensor([[0, 5, 6, 7, 1], [0, 8, 1, -100, -100]])
        before = model.output_proj.weight.detach().clone()
        sdpa_records, ce_records, backward_states = [], [], []
        original_sdpa, original_ce = F.scaled_dot_product_attention, F.cross_entropy

        def sdpa(q, k, v, *args, **kwargs):
            # Call real fused-or-math SDPA; do not prescribe internal softmax
            # tensors or which hardware-specific kernel PyTorch selects.
            output = original_sdpa(q, k, v, *args, **kwargs)
            sdpa_records.append((tuple(t.dtype for t in (q, k, v)), output.dtype, kwargs))
            return output

        def cross_entropy(logits, labels, *args, **kwargs):
            output = original_ce(logits, labels, *args, **kwargs)
            ce_records.append((logits.dtype, output.dtype, torch.is_autocast_enabled()))
            with torch.autocast(device_type=device, enabled=False):
                reference = original_ce(logits.float(), labels, *args, **kwargs)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            return output

        def backward_hook(grad):
            backward_states.append((torch.is_autocast_enabled(),
                                    torch.is_autocast_cpu_enabled()))

        handle = model.output_proj.weight.register_hook(backward_hook)
        try:
            with record_boundaries(model) as records, \
                    mock.patch.object(mha_module.F, "scaled_dot_product_attention", side_effect=sdpa), \
                    mock.patch.object(F, "cross_entropy", side_effect=cross_entropy), \
                    torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=outer_autocast), \
                    torch.random.fork_rng(devices=[]):
                torch.manual_seed(118)
                loss = run_epoch(model, sequences, 2, device, optimizer,
                                 label_smoothing=0.1, precision=precision)
        finally:
            handle.remove()
        self.assertTrue(math.isfinite(loss))
        self.assertFalse(torch.equal(before, model.output_proj.weight))
        self.assertIs(model.output_proj.weight, model.embedding.embedding.weight)
        self.assertEqual(ce_records, [(torch.float32, torch.float32, False)])
        self.assertEqual(backward_states, [(False, False)])
        self.assert_model_boundaries(model, records, mixed)
        low = torch.bfloat16 if mixed else torch.float32
        self.assertEqual(len(sdpa_records), len(model.decoder.layers))
        for qkv_dtypes, output_dtype, kwargs in sdpa_records:
            self.assertEqual(qkv_dtypes, (low,) * 3)
            self.assertEqual(output_dtype, low)
            self.assertEqual(kwargs, {"dropout_p": 0.0, "is_causal": True})
        self.assert_parameter_and_adam_dtypes(model, optimizer)

    def test_stability_promotion_changes_only_low_precision_floating_tensors(self):
        for dtype in (torch.int64, torch.bool, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                original = torch.tensor([0, 1], dtype=dtype)
                self.assertIs(promote_for_stability(original), original)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                original = torch.tensor([0.5, 1.5], dtype=dtype, requires_grad=True)
                promoted = promote_for_stability(original)
                self.assertEqual(promoted.dtype, torch.float32)
                torch.testing.assert_close(promoted, original.float(), rtol=0, atol=0)
                promoted.sum().backward()
                torch.testing.assert_close(original.grad, torch.ones_like(original))

    def test_cpu_epoch_auto_and_fp32_keep_all_boundaries_fp32(self):
        for precision in ("auto", "fp32"):
            with self.subTest(precision=precision):
                self.check_epoch_boundaries("cpu", precision, mixed=False, outer_autocast=True)

    @unittest.skipUnless(CUDA_BF16, "CUDA BF16 is not available")
    def test_cuda_bf16_epoch_module_loss_gradient_and_adam_boundaries(self):
        self.check_epoch_boundaries("cuda", "bf16", mixed=True, outer_autocast=True)

    @unittest.skipUnless(CUDA_BF16, "CUDA BF16 is not available")
    def test_cuda_explicit_fp32_overrides_outer_bf16_for_every_stage(self):
        self.check_epoch_boundaries("cuda", "fp32", mixed=False, outer_autocast=True)

    def check_low_precision_rms_norm(self, device, dtype, autocast=False):
        # 1000**2 overflows FP16. The other row detects BF16 rounding in
        # pow/mean/divide even if a final FP32 weight promotes only the output.
        values = [[1000., -1500., 2000., -500., 750., -1250., 250., 1750.],
                  [0.992, 1.008, 1.125, 1.3125, -2.625, 0.001, 3.75, -0.45]]
        x = torch.tensor(values, device=device, dtype=dtype, requires_grad=True)
        norm = RMSNorm(8).to(device)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.7, 1.4, 8, device=device))
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
            actual = norm(x)
        reference_x = x.detach().float()
        reference = norm.weight * (reference_x / torch.sqrt(
            reference_x.square().mean(-1, keepdim=True) + norm.eps))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
        actual.square().mean().backward()
        for tensor in (x, norm.weight):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
        self.assertEqual(norm.weight.grad.dtype, torch.float32)

    def test_low_precision_rms_norm_uses_fp32_statistics_and_output_on_cpu(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check_low_precision_rms_norm("cpu", dtype)

    @unittest.skipUnless(CUDA_BF16, "CUDA BF16 is not available")
    def test_cuda_autocast_low_precision_rms_norm_statistics_and_output(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check_low_precision_rms_norm("cuda", dtype, autocast=True)

    def check_low_precision_rope(self, device, dtype, autocast=False):
        length, width, base = 513, 8, 10000.0
        # Positions past 256 detect accidental BF16 position-table generation.
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
            rope = RotaryPositionalEmbedding(width, length, base=base).to(device)
            x = torch.linspace(-2, 2, length * width, device=device).reshape(1, 1, length, width)
            x = x.to(dtype).requires_grad_()
            actual = rope(x)
        positions = torch.arange(length, device=device, dtype=torch.float32)
        frequencies = base ** (-torch.arange(0, width, 2, device=device, dtype=torch.float32) / width)
        angles = positions[:, None] * frequencies[None, :]
        cos, sin = angles.cos(), angles.sin()
        self.assertEqual(rope.cos.dtype, torch.float32)
        self.assertEqual(rope.sin.dtype, torch.float32)
        # Production tables may be built on CPU then moved to CUDA; CPU/CUDA
        # pow/trig round differently by a few FP32 ULPs, unlike BF16 tables.
        torch.testing.assert_close(rope.cos, cos, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(rope.sin, sin, rtol=2e-5, atol=1e-6)
        even, odd = x.float()[..., ::2], x.float()[..., 1::2]
        reference = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
        actual.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_low_precision_rope_tables_computation_and_output_are_fp32_on_cpu(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check_low_precision_rope("cpu", dtype)

    @unittest.skipUnless(CUDA_BF16, "CUDA BF16 is not available")
    def test_cuda_autocast_rope_tables_computation_and_output_are_fp32(self):
        self.check_low_precision_rope("cuda", torch.bfloat16, autocast=True)

    @unittest.skipUnless(CUDA_BF16, "CUDA BF16 is not available")
    def test_cuda_block_promotes_low_precision_residual_stream_to_fp32(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(83)
            block = GPTBlock().cuda().eval()
        width = block.norm1.weight.numel()
        x = torch.linspace(-1, 1, 3 * width, device="cuda").reshape(1, 3, width)
        x = x.to(torch.bfloat16).requires_grad_()
        with record_boundaries(block) as records, torch.autocast("cuda", dtype=torch.bfloat16):
            actual = block(x)
        self.assertEqual(actual.dtype, torch.float32)
        for name in ("norm1", "norm2"):
            self.assertEqual(records[name], [(torch.float32, torch.float32)])
        self.assertEqual(records["dropout"], [(torch.float32, torch.float32)] * 2)
        self.assertTrue(torch.isfinite(actual).all().item())
        actual.square().mean().backward()
        for tensor in (x,) + tuple(block.parameters()):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())

    def test_cpu_fp32_epoch_preserves_float64_loss_contrasts(self):
        class FixedDoubleLogits(nn.Module):
            def forward(self, tokens):
                # FP32 rounds these distinct logits to the same number.
                return torch.tensor([[[1e8, 1e8 + 1.]]], dtype=torch.float64,
                                    device=tokens.device)

        sequences = torch.tensor([[0, 1]])
        original_ce = F.cross_entropy
        with mock.patch.object(F, "cross_entropy", wraps=original_ce) as ce:
            loss = run_epoch(FixedDoubleLogits(), sequences, 1, "cpu", precision="fp32")
        self.assertEqual(ce.call_args.args[0].dtype, torch.float64)
        self.assertAlmostEqual(loss, math.log1p(math.exp(-1.)), places=13)
        self.assertGreater(abs(loss - math.log(2.)), 0.3)

    def test_cpu_double_model_epoch_matches_direct_float64_cross_entropy(self):
        model = self.make_model("cpu").double().eval()
        sequences = torch.tensor([[0, 5, 6, 1]])
        original_ce = F.cross_entropy
        with torch.no_grad():
            logits = model(sequences[:, :-1])
            expected = original_ce(logits.reshape(-1, 32), sequences[:, 1:].reshape(-1))
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(expected.dtype, torch.float64)
        with mock.patch.object(F, "cross_entropy", wraps=original_ce) as ce, \
                torch.autocast("cpu", dtype=torch.bfloat16):
            actual = run_epoch(model, sequences, 1, "cpu", precision="fp32")
        self.assertEqual(ce.call_args.args[0].dtype, torch.float64)
        self.assertTrue(math.isclose(actual, expected.item(), rel_tol=1e-14, abs_tol=1e-14))

    def test_float64_rms_norm_preserves_math_and_input_parameter_gradients(self):
        norm = RMSNorm(8).double()
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.7, 1.4, 8, dtype=torch.float64))
        x = (torch.linspace(-2, 3, 16, dtype=torch.float64).reshape(2, 8)
             + 1.23456789e-8).requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        reference_weight = norm.weight.detach().clone().requires_grad_()
        actual = norm(x)
        reference = reference_weight * (reference_x / torch.sqrt(
            reference_x.square().mean(-1, keepdim=True) + norm.eps))
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-13)
        upstream = torch.linspace(-0.8, 1.3, 16, dtype=torch.float64).reshape_as(actual)
        actual_grads = torch.autograd.grad(actual, (x, norm.weight), upstream)
        reference_grads = torch.autograd.grad(reference, (reference_x, reference_weight), upstream)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            self.assertEqual(actual_grad.dtype, torch.float64)
            torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-11, atol=1e-12)

    def test_float64_rope_preserves_math_and_input_gradients(self):
        rope = RotaryPositionalEmbedding(8, 7).double()
        x = (torch.linspace(-1, 2, 56, dtype=torch.float64).reshape(1, 1, 7, 8)
             + 1.23456789e-8).requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        cos, sin = rope.cos[:7], rope.sin[:7]
        even, odd = reference_x[..., ::2], reference_x[..., 1::2]
        reference = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        actual = rope(x)
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-13)
        upstream = torch.linspace(-0.8, 1.3, 56, dtype=torch.float64).reshape_as(actual)
        actual_grad, = torch.autograd.grad(actual, x, upstream)
        reference_grad, = torch.autograd.grad(reference, reference_x, upstream)
        self.assertEqual(actual_grad.dtype, torch.float64)
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
