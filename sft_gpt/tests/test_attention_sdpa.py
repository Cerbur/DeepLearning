"""SDPA causal attention: independent numerics, gradients, MHA wiring and BF16."""
import math
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from sft_gpt.models import multi_head_attention as mha_module


def reference_causal_attention(q, k, v):
    """Compute each prefix separately: no SDPA and no production attention helper."""
    rows = []
    for index in range(q.shape[-2]):
        scores = q[..., index:index + 1, :] @ k[..., :index + 1, :].transpose(-2, -1)
        weights = torch.softmax(scores / math.sqrt(q.shape[-1]), dim=-1)
        rows.append(weights @ v[..., :index + 1, :])
    return torch.cat(rows, dim=-2)


def reference_mha(module, x):
    """Exercise the actual weights and RoPE, but independently join/reference heads."""
    unbatched = x.ndim == 2
    batched = x.unsqueeze(0) if unbatched else x
    batch, length, width = batched.shape
    q, k, v = F.linear(batched, module.qkv_proj.weight).chunk(3, dim=-1)
    q, k, v = [part.reshape(batch, length, mha_module.NUM_HEADS, module.head_dim)
               .transpose(1, 2) for part in (q, k, v)]
    q = module.rope(q).to(v.dtype)
    k = module.rope(k).to(v.dtype)
    context = reference_causal_attention(q, k, v)
    joined = context.transpose(1, 2).reshape(batch, length, width)
    output = F.linear(joined, module.out_proj.weight)
    return output.squeeze(0) if unbatched else output


class AttentionSDPATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        # Keep test determinism local rather than changing other suites' RNG state.
        self.generator = torch.Generator().manual_seed(2026)

    def make_qkv(self, length, dtype, layout):
        if layout == "unbatched":
            shape = (length, 8)
        elif layout == "head_transpose":
            shape = (2, length, 3, 8)
        else:
            shape = (2, 3, length, 8)
        result = []
        for _ in range(3):
            tensor = torch.randn(shape, dtype=dtype, generator=self.generator)
            if layout == "head_transpose":
                tensor = tensor.transpose(1, 2)
                if length > 1:
                    self.assertFalse(tensor.is_contiguous())
            result.append(tensor.requires_grad_())
        return result

    def check_reference_outputs_and_gradients(self, dtype, rtol, atol):
        for length in (1, 2, 7, 17):
            for layout in ("unbatched", "contiguous", "head_transpose"):
                with self.subTest(dtype=dtype, length=length, layout=layout):
                    qkv = self.make_qkv(length, dtype, layout)
                    reference_qkv = [tensor.detach().clone().requires_grad_() for tensor in qkv]
                    actual = mha_module.attention(*qkv)
                    expected = reference_causal_attention(*reference_qkv)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                    # A nonuniform upstream derivative tests every output coordinate.
                    upstream = torch.randn(actual.shape, dtype=dtype, generator=self.generator)
                    actual_grads = torch.autograd.grad(actual, qkv, upstream)
                    expected_grads = torch.autograd.grad(expected, reference_qkv, upstream)
                    for name, actual_grad, expected_grad in zip("qkv", actual_grads, expected_grads):
                        with self.subTest(gradient=name):
                            torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=atol)

    def test_float64_outputs_and_qkv_gradients_match_reference(self):
        self.check_reference_outputs_and_gradients(torch.float64, rtol=1e-11, atol=1e-12)

    def test_float32_outputs_and_qkv_gradients_match_reference(self):
        self.check_reference_outputs_and_gradients(torch.float32, rtol=2e-5, atol=2e-6)

    def test_future_tokens_cannot_change_prefix_outputs_or_gradients(self):
        for prefix in (1, 4, 6):
            with self.subTest(prefix=prefix):
                original = self.make_qkv(7, torch.float64, "head_transpose")
                changed = []
                for tensor in original:
                    altered = tensor.detach().clone()
                    altered[..., prefix:, :] += 20 * torch.randn(
                        altered[..., prefix:, :].shape, dtype=altered.dtype, generator=self.generator)
                    changed.append(altered.requires_grad_())
                actual = mha_module.attention(*original)
                perturbed = mha_module.attention(*changed)
                torch.testing.assert_close(actual[..., :prefix, :], perturbed[..., :prefix, :],
                                           rtol=0, atol=0)
                upstream = torch.randn(actual[..., :prefix, :].shape,
                                       dtype=actual.dtype, generator=self.generator)
                original_grads = torch.autograd.grad(actual[..., :prefix, :], original, upstream)
                changed_grads = torch.autograd.grad(perturbed[..., :prefix, :], changed, upstream)
                for name, first, second in zip("qkv", original_grads, changed_grads):
                    with self.subTest(gradient=name):
                        torch.testing.assert_close(first, second, rtol=0, atol=0)
                        torch.testing.assert_close(first[..., prefix:, :],
                                                   torch.zeros_like(first[..., prefix:, :]), rtol=0, atol=0)
                self.assertFalse(torch.equal(actual[..., prefix:, :], perturbed[..., prefix:, :]))

    def test_mha_batched_and_unbatched_match_reference_forward_and_backward(self):
        with torch.random.fork_rng():
            torch.manual_seed(91)
            module = mha_module.MultiHeadAttention().double()
        for shape in ((1, mha_module.D_MODEL), (5, mha_module.D_MODEL),
                      (2, 5, mha_module.D_MODEL)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float64, generator=self.generator, requires_grad=True)
                reference_x = x.detach().clone().requires_grad_()
                actual = module(x)
                expected = reference_mha(module, reference_x)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
                upstream = torch.randn(actual.shape, dtype=actual.dtype, generator=self.generator)
                parameters = tuple(module.parameters())
                actual_grads = torch.autograd.grad(actual, (x,) + parameters, upstream)
                expected_grads = torch.autograd.grad(expected, (reference_x,) + parameters, upstream)
                for first, second in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-11)
                if x.ndim == 2:
                    torch.testing.assert_close(module(x.unsqueeze(0))[0], actual, rtol=0, atol=0)

    def test_mha_routes_head_tensors_to_causal_sdpa_without_dropout(self):
        with torch.random.fork_rng():
            torch.manual_seed(92)
            module = mha_module.MultiHeadAttention()
        for training in (False, True):
            for unbatched in (False, True):
                with self.subTest(training=training, unbatched=unbatched):
                    module.train(training)
                    shape = (4, mha_module.D_MODEL) if unbatched else (2, 4, mha_module.D_MODEL)
                    x = torch.randn(shape, generator=self.generator)
                    with mock.patch.object(mha_module.F, "scaled_dot_product_attention",
                                           wraps=F.scaled_dot_product_attention) as sdpa:
                        actual = module(x)
                    sdpa.assert_called_once()
                    q, k, v = sdpa.call_args.args
                    self.assertEqual(sdpa.call_args.kwargs, {"dropout_p": 0.0, "is_causal": True})
                    self.assertEqual(q.shape, (1 if unbatched else 2, mha_module.NUM_HEADS,
                                               4, module.head_dim))
                    self.assertEqual(q.shape, k.shape)
                    self.assertEqual(k.shape, v.shape)
                    self.assertEqual((q.dtype, k.dtype, v.dtype), (x.dtype,) * 3)
                    self.assertFalse(v.is_contiguous())
                    expected = reference_mha(module, x)
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         "CUDA BF16 is not available")
    def test_cuda_bf16_forward_backward_and_mha_qkv_dtype_alignment(self):
        qkv = [tensor.detach().cuda().to(torch.bfloat16).requires_grad_()
               for tensor in self.make_qkv(7, torch.float32, "head_transpose")]
        output = mha_module.attention(*qkv)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all().item())
        output.float().square().mean().backward()
        for tensor in qkv:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())

        with torch.random.fork_rng():
            torch.manual_seed(93)
            module = mha_module.MultiHeadAttention().cuda()
        # FP32 RoPE buffers intentionally promote Q/K before the production cast.
        self.assertEqual(module.rope.cos.dtype, torch.float32)
        x = torch.randn((2, 7, mha_module.D_MODEL), generator=self.generator).cuda().requires_grad_()
        with mock.patch.object(mha_module.F, "scaled_dot_product_attention",
                               wraps=F.scaled_dot_product_attention) as sdpa:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = module(x)
        sdpa.assert_called_once()
        self.assertEqual(tuple(t.dtype for t in sdpa.call_args.args), (torch.bfloat16,) * 3)
        self.assertEqual(sdpa.call_args.kwargs, {"dropout_p": 0.0, "is_causal": True})
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all().item())
        output.float().square().mean().backward()
        for tensor in (x,) + tuple(module.parameters()):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())


if __name__ == "__main__":
    unittest.main()
