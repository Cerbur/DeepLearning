"""Small-model optimizer/checkpoint regressions; never instantiate the production size."""
import copy
import importlib
import inspect
import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

import torch
from torch import nn

from sft_gpt.models import MiniGPT
from sft_gpt.models import gpt_decoder as decoder_module
from sft_gpt.pre_training import optimizer as optimizer_module
from sft_gpt.pre_training.optimizer import build_optimizer


CUDA_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@contextmanager
def small_model_dimensions():
    # Patch imported dimension constants only while this isolated test runs.
    # Production config, initialization code and the configured dropout stay intact.
    overrides = {
        "mini_gpt": {"D_MODEL": 16, "NUM_LAYERS": 2},
        "token_embedding": {"D_MODEL": 16},
        "gpt_decoder": {"D_MODEL": 16},
        "gpt_block": {"D_MODEL": 16},
        "swiglu": {"D_MODEL": 16, "D_FF": 32},
        "multi_head_attention": {"D_MODEL": 16, "NUM_HEADS": 2, "MAX_SEQ_LEN": 8},
    }
    with ExitStack() as stack:
        for name, values in overrides.items():
            module = importlib.import_module("sft_gpt.models." + name)
            stack.enter_context(mock.patch.multiple(module, **values))
        yield


class SmallModelOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(small_model_dimensions())
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        self.stack.enter_context(torch.random.fork_rng(devices=devices))
        torch.manual_seed(721)

    def test_decay_groups_and_shared_parameters_are_unique(self):
        model = MiniGPT(19)
        self.assertIs(model.embedding.embedding.weight, model.output_proj.weight)
        optimizer = build_optimizer(model, 0.002, 0.1)
        self.assertEqual(len(optimizer.param_groups), 2)
        actual = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(actual), len({id(p) for p in actual}))
        self.assertEqual({id(p) for p in actual}, {id(p) for p in model.parameters()})
        for group, matrices in zip(optimizer.param_groups, (True, False)):
            self.assertEqual(group["weight_decay"], 0.1 if matrices else 0.0)
            self.assertTrue(all((p.ndim >= 2) == matrices for p in group["params"]))
            self.assertEqual(group["betas"], (0.9, 0.95))
        self.assertEqual(sum(p is model.output_proj.weight for p in actual), 1)

    def test_defensive_deduplication_and_scalar_no_decay(self):
        scalar = nn.Parameter(torch.ones(()))
        vector = nn.Parameter(torch.ones(3))
        matrix = nn.Parameter(torch.ones(3, 3))
        model = mock.Mock()
        model.parameters.side_effect = lambda: iter([scalar, matrix, vector, matrix, scalar])
        grouped = build_optimizer(model, 0.01, 0.2)
        self.assertEqual([id(p) for p in grouped.param_groups[0]["params"]], [id(matrix)])
        self.assertEqual([id(p) for p in grouped.param_groups[1]["params"]],
                         [id(scalar), id(vector)])
        legacy = build_optimizer(model, 0.01, 0.2, decay_groups=False)
        self.assertEqual([id(p) for p in legacy.param_groups[0]["params"]],
                         [id(scalar), id(matrix), id(vector)])

    def test_legacy_group_order_betas_and_state_resume(self):
        model = MiniGPT(19)
        next(model.parameters()).requires_grad_(False)
        expected = list(model.parameters())
        legacy = build_optimizer(model, 0.002, 0.1, decay_groups=False,
                                 betas=(0.9, 0.999), fused=False)
        self.assertEqual(len(legacy.param_groups), 1)
        self.assertEqual([id(p) for p in legacy.param_groups[0]["params"]],
                         [id(p) for p in expected])
        self.assertEqual(legacy.param_groups[0]["betas"], (0.9, 0.999))
        original = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.1)
        self.assertEqual(legacy.state_dict()["param_groups"],
                         original.state_dict()["param_groups"])
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 0.125)
        original.step()
        legacy.load_state_dict(copy.deepcopy(original.state_dict()))
        for parameter, state in original.state.items():
            for key, value in state.items():
                torch.testing.assert_close(legacy.state[parameter][key], value, rtol=0, atol=0)
        # Defaults and explicit non-legacy beta choices are never auto-adjusted.
        custom = build_optimizer(model, 0.002, 0.1, betas=(0.8, 0.88))
        self.assertTrue(all(g["betas"] == (0.8, 0.88) for g in custom.param_groups))

    def test_fused_cpu_and_misleading_device_raise(self):
        model = MiniGPT(19)
        for device in (None, "cpu", torch.device("cpu"), "cuda"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "CUDA"):
                build_optimizer(model, 0.002, 0.1, fused=True, device=device)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_fused_cuda_update_and_betas(self):
        model = MiniGPT(19).cuda()
        optimizer = build_optimizer(model, 0.002, 0.1, fused=True, device="cuda",
                                    betas=(0.8, 0.88))
        self.assertTrue(all(g["fused"] for g in optimizer.param_groups))
        self.assertTrue(all(g["betas"] == (0.8, 0.88) for g in optimizer.param_groups))
        before = model.output_proj.weight.detach().clone()
        model(torch.tensor([[0, 1, 2, 3]], device="cuda")).square().mean().backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(model.output_proj.weight).all().item())
        self.assertFalse(torch.equal(before, model.output_proj.weight))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_unsupported_fused_api_raises_instead_of_fallback(self):
        model = MiniGPT(19).cuda()
        signature = inspect.Signature([inspect.Parameter("params", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        with mock.patch.object(optimizer_module.inspect, "signature", return_value=signature):
            with self.assertRaisesRegex(RuntimeError, "does not support fused AdamW"):
                build_optimizer(model, 0.002, 0.1, fused=True)

    def test_checkpoint_runtime_flag_does_not_change_state_or_initialization(self):
        torch.manual_seed(12)
        ordinary = MiniGPT(19)
        torch.manual_seed(12)
        checkpointed = MiniGPT(19)
        self.assertFalse(ordinary.decoder.gradient_checkpointing)
        checkpointed.decoder.gradient_checkpointing = True
        self.assertEqual(list(ordinary.state_dict()), list(checkpointed.state_dict()))
        for name, value in ordinary.state_dict().items():
            torch.testing.assert_close(checkpointed.state_dict()[name], value, rtol=0, atol=0)
        ordinary.load_state_dict(checkpointed.state_dict(), strict=True)

    def compare_checkpoint_gradients(self, device, mixed=False):
        ordinary = MiniGPT(19).to(device).train()
        checkpointed = copy.deepcopy(ordinary)
        checkpointed.decoder.gradient_checkpointing = True
        tokens = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], device=device)
        reference_checkpoint = decoder_module.checkpoint

        def run(model, cache):
            torch.manual_seed(117)
            with torch.autocast(device_type=device, enabled=mixed, dtype=torch.bfloat16,
                                cache_enabled=cache):
                output = model(tokens)
                self.assertEqual(torch.is_autocast_cache_enabled(), cache)
            output.float().square().mean().backward()
            return output.detach(), torch.rand(6, device=device)

        expected, expected_rng = run(ordinary, False)
        with mock.patch.object(decoder_module, "checkpoint", wraps=reference_checkpoint) as checkpoint:
            actual, actual_rng = run(checkpointed, True)
        self.assertEqual(checkpoint.call_count, 2)
        for call in checkpoint.call_args_list:
            self.assertFalse(call.kwargs["use_reentrant"])
            self.assertTrue(call.kwargs["preserve_rng_state"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)
        for (name, parameter), (actual_name, actual_parameter) in zip(
                ordinary.named_parameters(), checkpointed.named_parameters()):
            with self.subTest(parameter=name):
                self.assertEqual(name, actual_name)
                self.assertIsNotNone(actual_parameter.grad)
                self.assertEqual(actual_parameter.dtype, torch.float32)
                self.assertEqual(actual_parameter.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual_parameter.grad).all().item())
                torch.testing.assert_close(actual_parameter.grad, parameter.grad, rtol=0, atol=0)

    def test_checkpoint_fp32_outputs_gradients_and_dropout_rng(self):
        self.compare_checkpoint_gradients("cpu")

    def test_checkpoint_cpu_bf16_autocast_cache_and_dropout_rng(self):
        self.compare_checkpoint_gradients("cpu", mixed=True)

    def test_checkpoint_non_reentrant_supports_frozen_embedding_inputs(self):
        model = MiniGPT(19).train()
        model.embedding.embedding.weight.requires_grad_(False)
        model.decoder.gradient_checkpointing = True
        model(torch.tensor([[0, 1, 2, 3]])).square().mean().backward()
        for name, parameter in model.decoder.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_checkpoint_cuda_fp32_outputs_gradients_and_dropout_rng(self):
        self.compare_checkpoint_gradients("cuda")

    @unittest.skipUnless(CUDA_BF16, "CUDA BF16 is not available")
    def test_checkpoint_cuda_bf16_autocast_cache_and_dropout_rng(self):
        self.compare_checkpoint_gradients("cuda", mixed=True)

    @unittest.skipUnless(CUDA_BF16, "CUDA BF16 is not available")
    def test_checkpoint_bf16_preserves_fp32_stability_boundaries_on_recompute(self):
        model = MiniGPT(19).cuda().train()
        model.decoder.gradient_checkpointing = True
        records = {}
        handles = []
        for name, module in model.decoder.named_modules():
            if isinstance(module, (nn.Linear, nn.Dropout, nn.SiLU)) or name.endswith(("norm1", "norm2", "rope")):
                def record(module, args, output, name=name):
                    records.setdefault(name, []).append((args[0].dtype, output.dtype,
                                                         torch.is_autocast_cache_enabled()))
                handles.append(module.register_forward_hook(record))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
                output = model(torch.tensor([[0, 1, 2, 3]], device="cuda"))
            output.float().square().mean().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(records)
        for name, boundaries in records.items():
            with self.subTest(module=name):
                self.assertGreaterEqual(len(boundaries), 2, "checkpoint must recompute each block")
                for input_dtype, output_dtype, cache in boundaries:
                    self.assertFalse(cache)
                    if name.endswith(("norm1", "norm2", "dropout", "activation")):
                        self.assertEqual((input_dtype, output_dtype), (torch.float32, torch.float32))
                    elif name.endswith("rope"):
                        self.assertEqual((input_dtype, output_dtype), (torch.bfloat16, torch.float32))
                    else:
                        self.assertEqual(output_dtype, torch.bfloat16)

    def test_checkpoint_bypassed_for_eval_no_grad_and_inference(self):
        model = MiniGPT(19)
        model.decoder.gradient_checkpointing = True
        tokens = torch.tensor([[0, 1, 2, 3]])
        with mock.patch.object(decoder_module, "checkpoint", side_effect=AssertionError("checkpoint invoked")):
            model.eval()(tokens).sum().backward()
            with torch.no_grad():
                model.train()(tokens)
            with torch.inference_mode():
                model.train()(tokens)
            model.decoder.gradient_checkpointing = False
            model.train()(tokens).sum().backward()


if __name__ == "__main__":
    unittest.main()
