"""CPU accumulation regressions with a tiny, dropout-free language model.

The independent reference concatenates each accumulation window into a real
large batch. Fixed permutations preserve sample order across both executions.
"""
import copy
import inspect
import math
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from sft_gpt.pre_training.epoch import run_epoch


class TinyLanguageModel(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.embedding = nn.Embedding(13, 5)
        self.output = nn.Linear(5, 13)
        self.to(dtype=dtype)

    def forward(self, inputs):
        return self.output(self.embedding(inputs))


def reference_epoch(model, sequences, batch_size, optimizer, order,
                    grad_accum_steps=1, label_smoothing=0.0):
    """One large-batch loss/backward per window, without calling run_epoch."""
    total_loss = 0.0
    total_weight = 0
    padded = bool((sequences == -100).any())
    window_size = batch_size * grad_accum_steps
    model.train()
    for start in range(0, len(order), window_size):
        batch = sequences[order[start:start + window_size]]
        labels = batch[:, 1:]
        valid_tokens = int((labels != -100).sum())
        if not valid_tokens:
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch[:, :-1].clamp_min(0))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, label_smoothing=label_smoothing,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        weight = valid_tokens if padded else len(batch)
        total_loss += loss.item() * weight
        total_weight += weight
    if not total_weight:
        raise ValueError("no valid target tokens")
    return total_loss / total_weight


class GradientAccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self, dtype=torch.float64):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(273)
            return TinyLanguageModel(dtype)

    @staticmethod
    def sequences(count=9):
        return (torch.arange(count * 5).reshape(count, 5) * 3 + 1) % 13

    @staticmethod
    def optimizer(model):
        return torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0.01)

    def assert_models_close(self, first, second):
        for (name, left), (other_name, right) in zip(
                first.named_parameters(), second.named_parameters()):
            self.assertEqual(name, other_name)
            with self.subTest(parameter=name):
                if left.dtype == torch.float64:
                    torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-11)
                else:
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

    def assert_optimizer_states_close(self, first, second):
        left = first.state_dict()
        right = second.state_dict()
        self.assertEqual(left["param_groups"], right["param_groups"])
        self.assertEqual(left["state"].keys(), right["state"].keys())
        for index, state in left["state"].items():
            self.assertEqual(state.keys(), right["state"][index].keys())
            for key, value in state.items():
                expected = right["state"][index][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, expected, rtol=1e-10, atol=1e-11)
                else:
                    self.assertEqual(value, expected)

    def test_new_options_are_keyword_only(self):
        parameters = inspect.signature(run_epoch).parameters
        for name in ("grad_accum_steps", "max_updates", "max_tokens",
                     "progress", "on_update"):
            with self.subTest(parameter=name):
                self.assertEqual(parameters[name].kind,
                                 inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["grad_accum_steps"].default, 1)

    def test_accumulation_matches_large_batches_in_fp64_and_fp32(self):
        sequences = self.sequences(8)
        order = torch.tensor([7, 1, 5, 3, 0, 6, 2, 4])
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                large = self.make_model(dtype)
                micro = copy.deepcopy(large)
                expected = reference_epoch(
                    large, sequences, 4, self.optimizer(large), order,
                    label_smoothing=0.1,
                )
                with mock.patch("torch.randperm", return_value=order.clone()):
                    actual = run_epoch(
                        micro, sequences, 2, "cpu", self.optimizer(micro),
                        label_smoothing=0.1, precision="fp32", grad_accum_steps=2,
                    )
                self.assertIsInstance(actual, float)
                self.assertAlmostEqual(actual, expected,
                                       places=10 if dtype == torch.float64 else 5)
                self.assert_models_close(micro, large)

    def test_unequal_tail_microbatch_and_partial_window_are_flushed(self):
        # Micros [2, 2, 1]: both a smaller final window and a single window
        # whose last microbatch has fewer rows than the preceding ones.
        sequences = self.sequences(5)
        order = torch.tensor([4, 0, 3, 1, 2])
        for accumulation in (2, 3, 8):
            with self.subTest(grad_accum_steps=accumulation):
                expected_model = self.make_model()
                actual_model = copy.deepcopy(expected_model)
                expected = reference_epoch(
                    expected_model, sequences, 2, self.optimizer(expected_model),
                    order, grad_accum_steps=accumulation,
                )
                optimizer = self.optimizer(actual_model)
                scheduler = mock.Mock()
                with mock.patch("torch.randperm", return_value=order.clone()), \
                        mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    actual = run_epoch(
                        actual_model, sequences, 2, "cpu", optimizer,
                        scheduler=scheduler, precision="fp32",
                        grad_accum_steps=accumulation,
                    )
                self.assertEqual(step.call_count, math.ceil(3 / accumulation))
                self.assertEqual(scheduler.step.call_count, step.call_count)
                self.assertAlmostEqual(actual, expected, places=10)
                self.assert_models_close(actual_model, expected_model)

    def test_ignored_targets_use_token_weight_not_microbatch_means(self):
        sequences = torch.tensor([
            [1, 2, 3, 4, 5],
            [2, 6, -100, -100, -100],
            [4, 8, 7, -100, -100],
            [5, 9, 10, 11, -100],
            [6, 12, -100, -100, -100],
        ])
        order = torch.tensor([1, 0, 4, 2, 3])
        expected_model = self.make_model()
        actual_model = copy.deepcopy(expected_model)
        expected = reference_epoch(
            expected_model, sequences, 1, self.optimizer(expected_model), order,
            grad_accum_steps=3, label_smoothing=0.15,
        )
        progress = {}
        with mock.patch("torch.randperm", return_value=order.clone()):
            actual = run_epoch(
                actual_model, sequences, 1, "cpu", self.optimizer(actual_model),
                label_smoothing=0.15, precision="fp32", grad_accum_steps=3,
                progress=progress,
            )
        self.assertAlmostEqual(actual, expected, places=10)
        self.assert_models_close(actual_model, expected_model)
        self.assertEqual(progress["tokens"], 11)
        self.assertEqual(progress["total_weight"], 11)
        self.assertAlmostEqual(progress["total_loss"], expected * 11, places=10)

    def test_empty_microbatches_and_empty_windows_do_not_update(self):
        # Windows: [empty, empty], [valid, empty], [valid, empty], [empty].
        sequences = torch.tensor([
            [1, -100, -100, -100],
            [2, -100, -100, -100],
            [3, 4, 5, -100],
            [4, -100, -100, -100],
            [5, 6, -100, -100],
            [6, -100, -100, -100],
            [7, -100, -100, -100],
        ])
        order = torch.arange(len(sequences))
        expected_model = self.make_model()
        actual_model = copy.deepcopy(expected_model)
        expected = reference_epoch(
            expected_model, sequences, 1, self.optimizer(expected_model), order,
            grad_accum_steps=2,
        )
        optimizer = self.optimizer(actual_model)
        scheduler = mock.Mock()
        progress = {}
        cursors = []
        with mock.patch("torch.randperm", return_value=order), \
                mock.patch.object(actual_model, "forward", wraps=actual_model.forward) as forward, \
                mock.patch.object(optimizer, "step", wraps=optimizer.step) as step, \
                mock.patch("torch.nn.utils.clip_grad_norm_",
                           wraps=nn.utils.clip_grad_norm_) as clip:
            actual = run_epoch(
                actual_model, sequences, 1, "cpu", optimizer,
                scheduler=scheduler, precision="fp32", grad_accum_steps=2,
                progress=progress, on_update=lambda state: cursors.append(state["next_batch"]),
            )
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(step.call_count, 2)
        self.assertEqual(clip.call_count, 2)
        self.assertEqual(scheduler.step.call_count, 2)
        self.assertEqual(cursors, [4, 6])
        self.assertEqual(progress["next_batch"], 7)
        self.assertEqual(progress["updates"], 2)
        self.assertEqual(progress["tokens"], 3)
        self.assertEqual(progress["total_weight"], 3)
        self.assertAlmostEqual(actual, expected, places=10)
        self.assert_models_close(actual_model, expected_model)

    def test_all_empty_epoch_preserves_old_error_without_updates(self):
        sequences = torch.tensor([[1, -100, -100]] * 5)
        model = self.make_model()
        original = copy.deepcopy(model)
        optimizer = self.optimizer(model)
        scheduler = mock.Mock()
        callback = mock.Mock()
        with mock.patch("torch.randperm", return_value=torch.arange(5)), \
                mock.patch.object(model, "forward", wraps=model.forward) as forward, \
                mock.patch.object(optimizer, "step", wraps=optimizer.step) as step, \
                mock.patch("torch.nn.utils.clip_grad_norm_",
                           wraps=nn.utils.clip_grad_norm_) as clip:
            with self.assertRaises(ValueError):
                run_epoch(model, sequences, 1, "cpu", optimizer,
                          scheduler=scheduler, precision="fp32", grad_accum_steps=2,
                          progress={}, on_update=callback)
        forward.assert_not_called()
        step.assert_not_called()
        clip.assert_not_called()
        scheduler.step.assert_not_called()
        callback.assert_not_called()
        self.assert_models_close(model, original)

    def test_clip_optimizer_and_scheduler_run_once_per_nonempty_window(self):
        sequences = self.sequences(9)
        model = self.make_model()
        optimizer = self.optimizer(model)
        scheduler = mock.Mock()
        events = []
        original_clip = nn.utils.clip_grad_norm_
        original_step = optimizer.step

        def clip(*args, **kwargs):
            events.append("clip")
            return original_clip(*args, **kwargs)

        def step(*args, **kwargs):
            events.append("optimizer")
            return original_step(*args, **kwargs)

        scheduler.step.side_effect = lambda: events.append("scheduler")
        with mock.patch("torch.randperm", return_value=torch.arange(9)), \
                mock.patch("torch.nn.utils.clip_grad_norm_", side_effect=clip), \
                mock.patch.object(optimizer, "step", side_effect=step):
            run_epoch(model, sequences, 2, "cpu", optimizer,
                      scheduler=scheduler, precision="fp32", grad_accum_steps=2,
                      on_update=lambda state: events.append("callback"))
        self.assertEqual(events, ["clip", "optimizer", "scheduler", "callback"] * 3)

    def test_accumulation_one_matches_legacy_loop_and_default_exactly(self):
        for padded in (False, True):
            with self.subTest(padded=padded):
                sequences = self.sequences(5)
                if padded:
                    sequences[1, 2:] = -100
                    sequences[4, 3:] = -100
                order = torch.tensor([4, 1, 0, 3, 2])
                reference = self.make_model()
                default = copy.deepcopy(reference)
                explicit = copy.deepcopy(reference)
                expected = reference_epoch(
                    reference, sequences, 2, self.optimizer(reference), order,
                    label_smoothing=0.1,
                )
                losses = []
                for model, options in ((default, {}), (explicit, {"grad_accum_steps": 1})):
                    with mock.patch("torch.randperm", return_value=order.clone()):
                        losses.append(run_epoch(
                            model, sequences, 2, "cpu", self.optimizer(model),
                            label_smoothing=0.1, precision="fp32", **options,
                        ))
                self.assertEqual(losses[0], losses[1])
                self.assertAlmostEqual(losses[0], expected, places=12)
                self.assert_models_close(default, reference)
                for left, right in zip(default.parameters(), explicit.parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_nonfinite_gradient_stops_before_optimizer_and_scheduler(self):
        model = self.make_model()
        optimizer = self.optimizer(model)
        scheduler = mock.Mock()
        parameter = next(model.parameters())
        handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        progress = {}
        try:
            with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    run_epoch(model, self.sequences(2), 1, "cpu", optimizer,
                              scheduler=scheduler, grad_accum_steps=2, progress=progress)
            step.assert_not_called()
            scheduler.step.assert_not_called()
            self.assertEqual(progress["updates"], 0)
        finally:
            handle.remove()

    def test_invalid_accumulation_is_rejected_before_forward_or_update(self):
        for invalid in (0, -1, 1.5, True, "2", None):
            with self.subTest(grad_accum_steps=invalid):
                model = self.make_model()
                optimizer = self.optimizer(model)
                with mock.patch.object(model, "forward", wraps=model.forward) as forward, \
                        mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    with self.assertRaises((ValueError, TypeError)):
                        run_epoch(model, self.sequences(2), 1, "cpu", optimizer,
                                  precision="fp32", grad_accum_steps=invalid)
                forward.assert_not_called()
                step.assert_not_called()

    def test_progress_callback_observes_completed_window_state(self):
        sequences = self.sequences(9)
        order = torch.tensor([8, 0, 6, 2, 7, 1, 5, 3, 4])
        model = self.make_model()
        optimizer = self.optimizer(model)
        scheduler = mock.Mock()
        progress = {}
        snapshots = []

        def on_update(state):
            self.assertIs(state, progress)
            self.assertEqual(state["updates"], scheduler.step.call_count)
            self.assertIsInstance(state["order"], torch.Tensor)
            self.assertEqual(state["order"].device.type, "cpu")
            self.assertEqual(state["order"].dtype, torch.int64)
            self.assertTrue(math.isfinite(state["total_loss"]))
            snapshots.append(dict(state))

        with mock.patch("torch.randperm", return_value=order.clone()) as permutation:
            loss = run_epoch(
                model, sequences, 2, "cpu", optimizer,
                scheduler=scheduler, precision="fp32", grad_accum_steps=2,
                progress=progress, on_update=on_update,
            )
        self.assertEqual(permutation.call_count, 1)
        torch.testing.assert_close(progress["order"], order, rtol=0, atol=0)
        self.assertEqual([state["next_batch"] for state in snapshots], [2, 4, 5])
        self.assertEqual([state["updates"] for state in snapshots], [1, 2, 3])
        self.assertEqual([state["tokens"] for state in snapshots], [16, 32, 36])
        self.assertEqual(progress["total_weight"], 9)
        self.assertAlmostEqual(loss, progress["total_loss"] / 9, places=12)

    def test_max_updates_resume_reuses_order_and_matches_uninterrupted_epoch(self):
        sequences = self.sequences(9)
        order = torch.tensor([8, 0, 6, 2, 7, 1, 5, 3, 4])
        uninterrupted = self.make_model()
        resumed = copy.deepcopy(uninterrupted)
        full_optimizer = self.optimizer(uninterrupted)
        resumed_optimizer = self.optimizer(resumed)
        full_scheduler = torch.optim.lr_scheduler.StepLR(full_optimizer, 1, gamma=0.8)
        resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, 1, gamma=0.8)
        with mock.patch("torch.randperm", return_value=order.clone()):
            expected = run_epoch(
                uninterrupted, sequences, 2, "cpu", full_optimizer,
                scheduler=full_scheduler, precision="fp32", grad_accum_steps=2,
            )
        progress = {}
        callbacks = []
        for call_index, expected_cursor in enumerate((2, 4, 5), 1):
            if call_index == 1:
                permutation = mock.patch("torch.randperm", return_value=order.clone())
            else:
                permutation = mock.patch("torch.randperm", side_effect=AssertionError("resume reshuffled"))
            with permutation:
                actual = run_epoch(
                    resumed, sequences, 2, "cpu", resumed_optimizer,
                    scheduler=resumed_scheduler, precision="fp32", grad_accum_steps=2,
                    max_updates=1, progress=progress,
                    on_update=lambda state: callbacks.append(state["next_batch"]),
                )
            self.assertEqual(progress["next_batch"], expected_cursor)
            self.assertEqual(progress["updates"], call_index)
            self.assertAlmostEqual(actual, progress["total_loss"] / progress["total_weight"], places=12)
            if call_index == 1:
                saved_order = progress["order"]
            else:
                self.assertIs(progress["order"], saved_order)
        self.assertEqual(callbacks, [2, 4, 5])
        self.assertAlmostEqual(actual, expected, places=12)
        self.assert_models_close(resumed, uninterrupted)
        self.assert_optimizer_states_close(resumed_optimizer, full_optimizer)
        self.assertEqual(resumed_scheduler.state_dict(), full_scheduler.state_dict())
        # An exhausted progress record must not repeat its final window.
        with mock.patch("torch.randperm", side_effect=AssertionError("completed epoch reshuffled")), \
                mock.patch.object(resumed_optimizer, "step", wraps=resumed_optimizer.step) as step:
            repeated = run_epoch(
                resumed, sequences, 2, "cpu", resumed_optimizer,
                scheduler=resumed_scheduler, precision="fp32", grad_accum_steps=2,
                max_updates=1, progress=progress,
            )
        step.assert_not_called()
        self.assertEqual(repeated, actual)

    def test_max_tokens_is_per_call_soft_valid_target_budget(self):
        sequences = torch.tensor([
            [1, 2, -100, -100, -100],
            [2, 3, -100, -100, -100],
            [3, 4, -100, -100, -100],
            [4, 5, -100, -100, -100],
            [5, 6, 7, 8, 9],
            [6, 7, 8, 9, 10],
        ])
        order = torch.arange(6)
        expected_model = self.make_model()
        resumed = copy.deepcopy(expected_model)
        expected_optimizer = self.optimizer(expected_model)
        resumed_optimizer = self.optimizer(resumed)
        expected = reference_epoch(
            expected_model, sequences, 1, expected_optimizer, order,
            grad_accum_steps=2,
        )
        progress = {}
        callback_cursors = []
        with mock.patch("torch.randperm", return_value=order.clone()):
            first = run_epoch(
                resumed, sequences, 1, "cpu", resumed_optimizer, precision="fp32",
                grad_accum_steps=2, max_tokens=3, progress=progress,
                on_update=lambda state: callback_cursors.append(state["next_batch"]),
            )
        # Need two windows of two valid tokens each to reach budget 3.
        self.assertEqual(progress["next_batch"], 4)
        self.assertEqual(progress["tokens"], 4)
        self.assertEqual(progress["updates"], 2)
        self.assertEqual(callback_cursors, [2, 4])
        self.assertAlmostEqual(first, progress["total_loss"] / 4, places=12)
        with mock.patch("torch.randperm", side_effect=AssertionError("resume reshuffled")):
            actual = run_epoch(
                resumed, sequences, 1, "cpu", resumed_optimizer, precision="fp32",
                grad_accum_steps=2, max_tokens=1, progress=progress,
                on_update=lambda state: callback_cursors.append(state["next_batch"]),
            )
        # Budget is soft: finish all eight valid tokens in the final window.
        self.assertEqual(progress["next_batch"], 6)
        self.assertEqual(progress["tokens"], 12)
        self.assertEqual(progress["updates"], 3)
        self.assertEqual(callback_cursors, [2, 4, 6])
        self.assertAlmostEqual(actual, expected, places=10)
        self.assert_models_close(resumed, expected_model)
        self.assert_optimizer_states_close(resumed_optimizer, expected_optimizer)


if __name__ == "__main__":
    unittest.main()
