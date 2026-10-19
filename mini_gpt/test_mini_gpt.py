"""运行: python -m unittest discover -s mini_gpt -p "test_*.py" -v"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import mini_gpt as gpt


class MiniGPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.poems = [
            "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
            "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        ]
        cls.tokenizer = gpt.CharTokenizer.build(cls.poems, min_freq=1)

    def test_dataset(self):
        poems = gpt.load_poems(gpt.DEFAULT_DATA)
        self.assertGreater(len(poems), 16000)
        self.assertEqual(len(poems), len(set(poems)))
        self.assertTrue(all(len(poem) == 24 for poem in poems))

    def test_tsv_validation_and_dedup(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "data.tsv"
            path.write_text("text\n" + "\n".join(self.poems + [self.poems[0], "非五绝"]), encoding="utf-8")
            self.assertEqual(gpt.load_poems(path), self.poems)
            path.write_text("title\n无正文\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                gpt.load_poems(path)

    def test_select_train_eval_indices_is_local_and_repeatable(self):
        python_state = gpt.random.getstate()
        torch_state = torch.get_rng_state().clone()
        expected = torch.randperm(37, generator=torch.Generator(device="cpu").manual_seed(123))[:9].tolist()
        for _ in range(2):
            indices = gpt.select_train_eval_indices(37, 9, 123)
            self.assertIsInstance(indices, list)
            self.assertTrue(all(type(index) is int for index in indices))
            self.assertEqual(indices, expected)
            self.assertEqual(len(set(indices)), 9)
            self.assertTrue(all(0 <= index < 37 for index in indices))
        self.assertEqual(gpt.random.getstate(), python_state)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_select_train_eval_indices_full_and_saved(self):
        python_state = gpt.random.getstate()
        torch_state = torch.get_rng_state().clone()
        for train_count, sample_size in ((1, 1024), (7, 1024), (7, 7), (7, 0), (1024, 1024)):
            with self.subTest(train_count=train_count, sample_size=sample_size):
                self.assertEqual(gpt.select_train_eval_indices(train_count, sample_size, 99),
                                 list(range(train_count)))
        for sample_size, saved in ((3, [5, 0, 2]), (0, [5, 0, 2, 1, 4, 3]),
                                   (1024, [5, 0, 2, 1, 4, 3])):
            original = list(saved)
            with self.subTest(sample_size=sample_size), \
                    mock.patch.object(torch, "randperm", side_effect=AssertionError("saved indices must not resample")):
                indices = gpt.select_train_eval_indices(6, sample_size, 999, saved)
                self.assertEqual(indices, original)
                self.assertIsNot(indices, saved)
                indices[0] = -1
                self.assertEqual(saved, original)
        self.assertEqual(gpt.random.getstate(), python_state)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_select_train_eval_indices_rejects_invalid_saved(self):
        invalid = ([], (), (0, 1, 2), "012", {0, 1, 2}, {"0": 0},
                   [0, 1], [0, 1, 2, 3], [0, 0, 1], [0, True, 2],
                   [False, 1, 2], [0, 1.0, 2], [0, "1", 2], [0, None, 2],
                   [-1, 1, 2], [0, 1, 6])
        python_state = gpt.random.getstate()
        torch_state = torch.get_rng_state().clone()
        for saved in invalid:
            with self.subTest(saved=saved), self.assertRaises((TypeError, ValueError)):
                gpt.select_train_eval_indices(6, 3, 123, saved)
        for sample_size, saved in ((0, [0, 1, 2]), (1024, [0, 1, 2]),
                                   (0, [0, 1, 2, 3, 4, 4])):
            with self.subTest(sample_size=sample_size, saved=saved), self.assertRaises((TypeError, ValueError)):
                gpt.select_train_eval_indices(6, sample_size, 123, saved)
        self.assertEqual(gpt.random.getstate(), python_state)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_train_eval_size_cli_default_and_validation(self):
        self.assertEqual(gpt.build_parser().parse_args(["train"]).train_eval_size, 1024)
        self.assertEqual(gpt.build_parser().parse_args(["train", "--train-eval-size", "0"]).train_eval_size, 0)
        with mock.patch.object(gpt, "train") as train, mock.patch("sys.stderr"), \
                self.assertRaises(SystemExit) as error:
            gpt.main(["train", "--train-eval-size", "-1"])
        self.assertEqual(error.exception.code, 2)
        train.assert_not_called()

    def test_tokenizer(self):
        ids = self.tokenizer.encode(self.poems[0])
        self.assertEqual((ids[0], ids[-1], len(ids)), (0, 1, 26))
        self.assertEqual(self.tokenizer.decode(ids), self.poems[0])
        self.assertEqual(self.tokenizer.encode("🙂", False), [2])

    def test_model_and_training(self):
        model = gpt.MiniGPT(len(self.tokenizer.tokens))
        self.assertEqual(len(model.decoder.layers), gpt.NUM_LAYERS)
        self.assertIs(model.output_proj.weight, model.embedding.embedding.weight)
        self.assertIn("decoder.layers.0.norm1.weight", model.state_dict())
        sequences = gpt.make_sequences(self.poems, self.tokenizer)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.embedding.embedding.weight.detach().clone()
        loss = gpt.run_epoch(model, sequences, 2, torch.device("cpu"), optimizer, 0.1)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.embedding.embedding.weight))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "model.pt"
            gpt.save_checkpoint(path, model, self.tokenizer, optimizer, epoch=1)
            loaded, tokenizer, metadata = gpt.load_checkpoint(path, torch.device("cpu"))
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(sequences[:, :-1]), loaded(sequences[:, :-1]), rtol=0, atol=0)
            self.assertEqual(tokenizer.tokens, self.tokenizer.tokens)
            self.assertEqual(metadata["epoch"], 1)

    def test_warmup_cosine_curve_and_parameter_groups(self):
        parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
        base_lrs = [0.02, 0.005]
        optimizer = torch.optim.SGD([
            {"params": [parameter], "lr": lr}
            for parameter, lr in zip(parameters, base_lrs)
        ])
        scheduler = gpt.WarmupCosineScheduler(optimizer, total_steps=6, warmup_ratio=1 / 3)
        # Constructor prepares batch zero; step() prepares the next batch.
        # The warmup peak and cosine origin are consecutive peak batches.
        factors = [0.5, 1.0, 1.0, 0.775, 0.325, 0.1, 0.1, 0.1]
        for step, factor in enumerate(factors):
            with self.subTest(step=step):
                self.assertEqual(scheduler.completed_steps, step)
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    self.assertAlmostEqual(group["lr"], base_lr * factor, places=14)
            optimizer.step()
            scheduler.step()

    def test_warmup_cosine_no_warmup_and_small_schedules(self):
        cases = [
            (1, 0, [1.0]),
            (2, 0, [1.0, 0.2]),
            (3, 0, [1.0, 0.6, 0.2]),
            (4, 1, [1.0, 1.0, 0.6, 0.2]),
        ]
        for total_steps, warmup_steps, factors in cases:
            with self.subTest(total_steps=total_steps, warmup_steps=warmup_steps):
                parameter = torch.nn.Parameter(torch.ones(1))
                optimizer = torch.optim.SGD([parameter], lr=0.01)
                scheduler = gpt.WarmupCosineScheduler(
                    optimizer, total_steps, warmup_ratio=warmup_steps / total_steps,
                    min_lr_ratio=0.2)
                for factor in factors + [0.2, 0.2, 0.2]:
                    self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01 * factor, places=14)
                    optimizer.step()
                    scheduler.step()

    def test_warmup_cosine_ratio_rounding_and_cap(self):
        cases = [
            (1, 0.9, 0), (2, 0.9, 0), (3, 0.9, 1),
            (6, 0, 0), (6, 1 / 3, 2), (6, 0.334, 3), (6, 0.99, 4),
            (20, 0.05, 1), (21, 0.05, 2), (100, 0.0001, 1),
        ]
        for total_steps, ratio, expected in cases:
            with self.subTest(total_steps=total_steps, ratio=ratio):
                optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.02)
                scheduler = gpt.WarmupCosineScheduler(optimizer, total_steps, ratio)
                self.assertEqual(scheduler.warmup_steps, expected)
                self.assertEqual(scheduler.completed_steps, 0)
                self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.02 / max(1, expected))
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.02)
        scheduler = gpt.WarmupCosineScheduler(optimizer, 21)
        self.assertEqual(scheduler.warmup_steps, 2)
        self.assertEqual(scheduler.min_lr_ratio, 0.1)

    def test_warmup_cosine_load_legacy_schema(self):
        # Old checkpoints allow a one-batch cosine phase, unlike new ratio construction.
        for total_steps, warmup_steps, factors in (
                (2, 1, [1.0, 1.0, 0.2, 0.2]),
                (4, 3, [1 / 3, 2 / 3, 1.0, 1.0, 0.2, 0.2])):
            for completed in range(len(factors)):
                with self.subTest(total_steps=total_steps, completed=completed):
                    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.9)
                    scheduler = gpt.WarmupCosineScheduler(optimizer, 17, 0, 0.8)
                    state = dict(total_steps=total_steps, warmup_steps=warmup_steps,
                                 min_lr_ratio=0.2, completed_steps=completed, base_lrs=[0.03])
                    scheduler.load_state_dict(state)
                    self.assertEqual(scheduler.state_dict(), state)
                    for factor in factors[completed:]:
                        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.03 * factor)
                        scheduler.step()

    def test_warmup_cosine_invalid_constructor_is_atomic(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.02)
        scheduler = gpt.WarmupCosineScheduler(optimizer, 6, 1 / 3)
        scheduler.step()
        state, lrs = scheduler.state_dict(), scheduler.get_last_lr()
        invalid = [dict(total_steps=value) for value in (0, -1, 2.5, True, None, "6")]
        invalid += [dict(warmup_ratio=value) for value in
                    (-0.1, 1, float("nan"), float("inf"), -float("inf"), None, "0.1")]
        invalid += [dict(min_lr_ratio=value) for value in
                    (-0.1, 1.1, float("nan"), float("inf"), None, "0.1")]
        # State fields are no longer accepted as constructor arguments.
        invalid += [dict(warmup_steps=2), dict(completed_steps=1), dict(base_lrs=[0.02])]
        for changes in invalid:
            with self.subTest(changes=changes):
                kwargs = dict(total_steps=6, warmup_ratio=1 / 3, min_lr_ratio=0.1)
                kwargs.update(changes)
                with self.assertRaises((TypeError, ValueError)):
                    gpt.WarmupCosineScheduler(optimizer, **kwargs)
                self.assertEqual(scheduler.state_dict(), state)
                self.assertEqual(scheduler.get_last_lr(), lrs)
        # An invalid later parameter group must not partially update earlier groups.
        second = torch.nn.Parameter(torch.ones(1))
        optimizer.add_param_group({"params": [second], "lr": 0.03})
        for invalid_lr in (-0.1, float("nan"), float("inf")):
            with self.subTest(base_lr=invalid_lr):
                optimizer.param_groups[1]["lr"] = invalid_lr
                before = optimizer.param_groups[0]["lr"]
                with self.assertRaises((TypeError, ValueError)):
                    gpt.WarmupCosineScheduler(optimizer, 8, 0.5)
                self.assertEqual(optimizer.param_groups[0]["lr"], before)
                self.assertIs(optimizer.param_groups[1]["lr"], invalid_lr)

    def test_warmup_cosine_invalid_load_is_atomic(self):
        optimizer = torch.optim.SGD([
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": lr}
            for lr in (0.02, 0.005)])
        scheduler = gpt.WarmupCosineScheduler(optimizer, 6, 1 / 3)
        scheduler.step()
        state, lrs = scheduler.state_dict(), scheduler.get_last_lr()
        invalid = [None, [], {}, "invalid"]
        for key in state:
            missing = copy.deepcopy(state)
            del missing[key]
            invalid.append(missing)
        values = dict(
            total_steps=(0, -1, True, 6.0, "6", None),
            warmup_steps=(-1, 6, True, 2.0, "2", None),
            completed_steps=(-1, True, 1.0, "1", None),
            min_lr_ratio=(-0.1, 1.1, float("nan"), float("inf"), "0.1", None),
            base_lrs=([], [0.02], [0.02, 0.005, 0.01], None,
                      [0.02, -0.1], [0.02, float("nan")],
                      [0.02, float("inf")], [0.02, "0.005"]))
        for key, candidates in values.items():
            for value in candidates:
                invalid.append(dict(copy.deepcopy(state), **{key: value}))
        for bad_state in invalid:
            with self.subTest(state=bad_state):
                with self.assertRaises((TypeError, ValueError)):
                    scheduler.load_state_dict(bad_state)
                self.assertEqual(scheduler.state_dict(), state)
                self.assertEqual(scheduler.get_last_lr(), lrs)

    def test_warmup_cosine_queries_are_read_only_copies(self):
        optimizer = torch.optim.SGD([
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": lr}
            for lr in (0.02, 0.005)])
        scheduler = gpt.WarmupCosineScheduler(optimizer, 6, 1 / 3, 0.25)
        initial_state = scheduler.state_dict()
        initial_lrs = scheduler.get_last_lr()
        self.assertEqual(scheduler.get_info(), dict(
            total_steps=6, warmup_steps=2, completed_steps=0, phase="warmup",
            peak_lrs=[0.02, 0.005], min_lrs=[0.005, 0.00125], current_lrs=initial_lrs))
        for attribute in ("total_steps", "warmup_steps", "min_lr_ratio", "completed_steps", "base_lrs"):
            with self.subTest(attribute=attribute):
                with self.assertRaises(AttributeError):
                    setattr(scheduler, attribute, 123)
        for snapshot in (scheduler.base_lrs, scheduler.get_last_lr(),
                         scheduler.state_dict()["base_lrs"]):
            snapshot[0] = 123
            snapshot.append(456)
        info = scheduler.get_info()
        self.assertIsNot(info, scheduler.get_info())
        for key in ("peak_lrs", "min_lrs", "current_lrs"):
            self.assertIsNot(info[key], scheduler.get_info()[key])
            info[key][0] = 123
            info[key].append(456)
        info["completed_steps"] = 999
        self.assertEqual(scheduler.state_dict(), initial_state)
        self.assertEqual(scheduler.get_last_lr(), initial_lrs)
        # Loading must copy caller-owned data as well.
        state = copy.deepcopy(initial_state)
        state["completed_steps"] = 2
        scheduler.load_state_dict(state)
        state["base_lrs"][0] = 123
        state["completed_steps"] = 999
        self.assertEqual(scheduler.base_lrs, [0.02, 0.005])
        self.assertEqual(scheduler.completed_steps, 2)
        self.assertEqual(scheduler.get_info()["phase"], "cosine")
        # Queries report actual optimizer rates, not a recomputed curve.
        optimizer.param_groups[0]["lr"] = 0.123
        self.assertEqual(scheduler.get_last_lr(), [0.123, 0.005])
        self.assertEqual(scheduler.get_info()["current_lrs"], [0.123, 0.005])
        for _ in range(5):
            scheduler.step()
        self.assertEqual(scheduler.completed_steps, 7)
        self.assertEqual(scheduler.get_info()["phase"], "finished")
        self.assertEqual(scheduler.get_last_lr(), [0.005, 0.00125])

    def test_warmup_cosine_resume_is_exact(self):
        # Resume in warmup, at the transition, during decay, and after completion.
        for split in (0, 1, 2, 4, 6, 8):
            with self.subTest(completed_steps=split):
                parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
                optimizer = torch.optim.AdamW([parameter], lr=0.02)
                scheduler = gpt.WarmupCosineScheduler(optimizer, 6, warmup_ratio=1 / 3, min_lr_ratio=0.25)

                def update(current_parameter, current_optimizer, current_scheduler):
                    current_parameter.grad = torch.full_like(current_parameter, 0.3)
                    current_optimizer.step()
                    current_scheduler.step()

                for _ in range(split):
                    update(parameter, optimizer, scheduler)
                state = scheduler.state_dict()
                self.assertEqual(state, dict(total_steps=6, warmup_steps=2,
                                             min_lr_ratio=0.25, completed_steps=split,
                                             base_lrs=[0.02]))
                restored_parameter = torch.nn.Parameter(parameter.detach().clone())
                restored_optimizer = torch.optim.AdamW([restored_parameter], lr=0.9)
                restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                restored_scheduler = gpt.WarmupCosineScheduler(
                    restored_optimizer, 19, warmup_ratio=0, min_lr_ratio=0.8)
                restored_scheduler.load_state_dict(state)
                self.assertEqual(restored_scheduler.state_dict(), state)
                for _ in range(8):
                    self.assertEqual(restored_optimizer.param_groups[0]["lr"],
                                     optimizer.param_groups[0]["lr"])
                    update(parameter, optimizer, scheduler)
                    update(restored_parameter, restored_optimizer, restored_scheduler)
                    self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
                    torch.testing.assert_close(restored_parameter, parameter, rtol=0, atol=0)

    def test_run_epoch_scheduler_steps_only_after_training_updates(self):
        model = gpt.MiniGPT(len(self.tokenizer.tokens))
        # Three examples / batch size two also checks the final partial batch.
        sequences = gpt.make_sequences(self.poems + self.poems[:1], self.tokenizer)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        scheduler = gpt.WarmupCosineScheduler(optimizer, 6, warmup_ratio=1 / 3)
        events = []
        used_lrs = []
        optimizer_step, scheduler_step = optimizer.step, scheduler.step

        def record_optimizer_step(*args, **kwargs):
            events.append("optimizer")
            used_lrs.append(optimizer.param_groups[0]["lr"])
            return optimizer_step(*args, **kwargs)

        def record_scheduler_step():
            events.append("scheduler")
            return scheduler_step()

        with mock.patch.object(optimizer, "step", side_effect=record_optimizer_step), \
                mock.patch.object(scheduler, "step", side_effect=record_scheduler_step):
            loss = gpt.run_epoch(model, sequences, 2, torch.device("cpu"),
                                 optimizer, 0.1, scheduler)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertEqual(events, ["optimizer", "scheduler"] * 2)
            self.assertEqual(used_lrs, [0.01, 0.02])
            self.assertEqual(scheduler.completed_steps, 2)
            state = scheduler.state_dict()
            parameters = [parameter.detach().clone() for parameter in model.parameters()]
            eval_loss = gpt.run_epoch(model, sequences, 2, torch.device("cpu"),
                                      scheduler=scheduler)
            self.assertTrue(torch.isfinite(torch.tensor(eval_loss)))
            self.assertFalse(model.training)
            self.assertEqual(events, ["optimizer", "scheduler"] * 2)
            self.assertEqual(scheduler.state_dict(), state)
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.02)
            for before, after in zip(parameters, model.parameters()):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_train_warmup_cosine_resume_in_memory(self):
        class InterruptedTraining(RuntimeError):
            pass

        checkpoints = {}
        interrupt = False
        poems = [character + self.poems[0][1:] for character in "甲乙丙丁戊己"]
        epoch_calls = []
        real_run_epoch = gpt.run_epoch

        def record_epoch(model, sequences, batch_size, device, optimizer=None, *args, **kwargs):
            epoch_calls.append((sequences.clone(), optimizer is not None))
            return real_run_epoch(model, sequences, batch_size, device, optimizer, *args, **kwargs)

        def save_in_memory(path, model, tokenizer, optimizer, **metadata):
            checkpoint = dict(model_state=model.state_dict(), tokens=tokenizer.tokens,
                              optimizer_state=optimizer.state_dict(),
                              python_rng=gpt.random.getstate(), torch_rng=torch.get_rng_state(),
                              cuda_rng=None, **metadata)
            checkpoints[str(path)] = copy.deepcopy(checkpoint)
            if interrupt and Path(path).name == "last.pt" and metadata["epoch"] == 2:
                raise InterruptedTraining()

        def load_in_memory(path, device):
            checkpoint = copy.deepcopy(checkpoints[str(path)])
            tokenizer = gpt.CharTokenizer(checkpoint["tokens"])
            model = gpt.MiniGPT(len(tokenizer.tokens)).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            return model, tokenizer, checkpoint

        def assert_nested_equal(expected, actual):
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            elif isinstance(expected, dict):
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    assert_nested_equal(expected[key], actual[key])
            elif isinstance(expected, (list, tuple)):
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(expected, actual):
                    assert_nested_equal(left, right)
            else:
                self.assertEqual(actual, expected)

        arguments = ["train", "--epochs", "4", "--batch-size", "2", "--min-freq", "1",
                     "--valid-ratio", "0.34", "--train-eval-size", "2",
                     "--device", "cpu", "--warmup-ratio", "0.25", "--min-lr-ratio", "0.1"]
        directory = Path("memory")
        last_path = directory / "last.pt"
        with mock.patch.object(gpt, "load_poems", side_effect=lambda _: list(poems)), \
                mock.patch.object(gpt, "save_checkpoint", side_effect=save_in_memory), \
                mock.patch.object(gpt, "load_checkpoint", side_effect=load_in_memory), \
                mock.patch.object(gpt, "run_epoch", side_effect=record_epoch), \
                mock.patch("builtins.print") as printed, \
                mock.patch.object(Path, "mkdir", side_effect=AssertionError("unexpected file I/O")), \
                mock.patch.object(torch, "save", side_effect=AssertionError("unexpected file I/O")), \
                mock.patch.object(torch, "load", side_effect=AssertionError("unexpected file I/O")):
            gpt.main(arguments + ["--output-dir", str(directory)])
            full = copy.deepcopy(checkpoints[str(last_path)])
            self.assertEqual(full["train_settings"]["train_eval_size"], 2)
            self.assertEqual(full["train_settings"]["train_eval_seed"], 42)
            self.assertEqual(full["train_eval_indices"], gpt.select_train_eval_indices(4, 2, 42))
            train_sequences = gpt.make_sequences(full["train_poems"], gpt.CharTokenizer(full["tokens"]))
            valid_sequences = gpt.make_sequences(full["valid_poems"], gpt.CharTokenizer(full["tokens"]))

            def assert_epoch_calls():
                self.assertEqual(len(epoch_calls), 12)
                for epoch in range(4):
                    training, train_eval, valid = epoch_calls[epoch * 3:epoch * 3 + 3]
                    self.assertEqual([len(call[0]) for call in (training, train_eval, valid)], [4, 2, 2])
                    self.assertEqual([call[1] for call in (training, train_eval, valid)], [True, False, False])
                    torch.testing.assert_close(training[0], train_sequences, rtol=0, atol=0)
                    torch.testing.assert_close(train_eval[0], train_sequences[full["train_eval_indices"]], rtol=0, atol=0)
                    torch.testing.assert_close(valid[0], valid_sequences, rtol=0, atol=0)

            assert_epoch_calls()
            for row in full["history"]:
                self.assertIn("train_eval_loss", row)
                self.assertNotIn("train_loss", row)
            self.assertTrue(any("train_eval=" in str(call) for call in printed.call_args_list))
            checkpoints.clear()
            epoch_calls.clear()
            interrupt = True
            with self.assertRaises(InterruptedTraining):
                gpt.main(arguments + ["--output-dir", str(directory)])
            self.assertEqual(checkpoints[str(last_path)]["scheduler_state"]["completed_steps"], 4)
            interrupt = False
            gpt.main(arguments + ["--output-dir", str(directory), "--resume", str(last_path),
                                  "--seed", "987", "--train-eval-size", "0"])
            resumed = checkpoints[str(last_path)]
            assert_epoch_calls()
            for key in ("model_state", "optimizer_state", "scheduler_state", "history",
                        "train_eval_indices", "train_settings", "python_rng", "torch_rng"):
                with self.subTest(checkpoint_key=key):
                    assert_nested_equal(full[key], resumed[key])
            # Legacy migration preserves old rows and initializes subset settings only once.
            legacy = copy.deepcopy(resumed)
            legacy.pop("train_eval_indices")
            legacy["train_settings"].pop("train_eval_size")
            legacy["train_settings"].pop("train_eval_seed")
            for row in legacy["history"]:
                row["train_loss"] = row.pop("train_eval_loss")
            old_history = copy.deepcopy(legacy["history"])
            checkpoints[str(last_path)] = legacy
            gpt.main(arguments + ["--output-dir", str(directory), "--resume", str(last_path),
                                  "--epochs", "5", "--seed", "314", "--train-eval-size", "1"])
            initialized = copy.deepcopy(checkpoints[str(last_path)])
            self.assertEqual(initialized["train_settings"]["train_eval_size"], 1)
            self.assertEqual(initialized["train_settings"]["train_eval_seed"], 314)
            self.assertEqual(initialized["train_eval_indices"], gpt.select_train_eval_indices(4, 1, 314))
            self.assertEqual(initialized["history"][:4], old_history)
            self.assertIn("train_eval_loss", initialized["history"][4])
            self.assertNotIn("train_loss", initialized["history"][4])
            gpt.main(arguments + ["--output-dir", str(directory), "--resume", str(last_path),
                                  "--epochs", "6", "--seed", "271", "--train-eval-size", "0"])
            migrated = checkpoints[str(last_path)]
            self.assertEqual(migrated["train_settings"], initialized["train_settings"])
            self.assertEqual(migrated["train_eval_indices"], initialized["train_eval_indices"])
            self.assertEqual(migrated["history"][:5], initialized["history"])
            # Continuing a finished zero-minimum curve must also construct then load safely.
            zero_state = copy.deepcopy(resumed["scheduler_state"])
            zero_state["min_lr_ratio"] = 0
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.SGD([parameter], lr=0)
            scheduler = gpt.WarmupCosineScheduler(optimizer, 2)
            scheduler.load_state_dict(zero_state)
            self.assertEqual(scheduler.get_last_lr(), [0.0])

    def test_train_warmup_cosine_checkpoint_resume_and_extension(self):
        class InterruptedTraining(RuntimeError):
            pass

        def assert_nested_equal(expected, actual):
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            elif isinstance(expected, dict):
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    assert_nested_equal(expected[key], actual[key])
            elif isinstance(expected, (list, tuple)):
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(expected, actual):
                    assert_nested_equal(left, right)
            else:
                self.assertEqual(actual, expected)

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            data = root / "poems.tsv"
            data.write_text("text\n" + "\n".join(self.poems), encoding="utf-8")
            full_dir, resumed_dir = root / "full", root / "resumed"
            arguments = ["train", "--data", str(data), "--epochs", "4",
                         "--batch-size", "1", "--min-freq", "1", "--device", "cpu",
                         "--warmup-ratio", "0.25", "--min-lr-ratio", "0.1",
                         "--lr", "0.002", "--seed", "123"]
            gpt.main(arguments + ["--output-dir", str(full_dir)])
            full = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
            save_checkpoint = gpt.save_checkpoint

            def save_then_interrupt(path, *args, **metadata):
                save_checkpoint(path, *args, **metadata)
                if Path(path).name == "last.pt" and metadata["epoch"] == 2:
                    raise InterruptedTraining("simulate interruption after epoch two")

            with mock.patch.object(gpt, "save_checkpoint", side_effect=save_then_interrupt):
                with self.assertRaises(InterruptedTraining):
                    gpt.main(arguments + ["--output-dir", str(resumed_dir)])
            interrupted = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(interrupted["scheduler_state"]["completed_steps"], 2)
            self.assertEqual(interrupted["scheduler_state"]["total_steps"], 4)
            gpt.main(arguments + ["--output-dir", str(resumed_dir),
                                  "--resume", str(resumed_dir / "last.pt")])
            resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
            for key in ("model_state", "optimizer_state", "scheduler_state", "history"):
                with self.subTest(checkpoint_key=key):
                    assert_nested_equal(full[key], resumed[key])
            self.assertEqual(full["scheduler_state"], dict(
                total_steps=4, warmup_steps=1, min_lr_ratio=0.1,
                completed_steps=4, base_lrs=[0.002]))
            for row, lr_start, lr_next in zip(full["history"],
                                              [0.002, 0.002, 0.0011, 0.0002],
                                              [0.002, 0.0011, 0.0002, 0.0002]):
                self.assertAlmostEqual(row["lr_start"], lr_start, places=14)
                self.assertAlmostEqual(row["lr_next"], lr_next, places=14)
            self.assertAlmostEqual(full["optimizer_state"]["param_groups"][0]["lr"], 0.0002)

            # Extending epochs must not restart warmup or stretch the saved curve.
            gpt.main(["train", "--resume", str(resumed_dir / "last.pt"),
                      "--output-dir", str(resumed_dir), "--epochs", "6", "--device", "cpu"])
            extended = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(extended["scheduler_state"]["total_steps"], 4)
            self.assertEqual(extended["scheduler_state"]["completed_steps"], 6)
            self.assertEqual(extended["history"][:4], resumed["history"])
            for row in extended["history"][4:]:
                self.assertAlmostEqual(row["lr_start"], 0.0002, places=14)
                self.assertAlmostEqual(row["lr_next"], 0.0002, places=14)

    def test_causal_attention(self):
        model = gpt.MiniGPT(len(self.tokenizer.tokens)).eval()
        inputs = gpt.make_sequences(self.poems, self.tokenizer)[:1, :-1]
        changed = inputs.clone()
        changed[:, 10:] = self.tokenizer.unk
        with torch.no_grad():
            torch.testing.assert_close(model(inputs)[:, :10], model(changed)[:, :10])

    def test_generation_and_validation(self):
        model = gpt.MiniGPT(len(self.tokenizer.tokens))
        text = gpt.generate(model, self.tokenizer, "春", top_k=1)
        self.assertEqual(len(text), 24)
        self.assertTrue(text.startswith("春"))
        self.assertEqual([text[i] for i in (5, 11, 17, 23)], list("，。，。"))
        self.assertNotIn("<", text)
        self.assertEqual(gpt.generate(model, self.tokenizer, self.poems[0]), self.poems[0])
        self.assertEqual(len(gpt.generate(model, self.tokenizer, "春", max_new_tokens=2)), 3)
        for prompt in ("🙂", "春，", "春" * 128):
            with self.assertRaises(ValueError):
                gpt.generate(model, self.tokenizer, prompt)
        with self.assertRaises(ValueError):
            gpt.generate(model, self.tokenizer, temperature=0)


if __name__ == "__main__":
    unittest.main()
