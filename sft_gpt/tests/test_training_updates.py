"""CPU-only update/budget/resume regressions with an isolated tiny GPT.

Run from the package parent: python -m unittest sft_gpt.tests.test_training_updates.
Only model-module dimension globals are patched: production MODEL_CONFIG and
512-token context remain untouched. All data/vocab/output files are temporary.
"""
import copy
import importlib
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from sft_gpt import config
from sft_gpt.cli import build_parser, main
from sft_gpt.data import make_pretrain_sequences
from sft_gpt.models import MiniGPT
from sft_gpt.pre_training import trainer
from sft_gpt.pre_training.checkpoint import save_checkpoint
from sft_gpt.pre_training.epoch import run_epoch
from sft_gpt.pre_training.optimizer import build_optimizer
from sft_gpt.pre_training.warmup_cosine_scheduler import WarmupCosineScheduler
from sft_gpt.tokenization import CharTokenizer


class TrainingUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.model_config = dict(config.MODEL_CONFIG)
        dimensions = {
            "mini_gpt": {"D_MODEL": 16, "NUM_LAYERS": 2},
            "token_embedding": {"D_MODEL": 16},
            "gpt_decoder": {"D_MODEL": 16},
            "gpt_block": {"D_MODEL": 16},
            "swiglu": {"D_MODEL": 16, "D_FF": 32},
            "multi_head_attention": {"D_MODEL": 16, "NUM_HEADS": 2},
        }
        for name, values in dimensions.items():
            module = importlib.import_module("sft_gpt.models." + name)
            self.stack.enter_context(mock.patch.multiple(module, **values))
        python_rng = random.getstate()
        self.addCleanup(random.setstate, python_rng)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.stack.enter_context(torch.random.fork_rng(devices=devices))
        self.stack.enter_context(mock.patch.object(trainer, "DEFAULT_VOCAB", self.root / "char.json"))
        self.stack.enter_context(mock.patch.object(trainer, "DEFAULT_TEXT_VOCAB", self.root / "hybrid.json"))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))
        # 100 encoded tokens -> 13 packed rows, 7 micro-batches, 3 updates.
        self.train_texts = ["abcde" + str(i) + "fg" for i in range(10)]
        self.valid_texts = ["held out text", "validation sample"]
        self.train_path = self.root / "train.jsonl"
        self.valid_path = self.root / "eval.jsonl"
        for path, texts in ((self.train_path, self.train_texts), (self.valid_path, self.valid_texts)):
            path.write_text("\n".join(json.dumps({"text": text}) for text in texts), encoding="utf-8")

    def arguments(self, directory, *overrides):
        return ["preTrain", "--train-data", str(self.train_path), "--eval-data", str(self.valid_path),
                "--output-dir", str(directory), "--tokenizer", "char", "--min-freq", "1",
                "--seq-len", "8", "--batch-size", "2", "--grad-accum-steps", "3",
                "--epochs", "2", "--max-steps", "0", "--data-workers", "1", "--device", "cpu",
                "--precision", "fp32", "--train-eval-size", "3", "--eval-batch-size", "2",
                "--eval-every-steps", "0", "--save-every-steps", "0", *map(str, overrides)]

    def state(self, directory):
        return torch.load(directory / "last.pt", map_location="cpu", weights_only=False)

    def assert_identical(self, expected, actual, location="checkpoint"):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype, location)
            self.assertEqual(expected.shape, actual.shape, location)
            self.assertTrue(torch.equal(expected, actual), location)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys(), location)
            for key, value in expected.items():
                self.assert_identical(value, actual[key], location + "." + str(key))
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(expected), type(actual), location)
            self.assertEqual(len(expected), len(actual), location)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_identical(left, right, location + "[" + str(index) + "]")
        else:
            self.assertEqual(expected, actual, location)

    def sequences(self, state):
        tokenizer = CharTokenizer(state["tokens"])
        return make_pretrain_sequences(state["train_texts"], tokenizer, state["train_settings"]["seq_len"])

    def test_tiny_fixture_keeps_production_context_and_configuration(self):
        model = MiniGPT(23)
        self.assertEqual(tuple(model.embedding.embedding.weight.shape), (23, 16))
        self.assertEqual(len(model.decoder.layers), 2)
        self.assertLess(sum(p.numel() for p in model.parameters()), 20000)
        self.assertEqual(config.MAX_SEQ_LEN, 512)
        attention = importlib.import_module("sft_gpt.models.multi_head_attention")
        self.assertEqual(attention.MAX_SEQ_LEN, 512)
        self.assertEqual(config.MODEL_CONFIG, self.model_config)

    def test_step_one_interruption_resumes_bitwise_with_changed_cli_batches(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                full_dir = self.root / ("full_" + str(checkpointing))
                resumed_dir = self.root / ("resumed_" + str(checkpointing))
                plan = ["--max-steps", "5", "--save-every-steps", "1", "--eval-every-steps", "2"]
                if checkpointing:
                    plan.append("--gradient-checkpointing")
                main(self.arguments(full_dir, *plan))

                def interrupt(path, *args, **metadata):
                    save_checkpoint(path, *args, **metadata)
                    if Path(path).name == "last.pt" and metadata["global_step"] == 1:
                        raise RuntimeError("intentional update-one interruption")

                with mock.patch.object(trainer, "save_checkpoint", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "intentional update-one"):
                        main(self.arguments(resumed_dir, *plan))
                boundary = self.state(resumed_dir)
                self.assertEqual(boundary["global_step"], 1)
                self.assertFalse(boundary["epoch_complete"])
                self.assertEqual(boundary["epoch_progress"]["next_batch"], 3)
                self.assertEqual(boundary["epoch_progress"]["updates"], 1)
                self.assertEqual(boundary["scheduler_state"]["total_steps"], 5)
                self.assertEqual(boundary["history"], [])
                # Incompatible new-experiment options must not reshape the saved plan.
                main(["preTrain", "--resume", str(resumed_dir / "last.pt"), "--epochs", "2",
                      "--batch-size", "7", "--grad-accum-steps", "9", "--seq-len", "4",
                      "--lr", "0.2", "--adam-beta2", "0.8", "--no-optimizer-decay-groups",
                      "--no-gradient-checkpointing", "--eval-every-steps", "0", "--save-every-steps", "0",
                      "--seed", "937", "--device", "cpu"])
                full, resumed = self.state(full_dir), self.state(resumed_dir)
                # Includes weights, Adam moments/groups, scheduler, all RNGs,
                # permutation/cursor, token counters, losses and complete history.
                self.assert_identical(full, resumed)
                self.assertEqual(resumed["global_step"], 5)
                self.assertFalse(resumed["epoch_complete"])
                self.assertEqual(resumed["epoch_progress"]["next_batch"], 6)
                self.assertEqual(len(resumed["history"]), 2)
                self.assertEqual(resumed["train_settings"]["batch_size"], 2)
                self.assertEqual(resumed["train_settings"]["grad_accum_steps"], 3)
                self.assertEqual(resumed["train_settings"]["gradient_checkpointing"], checkpointing)

    def test_first_update_checkpoint_resumes_to_new_output_without_existing_best(self):
        full_dir = self.root / "full_new_output"
        interrupted_dir = self.root / "first_update_only"
        resumed_dir = self.root / "new_output"
        plan = ["--save-every-steps", "1", "--eval-every-steps", "2"]
        main(self.arguments(full_dir, *plan))

        def interrupt(path, *args, **metadata):
            save_checkpoint(path, *args, **metadata)
            if Path(path).name == "last.pt" and metadata["global_step"] == 1:
                raise RuntimeError("first update before any best checkpoint")

        with mock.patch.object(trainer, "save_checkpoint", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "first update before any best"):
                main(self.arguments(interrupted_dir, *plan))
        boundary = self.state(interrupted_dir)
        self.assertEqual(boundary["best_epoch"], 0)
        self.assertEqual(boundary["best_loss"], float("inf"))
        self.assertFalse((interrupted_dir / "best.pt").exists())
        self.assertFalse(resumed_dir.exists())
        main(["preTrain", "--resume", str(interrupted_dir / "last.pt"), "--epochs", "2",
              "--output-dir", str(resumed_dir), "--device", "cpu"])
        self.assertTrue((resumed_dir / "best.pt").is_file())
        self.assert_identical(self.state(full_dir), self.state(resumed_dir))
        expected_best = torch.load(full_dir / "best.pt", map_location="cpu", weights_only=False)
        resumed_best = torch.load(resumed_dir / "best.pt", map_location="cpu", weights_only=False)
        self.assert_identical(expected_best, resumed_best)
        # The source remains the pre-validation boundary, not an output alias.
        self.assertEqual(self.state(interrupted_dir)["global_step"], 1)
        self.assertFalse((interrupted_dir / "best.pt").exists())

    def test_max_steps_is_lifetime_budget_and_resume_requires_explicit_extension(self):
        directory = self.root / "step_budget"
        main(self.arguments(directory, "--epochs", "4", "--max-steps", "4"))
        state = self.state(directory)
        self.assertEqual(state["global_step"], 4)  # Three in epoch one, one in epoch two.
        self.assertEqual(state["epoch"], 2)
        self.assertEqual(state["epoch_progress"]["updates"], 1)
        self.assertEqual(state["scheduler_state"]["completed_steps"], 4)
        original_bytes = (directory / "last.pt").read_bytes()
        with mock.patch.object(trainer, "run_epoch") as epoch:
            with self.assertRaises(SystemExit) as error:
                main(["preTrain", "--resume", str(directory / "last.pt"), "--epochs", "4", "--device", "cpu"])
            self.assertEqual(error.exception.code, 2)
            epoch.assert_not_called()
        self.assertEqual((directory / "last.pt").read_bytes(), original_bytes)
        # Independent low-level continuation oracle starts from the actual saved
        # LR curve, moments, RNG and cursor, not from a different initial plan.
        settings = state["train_settings"]
        model = MiniGPT(len(state["tokens"]))
        model.load_state_dict(state["model_state"])
        optimizer = build_optimizer(model, settings["lr"], settings["weight_decay"],
                                    decay_groups=settings["optimizer_decay_groups"], betas=settings["adam_betas"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler = WarmupCosineScheduler(optimizer, 4)
        scheduler.load_state_dict(state["scheduler_state"])
        progress = copy.deepcopy(state["epoch_progress"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        expected_objective = run_epoch(model, self.sequences(state), 2, torch.device("cpu"), optimizer,
                                       settings["label_smoothing"], scheduler, precision="fp32",
                                       grad_accum_steps=3, max_updates=1, progress=progress)
        expected_rng = torch.get_rng_state()
        main(["preTrain", "--resume", str(directory / "last.pt"), "--epochs", "4",
              "--max-steps", "5", "--device", "cpu"])
        extended = self.state(directory)
        self.assertEqual(extended["global_step"], 5)  # Not another five steps.
        self.assertEqual(extended["scheduler_state"]["completed_steps"], 5)
        self.assertEqual(extended["scheduler_state"]["total_steps"], 4)  # Original curve survives.
        self.assertEqual(extended["epoch_progress"]["next_batch"], 6)
        self.assert_identical(model.state_dict(), extended["model_state"])
        self.assert_identical(optimizer.state_dict(), extended["optimizer_state"])
        self.assert_identical(scheduler.state_dict(), extended["scheduler_state"])
        self.assert_identical(progress, extended["epoch_progress"])
        self.assert_identical(expected_rng, extended["torch_rng"])
        self.assertEqual(extended["history"][-1]["objective"], expected_objective)
        self.assertFalse(extended["history"][-1]["epoch_complete"])
        self.assertEqual(extended["history"][-1]["global_step"], 5)
        self.assertEqual(extended["trained_tokens"], 99 + progress["tokens"])

    def test_partial_budget_stop_does_not_consume_early_stopping_patience(self):
        directory = self.root / "partial_patience"
        main(self.arguments(directory, "--max-steps", "1"))
        state = self.state(directory)
        state["best_loss"] = -float("inf")  # Deterministically force no improvement.
        state["bad_epochs"] = 2
        state["train_settings"].update(patience=3, min_epochs=0)
        torch.save(state, directory / "last.pt")
        main(["preTrain", "--resume", str(directory / "last.pt"), "--epochs", "2",
              "--max-steps", "2", "--device", "cpu"])
        partial = self.state(directory)
        self.assertEqual(partial["bad_epochs"], 2)
        self.assertEqual(partial["global_step"], 2)
        self.assertFalse(partial["epoch_complete"])
        self.assertFalse(partial["history"][-1]["epoch_complete"])
        self.assertEqual(partial["history"][-1]["global_step"], 2)
        # Cancelling the budget finishes the original epoch/tail, then exactly
        # one completed bad epoch exhausts patience before epoch two starts.
        main(["preTrain", "--resume", str(directory / "last.pt"), "--epochs", "2",
              "--max-steps", "0", "--device", "cpu"])
        completed = self.state(directory)
        self.assertEqual(completed["global_step"], 3)
        self.assertEqual(completed["epoch"], 1)
        self.assertTrue(completed["epoch_complete"])
        self.assertEqual(completed["epoch_progress"], {})
        self.assertEqual(completed["bad_epochs"], 3)

    def test_max_tokens_completes_whole_accumulation_window_and_is_cumulative(self):
        for budget, expected_updates in ((1, 1), (100, 4)):
            with self.subTest(budget=budget):
                directory = self.root / ("token_budget_" + str(budget))
                main(self.arguments(directory, "--epochs", "4", "--max-tokens", budget))
                state = self.state(directory)
                sequences = self.sequences(state)
                progress = state["epoch_progress"]
                self.assertEqual(state["global_step"], expected_updates)
                self.assertEqual(state["epoch"], 1 if budget == 1 else 2)
                self.assertEqual(progress["next_batch"], 3)
                self.assertEqual(progress["updates"], 1)
                expected_tokens = int((sequences[progress["order"][:6], 1:] != -100).sum())
                self.assertEqual(progress["tokens"], expected_tokens)
                previous_epoch_tokens = (state["epoch"] - 1) * int((sequences[:, 1:] != -100).sum())
                self.assertEqual(state["trained_tokens"], previous_epoch_tokens + expected_tokens)
                self.assertGreater(state["trained_tokens"], budget)
                self.assertEqual(state["scheduler_state"]["total_steps"], expected_updates)
                with self.assertRaises(SystemExit):
                    main(["preTrain", "--resume", str(directory / "last.pt"), "--epochs", "4", "--device", "cpu"])

    def test_scheduler_counts_tail_update_and_is_clipped_by_step_budget(self):
        for budget in (0, 4, 40):
            with self.subTest(budget=budget):
                directory = self.root / ("schedule_" + str(budget))
                main(self.arguments(directory, "--epochs", "3", "--max-steps", budget))
                state = self.state(directory)
                sequences = self.sequences(state)
                self.assertEqual(len(sequences), 13)
                micro_batches = math.ceil(len(sequences) / 2)
                self.assertEqual(micro_batches, 7)
                expected = math.ceil(micro_batches / 3) * 3
                expected = min(expected, budget) if budget else expected
                self.assertEqual(state["scheduler_state"]["total_steps"], expected)
                self.assertEqual(state["scheduler_state"]["completed_steps"], expected)
                self.assertEqual(state["global_step"], expected)
                for moments in state["optimizer_state"]["state"].values():
                    self.assertEqual(int(moments["step"]), expected)

    def test_eval_and_save_intervals_are_optimizer_updates_not_micro_batches(self):
        directory = self.root / "intervals"
        events, update_count, micro_count = [], 0, 0
        real_build = trainer.build_optimizer
        real_forward = MiniGPT.forward

        def build(*args, **kwargs):
            optimizer = real_build(*args, **kwargs)
            real_step = optimizer.step

            def step(*step_args, **step_kwargs):
                nonlocal update_count
                result = real_step(*step_args, **step_kwargs)
                update_count += 1
                return result

            optimizer.step = step
            return optimizer

        def forward(model, inputs):
            nonlocal micro_count
            if model.training:
                micro_count += 1
            return real_forward(model, inputs)

        def epoch(model, sequences, batch_size, device, optimizer=None, *args, **kwargs):
            if optimizer is None:
                events.append(("eval", update_count, micro_count, len(sequences)))
            return run_epoch(model, sequences, batch_size, device, optimizer, *args, **kwargs)

        def save(path, *args, **metadata):
            events.append((Path(path).name, metadata["global_step"], micro_count, metadata["epoch_complete"]))
            return save_checkpoint(path, *args, **metadata)

        with mock.patch.object(trainer, "build_optimizer", side_effect=build), \
                mock.patch.object(MiniGPT, "forward", autospec=True, side_effect=forward), \
                mock.patch.object(trainer, "run_epoch", side_effect=epoch), \
                mock.patch.object(trainer, "save_checkpoint", side_effect=save):
            main(self.arguments(directory, "--epochs", "1", "--eval-every-steps", "2",
                                "--save-every-steps", "2", "--eval-batch-size", "1", "--eval-max-batches", "1"))
        state = self.state(directory)
        self.assertEqual(update_count, 3)
        self.assertEqual(micro_count, 7)
        self.assertEqual(events[0], ("eval", 2, 6, 1))
        self.assertEqual(events[1], ("last.pt", 2, 6, False))
        self.assertEqual(events[2][:3], ("eval", 3, 7))
        self.assertEqual(events[2][3], 3)  # Fixed training evaluation subset.
        self.assertEqual(events[3][:3], ("eval", 3, 7))
        valid = make_pretrain_sequences(state["valid_texts"], CharTokenizer(state["tokens"]), 8)
        self.assertEqual(events[3][3], len(valid))  # Final validation ignores subset cap.
        self.assertGreater(len(valid), 1)
        self.assertEqual(events[4:], [("best.pt", 3, 7, True), ("last.pt", 3, 7, True)])
        self.assertEqual(state["last_step_eval"]["global_step"], 2)

    def test_legacy_missing_settings_uses_accum_one_single_group_beta_999(self):
        directory = self.root / "legacy"

        def stop_after_epoch_one(path, *args, **metadata):
            save_checkpoint(path, *args, **metadata)
            if Path(path).name == "last.pt" and metadata["epoch"] == 1:
                raise RuntimeError("legacy checkpoint boundary")

        with mock.patch.object(trainer, "save_checkpoint", side_effect=stop_after_epoch_one):
            with self.assertRaisesRegex(RuntimeError, "legacy checkpoint boundary"):
                main(self.arguments(directory, "--grad-accum-steps", "1", "--adam-beta2", "0.999",
                                    "--no-optimizer-decay-groups", "--eval-max-batches", "0"))
        original = self.state(directory)
        resume = ["preTrain", "--resume", str(directory / "last.pt"), "--epochs", "2", "--device", "cpu"]
        main(resume)
        expected = self.state(directory)
        legacy = copy.deepcopy(original)
        for name in ("grad_accum_steps", "max_steps", "max_tokens", "gradient_checkpointing",
                     "optimizer_decay_groups", "adam_betas", "fused_adamw", "eval_batch_size",
                     "eval_every_steps", "eval_max_batches", "save_every_steps"):
            legacy["train_settings"].pop(name)
        for name in ("global_step", "trained_tokens", "epoch_complete", "epoch_progress", "last_step_eval"):
            legacy.pop(name)
        torch.save(legacy, directory / "last.pt")
        # New defaults (accum16/two groups/beta.95) must not reinterpret legacy state.
        main(resume + ["--batch-size", "9", "--grad-accum-steps", "17", "--adam-beta2", "0.7"])
        actual = self.state(directory)
        self.assert_identical(expected, actual)
        settings = actual["train_settings"]
        self.assertEqual(settings["grad_accum_steps"], 1)
        self.assertFalse(settings["optimizer_decay_groups"])
        self.assertEqual(settings["adam_betas"], (0.9, 0.999))
        self.assertEqual(settings["max_steps"], 0)
        groups = actual["optimizer_state"]["param_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["betas"], (0.9, 0.999))
        self.assertEqual(actual["global_step"], 14)

    def test_legacy_none_scheduler_recovers_global_updates_from_completed_epochs(self):
        directory = self.root / "legacy_none_scheduler"
        main(self.arguments(directory, "--epochs", "1", "--grad-accum-steps", "1",
                            "--adam-beta2", "0.999", "--no-optimizer-decay-groups"))
        legacy = self.state(directory)
        legacy["scheduler_state"] = None
        for name in ("global_step", "trained_tokens", "epoch_complete", "epoch_progress", "last_step_eval"):
            legacy.pop(name)
        for name in ("grad_accum_steps", "max_steps", "max_tokens", "gradient_checkpointing",
                     "optimizer_decay_groups", "adam_betas", "fused_adamw", "eval_batch_size",
                     "eval_every_steps", "eval_max_batches", "save_every_steps"):
            legacy["train_settings"].pop(name)
        torch.save(legacy, directory / "last.pt")
        main(["preTrain", "--resume", str(directory / "last.pt"), "--epochs", "2",
              "--max-steps", "9", "--device", "cpu"])
        actual = self.state(directory)
        self.assertEqual(actual["global_step"], 9)  # Seven historical + two new.
        self.assertEqual(actual["epoch"], 2)
        self.assertEqual(actual["epoch_progress"]["next_batch"], 2)
        self.assertEqual(actual["epoch_progress"]["updates"], 2)
        self.assertEqual(actual["scheduler_state"]["total_steps"], 2)
        self.assertEqual(actual["scheduler_state"]["completed_steps"], 2)
        self.assertEqual(len(actual["optimizer_state"]["param_groups"]), 1)
        self.assertEqual(actual["optimizer_state"]["param_groups"][0]["betas"], (0.9, 0.999))
        for moments in actual["optimizer_state"]["state"].values():
            self.assertEqual(int(moments["step"]), 9)

    def test_default_training_hyperparameters_and_saved_default_budget(self):
        args = build_parser().parse_args(["preTrain"])
        expected = dict(batch_size=4, grad_accum_steps=16, lr=3e-4, warmup_ratio=0.05,
                        min_lr_ratio=0.1, weight_decay=0.1, label_smoothing=0.0, adam_beta2=0.95,
                        optimizer_decay_groups=True, fused_adamw=False, gradient_checkpointing=False,
                        eval_batch_size=4, eval_every_steps=250, save_every_steps=250,
                        eval_max_batches=64, max_steps=None, max_tokens=None, seq_len=512)
        for name, value in expected.items():
            self.assertEqual(getattr(args, name), value, name)
        self.assertEqual(config.MAX_STEPS, 10000)
        directory = self.root / "default_budget"
        arguments = self.arguments(directory, "--epochs", "1")
        index = arguments.index("--max-steps")
        del arguments[index:index + 2]
        main(arguments)
        self.assertEqual(self.state(directory)["train_settings"]["max_steps"], 10000)

    def test_invalid_update_cli_arguments_fail_before_training(self):
        invalid = [("--grad-accum-steps", "0"), ("--grad-accum-steps", "-1"),
                   ("--batch-size", "0"), ("--max-steps", "-1"), ("--max-tokens", "-1"),
                   ("--eval-batch-size", "0"), ("--eval-every-steps", "-1"),
                   ("--eval-max-batches", "-1"), ("--save-every-steps", "-1"),
                   ("--label-smoothing", "-0.1"), ("--label-smoothing", "1.1"),
                   ("--label-smoothing", "nan"), ("--label-smoothing", "inf"),
                   ("--adam-beta2", "-0.1"), ("--adam-beta2", "1"),
                   ("--adam-beta2", "nan"), ("--adam-beta2", "inf")]
        for option, value in invalid:
            with self.subTest(option=option, value=value), mock.patch("sft_gpt.cli.pre_train") as train:
                with self.assertRaises(SystemExit) as error:
                    main(["preTrain", option, value, "--device", "cpu"])
                self.assertEqual(error.exception.code, 2)
                train.assert_not_called()
        with mock.patch("sft_gpt.cli.pre_train") as train:
            main(["preTrain", "--max-steps", "0", "--max-tokens", "0", "--label-smoothing", "1",
                  "--adam-beta2", "0", "--grad-accum-steps", "1", "--eval-every-steps", "0",
                  "--save-every-steps", "0", "--device", "cpu"])
            train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
