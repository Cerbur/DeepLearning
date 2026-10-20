"""Isolated CPU regressions for weight-only stage initialization.

Run from E:\Code\DeepLearning with .venv/Scripts/python.exe -m unittest
sft_gpt.tests.test_init_from. No real checkpoint, default vocabulary or GPU is used.
"""
import copy
import importlib
import io
import json
import random
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from sft_gpt import config
from sft_gpt.cli import build_parser, main
from sft_gpt.data import make_pretrain_sequences, select_train_eval_indices
from sft_gpt.models import MiniGPT
from sft_gpt.pre_training import trainer
from sft_gpt.pre_training.checkpoint import load_checkpoint, save_checkpoint
from sft_gpt.pre_training.epoch import run_epoch
from sft_gpt.pre_training.optimizer import build_optimizer
from sft_gpt.pre_training.warmup_cosine_scheduler import WarmupCosineScheduler
from sft_gpt.tokenization import CharTokenizer, HybridTokenizer, tokenizer_spec


class InitFromTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        python_rng = random.getstate()
        self.addCleanup(random.setstate, python_rng)
        self.stack.enter_context(torch.random.fork_rng(devices=[]))
        self.stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=False))
        dimensions = {
            "mini_gpt": {"D_MODEL": 16, "NUM_LAYERS": 1},
            "token_embedding": {"D_MODEL": 16},
            "gpt_decoder": {"D_MODEL": 16},
            "gpt_block": {"D_MODEL": 16},
            "swiglu": {"D_MODEL": 16, "D_FF": 32},
            "multi_head_attention": {"D_MODEL": 16, "NUM_HEADS": 2},
        }
        for name, values in dimensions.items():
            module = importlib.import_module("sft_gpt.models." + name)
            self.stack.enter_context(mock.patch.multiple(module, **values))
        self.production_config = copy.deepcopy(config.MODEL_CONFIG)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.stack.enter_context(redirect_stdout(self.stdout))
        self.stack.enter_context(redirect_stderr(self.stderr))
        self.defaults = [self.root / "default-char.json", self.root / "default-hybrid.json"]
        for name, path in zip(("DEFAULT_VOCAB", "DEFAULT_TEXT_VOCAB"), self.defaults):
            self.stack.enter_context(mock.patch.object(trainer, name, path))
            path.write_bytes(b"default vocabulary must remain untouched")
        self.train_texts = ["新阶段稀有字𠮷 " + str(i) + " abcdefghijklmnop" for i in range(8)]
        self.valid_texts = ["独立验证甲 abcdefgh", "独立验证乙 ijklmnop"]
        self.train_path = self.records("new-train.jsonl", self.train_texts)
        self.valid_path = self.records("new-valid.jsonl", self.valid_texts)
        # Learned IDs deliberately differ from build() order: loading may not sort.
        built = HybridTokenizer.build(["旧源中文旧源中文"], min_freq=1)
        self.tokenizer = HybridTokenizer(built.tokens[:260] + list(reversed(built.tokens[260:])))
        torch.manual_seed(812)
        random.seed(812)
        model = MiniGPT(len(self.tokenizer.tokens))
        optimizer = build_optimizer(model, 0.002, 0.1, decay_groups=True, betas=(0.9, 0.95))
        source_sequences = make_pretrain_sequences(["旧源中文 " * 4], self.tokenizer, 8)
        run_epoch(model, source_sequences, 2, torch.device("cpu"), optimizer, precision="fp32")
        self.assertTrue(optimizer.state)
        scheduler = WarmupCosineScheduler(optimizer, 57)
        schedule = scheduler.state_dict()
        schedule["completed_steps"] = 57
        scheduler.load_state_dict(schedule)
        self.source = self.root / "source" / "last.pt"
        save_checkpoint(
            self.source, model, self.tokenizer, optimizer,
            epoch=37, global_step=57, trained_tokens=99991,
            epoch_complete=False,
            epoch_progress=dict(order=torch.tensor([99999]), next_batch=999, updates=57, tokens=99991),
            train_settings=dict(batch_size=7, seq_len=4, max_steps=57, max_tokens=99991,
                                lr=0.002, patience=1, grad_accum_steps=9, precision="fp32"),
            scheduler_state=scheduler.state_dict(),
            best_loss=-100.0, best_epoch=36, bad_epochs=99,
            history=[dict(epoch=36, valid_loss=-100.0)],
            train_texts=["旧训练"], valid_texts=["旧验证"], train_eval_indices=torch.tensor([99999]),
            last_step_eval=dict(global_step=57, valid_loss=-100.0),
            initialization=dict(checkpoint="an older ancestor"))
        self.source_state = self.load(self.source)
        self.protected = self.snapshot_protected()

    def records(self, name, texts):
        path = self.root / name
        path.write_text("\n".join(json.dumps({"text": text}, ensure_ascii=False) for text in texts),
                        encoding="utf-8")
        return path

    def load(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def snapshot_protected(self):
        return {path: path.read_bytes() for path in [*self.source.parent.iterdir(), *self.defaults]}

    def assert_protected(self, snapshot=None):
        self.assertEqual(self.snapshot_protected(), self.protected if snapshot is None else snapshot)
        self.assertEqual(config.MODEL_CONFIG, self.production_config)
        self.assertEqual(config.MAX_SEQ_LEN, 512)

    def assert_nested_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(expected), type(actual))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def arguments(self, output, *overrides):
        return ["preTrain", "--init-from", str(self.source), "--train-data", str(self.train_path),
                "--eval-data", str(self.valid_path), "--output-dir", str(output),
                "--device", "cpu", "--precision", "fp32", "--data-workers", "1", "--seed", "129",
                "--epochs", "1", "--seq-len", "8", "--batch-size", "2", "--grad-accum-steps", "1",
                "--max-steps", "3", "--max-tokens", "0", "--lr", "0.0007", "--warmup-ratio", "0",
                "--min-lr-ratio", "0.2", "--weight-decay", "0.03", "--adam-beta2", "0.8",
                "--label-smoothing", "0.02", "--train-eval-size", "2", "--eval-batch-size", "2",
                "--eval-every-steps", "0", "--save-every-steps", "0", "--patience", "7",
                "--min-epochs", "0", *map(str, overrides)]

    def expected_initialization(self, state=None, path=None):
        state = self.source_state if state is None else state
        return dict(checkpoint=str((path or self.source).resolve()), source_epoch=state.get("epoch"),
                    source_global_step=state.get("global_step"), source_trained_tokens=state.get("trained_tokens"))

    def assert_cli_error(self, arguments, diagnostic=None):
        start = self.stderr.tell()
        with self.assertRaises(SystemExit) as error:
            main(arguments)
        self.assertEqual(error.exception.code, 2)
        if diagnostic:
            self.assertIn(diagnostic, self.stderr.getvalue()[start:])
        self.assert_protected()

    def test_cli_accepts_init_from_and_rejects_resume_combination(self):
        parsed = build_parser().parse_args(["preTrain", "--init-from", str(self.source)])
        self.assertEqual(parsed.init_from, self.source)
        self.assertIsNone(parsed.resume)
        self.assert_cli_error(self.arguments(self.root / "out", "--resume", self.source), "--resume")

    def test_init_requires_explicit_training_input_and_output_before_loading(self):
        cases = [
            (["--output-dir", str(self.root / "out")], "--train-data"),
            (["--eval-data", str(self.valid_path), "--output-dir", str(self.root / "out")], "--train-data"),
            (["--train-data", str(self.train_path)], "--output-dir"),
            (["--data", str(self.train_path)], "--output-dir"),
        ]
        with mock.patch.object(trainer, "load_checkpoint") as checkpoint, \
                mock.patch.object(trainer, "load_train_eval_texts") as data:
            for options, diagnostic in cases:
                with self.subTest(options=options):
                    self.assert_cli_error(["preTrain", "--init-from", str(self.source), *options], diagnostic)
            checkpoint.assert_not_called()
            data.assert_not_called()
        self.assertFalse((self.root / "out").exists())

    def test_resume_with_explicit_training_or_eval_input_points_to_init_from(self):
        with mock.patch.object(trainer, "load_checkpoint") as checkpoint:
            for option in ("--train-data", "--data", "--eval-data"):
                with self.subTest(option=option):
                    self.assert_cli_error(["preTrain", "--resume", str(self.source), option,
                                           str(self.root / "not-present.jsonl"), "--device", "cpu"], "--init-from")
            checkpoint.assert_not_called()

    def test_direct_trainer_also_rejects_source_conflicts_and_missing_inputs(self):
        for options in (
            ["--init-from", str(self.source)],
            ["--init-from", str(self.source), "--train-data", str(self.train_path)],
            ["--resume", str(self.source), "--eval-data", str(self.valid_path)],
        ):
            with self.subTest(options=options):
                args = build_parser().parse_args(["preTrain", *options])
                with mock.patch.object(trainer, "load_checkpoint") as checkpoint, self.assertRaises(ValueError):
                    trainer.pre_train(args, torch.device("cpu"))
                checkpoint.assert_not_called()
                self.assert_protected()

    def test_rejects_source_directory_nonempty_output_and_regular_file(self):
        nonempty = self.root / "occupied"
        nonempty.mkdir()
        marker = nonempty / ".keep"
        marker.write_bytes(b"do not overwrite")
        regular = self.root / "not-a-directory"
        regular.write_bytes(b"not a directory")
        for output in (self.source.parent, nonempty, regular):
            with self.subTest(output=output), mock.patch.object(trainer, "run_epoch") as epoch:
                self.assert_cli_error(self.arguments(output))
                epoch.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"do not overwrite")
        self.assertEqual(regular.read_bytes(), b"not a directory")
        self.assertEqual(list(nonempty.iterdir()), [marker])

    def test_first_update_inherits_weights_only_and_resets_all_stage_state(self):
        output = self.root / "new-stage"
        output.mkdir()  # An explicitly empty existing directory is supported.
        calls = []
        boundaries = []

        def epoch(model, sequences, batch_size, device, optimizer=None, *args, **kwargs):
            if optimizer is not None and not calls:
                calls.append(True)
                self.assert_nested_equal(self.source_state["model_state"], model.state_dict())
                self.assertEqual(optimizer.state_dict()["state"], {})
                self.assertEqual(batch_size, 2)
                self.assertEqual(device.type, "cpu")
                self.assertEqual(kwargs["grad_accum_steps"], 1)
                self.assertEqual(kwargs["max_updates"], 3)
                self.assertIsNone(kwargs["max_tokens"])
                self.assertEqual(kwargs["progress"], {})
                scheduler = args[1]
                self.assertEqual(scheduler.completed_steps, 0)
                self.assertEqual(scheduler.total_steps, 3)
                self.assertEqual(scheduler.base_lrs, [0.0007, 0.0007])
                self.assertTrue(all(group["betas"] == (0.9, 0.8) for group in optimizer.param_groups))
                expected = make_pretrain_sequences(self.train_texts, self.tokenizer, 8)
                self.assertTrue(torch.equal(sequences, expected))
            return run_epoch(model, sequences, batch_size, device, optimizer, *args, **kwargs)

        def save(path, *args, **metadata):
            self.assertEqual(Path(path).parent.resolve(), output.resolve())
            if metadata["global_step"] == 1:
                boundaries.append(copy.deepcopy(metadata))
            return save_checkpoint(path, *args, **metadata)

        with mock.patch.object(trainer, "run_epoch", side_effect=epoch), \
                mock.patch.object(trainer, "save_checkpoint", side_effect=save), \
                mock.patch.object(HybridTokenizer, "build", side_effect=AssertionError("rebuilt hybrid vocabulary")), \
                mock.patch.object(CharTokenizer, "build", side_effect=AssertionError("rebuilt char vocabulary")), \
                mock.patch.object(random, "setstate", wraps=random.setstate) as python_restore, \
                mock.patch.object(torch, "set_rng_state", wraps=torch.set_rng_state) as torch_restore:
            main(self.arguments(output, "--save-every-steps", "1", "--tokenizer", "char", "--vocab-size", "260"))
            python_restore.assert_not_called()
            torch_restore.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(boundaries), 1)
        first = boundaries[0]
        self.assertEqual(first["epoch"], 1)
        self.assertEqual(first["global_step"], 1)
        self.assertEqual(first["trained_tokens"], first["epoch_progress"]["tokens"])
        self.assertEqual(first["epoch_progress"]["next_batch"], 1)
        self.assertEqual(first["epoch_progress"]["updates"], 1)
        self.assertEqual(first["history"], [])
        self.assertEqual(first["bad_epochs"], 0)
        self.assertEqual(first["best_epoch"], 0)
        self.assertEqual(first["best_loss"], float("inf"))
        self.assertIsNone(first["last_step_eval"])
        state = self.load(output / "last.pt")
        self.assertEqual(state["initialization"], self.expected_initialization())
        self.assertEqual(state["global_step"], 3)
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(state["bad_epochs"], 0)
        self.assertEqual(state["train_texts"], self.train_texts)
        self.assertEqual(state["valid_texts"], self.valid_texts)
        self.assertEqual(state["tokens"], self.tokenizer.tokens)
        self.assertEqual(state["tokenizer"], tokenizer_spec(self.tokenizer))
        expected_indices = select_train_eval_indices(len(make_pretrain_sequences(self.train_texts, self.tokenizer, 8)), 2, 129)
        self.assert_nested_equal(state["train_eval_indices"], expected_indices)
        settings = state["train_settings"]
        for key, value in dict(seq_len=8, batch_size=2, grad_accum_steps=1, lr=0.0007,
                               weight_decay=0.03, label_smoothing=0.02, patience=7,
                               max_steps=3, max_tokens=0, train_eval_seed=129).items():
            self.assertEqual(settings[key], value)
        _, restored, _ = load_checkpoint(output / "last.pt", torch.device("cpu"))
        self.assertNotIn("𠮷", restored.token_ids)
        self.assertEqual(restored.encode("𠮷", special_tokens=False), [4 + value for value in "𠮷".encode("utf-8")])
        for text in self.train_texts + self.valid_texts:
            self.assertEqual(restored.decode(restored.encode(text)), text)
            self.assertNotIn(restored.unk, restored.encode(text))
        self.assertTrue(any(not torch.equal(value, state["model_state"][key])
                            for key, value in self.source_state["model_state"].items()))
        self.assert_protected()

    def test_data_alias_without_eval_splits_new_corpus_and_accepts_matching_vocab(self):
        output = self.root / "split-stage"
        arguments = self.arguments(output, "--vocab", self.source.parent / "vocab.json", "--valid-ratio", "0.25")
        index = arguments.index("--eval-data")
        del arguments[index:index + 2]
        arguments[arguments.index("--train-data")] = "--data"
        main(arguments)
        state = self.load(output / "last.pt")
        self.assertEqual(len(state["train_texts"]), 6)
        self.assertEqual(len(state["valid_texts"]), 2)
        self.assertEqual(set(state["train_texts"]) | set(state["valid_texts"]), set(self.train_texts))
        self.assertFalse(set(state["train_texts"]) & set(state["valid_texts"]))
        self.assertEqual(state["tokens"], self.tokenizer.tokens)
        self.assert_protected()

    def test_completed_source_and_legacy_missing_training_metadata_start_new_budget(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                state = copy.deepcopy(self.source_state)
                state["epoch_complete"] = True
                if legacy:
                    for field in ("train_settings", "epoch", "global_step", "trained_tokens", "scheduler_state"):
                        state.pop(field)
                source = self.root / ("legacy-source.pt" if legacy else "finished-source.pt")
                torch.save(state, source)
                before = source.read_bytes()
                output = self.root / ("legacy-new-stage" if legacy else "finished-new-stage")
                main(self.arguments(output, "--init-from", source, "--max-steps", "1", "--max-tokens", "1"))
                actual = self.load(output / "last.pt")
                self.assertEqual(actual["epoch"], 1)
                self.assertEqual(actual["global_step"], 1)
                self.assertLess(actual["trained_tokens"], self.source_state["trained_tokens"])
                self.assertEqual(actual["initialization"], self.expected_initialization(state, source))
                self.assertEqual(source.read_bytes(), before)
                self.assert_protected()

    def test_rejects_vocab_id_order_mismatch_without_output_or_source_changes(self):
        wrong = self.root / "wrong-vocab.json"
        tokens = list(self.tokenizer.tokens)
        tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
        HybridTokenizer(tokens).save(wrong)
        before = wrong.read_bytes()
        output = self.root / "wrong-vocab-stage"
        with mock.patch.object(trainer, "run_epoch") as epoch:
            self.assert_cli_error(self.arguments(output, "--vocab", wrong), "ID")
            epoch.assert_not_called()
        self.assertEqual(wrong.read_bytes(), before)
        self.assertFalse(output.exists())

    def test_architecture_mismatch_uses_checkpoint_loader_rejection(self):
        source = self.root / "wrong-architecture.pt"
        state = copy.deepcopy(self.source_state)
        state["model_config"] = dict(state["model_config"], incompatible_architecture=True)
        torch.save(state, source)
        before = source.read_bytes()
        output = self.root / "bad-architecture-stage"
        with mock.patch.object(trainer, "load_checkpoint", wraps=load_checkpoint) as loader, \
                mock.patch.object(trainer, "run_epoch") as epoch:
            self.assert_cli_error(self.arguments(output, "--init-from", source), "模型")
            loader.assert_called_once_with(source, torch.device("cpu"))
            epoch.assert_not_called()
        self.assertEqual(source.read_bytes(), before)
        self.assertFalse(output.exists())

    def test_missing_checkpoint_or_corpus_does_not_write_default_vocab(self):
        for option in ("--init-from", "--train-data", "--eval-data"):
            with self.subTest(option=option):
                output = self.root / ("missing-" + option[2:])
                self.assert_cli_error(self.arguments(output, option, self.root / "does-not-exist.jsonl"))
                self.assertFalse(output.exists())

    def test_initialized_stage_interruption_resumes_bitwise_including_provenance(self):
        full, interrupted = self.root / "full", self.root / "interrupted"
        plan = ["--epochs", "2", "--max-steps", "5", "--grad-accum-steps", "2",
                "--save-every-steps", "1", "--eval-every-steps", "2"]
        main(self.arguments(full, *plan))

        def interrupt(path, *args, **metadata):
            save_checkpoint(path, *args, **metadata)
            if Path(path).name == "last.pt" and metadata["global_step"] == 1:
                raise RuntimeError("new-stage update-one interruption")

        with mock.patch.object(trainer, "save_checkpoint", side_effect=interrupt), \
                self.assertRaisesRegex(RuntimeError, "new-stage update-one"):
            main(self.arguments(interrupted, *plan))
        boundary = self.load(interrupted / "last.pt")
        self.assertEqual(boundary["global_step"], 1)
        self.assertFalse(boundary["epoch_complete"])
        self.assertEqual(boundary["history"], [])
        self.assertEqual(boundary["initialization"], self.expected_initialization())
        main(["preTrain", "--resume", str(interrupted / "last.pt"), "--epochs", "2", "--device", "cpu",
              "--seed", "999", "--batch-size", "7", "--seq-len", "4", "--lr", "0.1"])
        self.assert_nested_equal(self.load(full / "last.pt"), self.load(interrupted / "last.pt"))
        self.assert_nested_equal(self.load(full / "best.pt"), self.load(interrupted / "best.pt"))
        self.assertEqual(self.load(interrupted / "last.pt")["initialization"], self.expected_initialization())
        # resume may refresh the default export; init-from itself may not.
        self.assertEqual({path: path.read_bytes() for path in self.source.parent.iterdir()},
                         {path: data for path, data in self.protected.items() if path.parent == self.source.parent})


if __name__ == "__main__":
    unittest.main()
