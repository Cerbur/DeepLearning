"""Concurrent tolerant corpus inputs, independent splits, and resume isolation."""
import inspect
import json
import random
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from sft_gpt import data
from sft_gpt.cli import build_parser, main
from sft_gpt.config import DEFAULT_TRAIN_DATA, DEFAULT_EVAL_DATA
from sft_gpt.pre_training import load_checkpoint
from sft_gpt.tokenization import HybridTokenizer


class CorpusInputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, source):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source if isinstance(source, bytes) else source.encode("utf-8"))
        return path

    def records(self, name, texts, field="text"):
        return self.write(name, json.dumps([{field: text} for text in texts], ensure_ascii=False))

    def test_recursive_reads_really_overlap_and_merge_in_path_not_completion_order(self):
        later = self.records("z/deep/later.JSON", ["shared", "last"])
        first = self.records("a/first.json", ["first", "shared", " first "])
        self.write("ignored.bin", b"unsupported")
        barrier, later_done, lock = threading.Barrier(2, timeout=10), threading.Event(), threading.Lock()
        active, peak, threads, completion = 0, 0, set(), []
        original = data._load_text_file

        def observed(path, *args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                threads.add(threading.get_ident())
            try:
                barrier.wait()  # A serial fake executor cannot satisfy this.
                if path == first and not later_done.wait(timeout=10):
                    raise AssertionError("later file did not finish concurrently")
                result = original(path, *args, **kwargs)
                with lock:
                    completion.append(path)
                if path == later:
                    later_done.set()
                return result
            finally:
                with lock:
                    active -= 1

        with mock.patch.object(data, "_load_text_file", side_effect=observed):
            actual = data.load_texts(self.root, workers=2)
        self.assertEqual(peak, 2)
        self.assertEqual(len(threads), 2)
        self.assertEqual(completion, [later, first])
        self.assertEqual(actual, ["first", "shared", " first ", "last"])
        self.assertEqual(actual, data.load_texts(self.root, workers=1))
        self.assertEqual(actual, data.load_texts(self.root, workers=8))

    def test_default_worker_count_and_legacy_minimum_are_explicit(self):
        signature = inspect.signature(data.load_texts).parameters
        self.assertEqual(signature["workers"].default, 8)
        self.assertEqual(signature["min_documents"].default, 2)
        self.assertIs(signature["strict"].default, False)
        split_signature = inspect.signature(data.load_train_eval_texts).parameters
        self.assertIsNone(split_signature["train_data"].default)
        self.assertIsNone(split_signature["eval_data"].default)
        self.assertEqual(split_signature["workers"].default, 8)
        path = self.records("one.json", ["only", "only"])
        with self.assertRaisesRegex(ValueError, "至少需要两个"):
            data.load_texts(path)
        self.assertEqual(data.load_texts(path, min_documents=1), ["only"])
        for name, values in (("workers", [0, -1, True, 1.5]),
                             ("min_documents", [0, -1, True, 1.5])):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    data.load_texts(path, **{name: value})

    def test_bad_middle_jsonl_lines_do_not_discard_later_valid_records(self):
        path = self.write("rows.jsonl", '\n{"text":"before"}\n{bad}\n{"text":3}\n'
                          '{"text":"after"}\n{"text":"before"}\n')
        with self.assertWarns(UserWarning) as warning:
            self.assertEqual(data.load_texts(path), ["before", "after"])
        self.assertIn(str(path) + ":3:", str(warning.warning))
        with self.assertRaises(ValueError) as error:
            data.load_texts(path, strict=True)
        self.assertIn(str(path) + ":3:", str(error.exception))

    def test_invalid_json_entries_and_empty_messages_are_filtered_as_whole_records(self):
        payload = [{"text": "before"}, None, 4, "not an object", {}, {"text": None},
                   {"messages": []}, {"messages": [{"role": "user", "content": "  "}]},
                   {"messages": [{"role": "user", "content": "must not leak"},
                                 {"role": "assistant", "content": ""}]},
                   {"messages": [{"role": "tool", "content": "bad role"}]},
                   {"messages": [{"role": "user", "content": "Русский"}]},
                   {"messages": [{"role": "assistant", "content": "🦊😀"}]},
                   {"messages": [{"role": "user", "content": "must not leak either"},
                                 {"role": "assistant", "content": "🦊"}]},
                   {"text": " \n\t"}, {"text": "after"},
                   {"messages": [{"role": "assistant", "content": " answer "}]}]
        path = self.write("records.json", json.dumps(payload, ensure_ascii=False))
        with self.assertWarns(UserWarning):
            self.assertEqual(data.load_texts(path), ["before", "after", "assistant:  answer "])

    def test_bad_csv_and_tsv_rows_preserve_records_after_the_error(self):
        for suffix, separator in (("csv", ","), ("tsv", "\t")):
            with self.subTest(suffix=suffix):
                source = (f"text{separator}metadata\nfirst{separator}ok\nmissing\n"
                          f"too{separator}many{separator}columns\nlast{separator}ok\n")
                path = self.write("rows." + suffix, source)
                with self.assertWarns(UserWarning):
                    self.assertEqual(data.load_texts(path), ["first", "last"])

    def test_bad_encoding_binary_controls_and_surrogates_do_not_enter_corpus(self):
        self.write("a/encoding.txt", b"apparently valid\n\n\xff\xfe")
        self.write("b/binary.txt", b"binary\x00payload")
        self.write("c/control.json", '[{"text":"bad\\u0001"},{"text":"bad\\ud800"}]')
        self.records("z/good.json", ["kept one", "kept two"])
        with self.assertWarns(UserWarning):
            self.assertEqual(data.load_texts(self.root), ["kept one", "kept two"])
        with self.assertRaises(UnicodeDecodeError):
            data.load_texts(self.root / "a/encoding.txt", strict=True)

    def test_mixed_formats_survive_file_level_exceptions_and_keep_order(self):
        self.write("a/broken.json", "[invalid JSON]")
        self.write("b/header.csv", "wrong\nignored\n")
        self.write("c/bad.csv", 'text\n"unterminated\n')
        self.write("d/paragraphs.txt", "first\n\nsecond")
        self.write("e/table.csv", 'text,meta\n"third, csv",x\n')
        self.write("f/table.tsv", "text\tmeta\nfourth\tx\n")
        self.write("g/records.jsonl", '{"text":"fifth"}\n{bad}\n{"text":"sixth"}\n')
        self.records("h/records.json", ["seventh", "second"])
        self.write("z/ignore.md", "not loaded")
        with self.assertWarns(UserWarning):
            self.assertEqual(data.load_texts(self.root),
                             ["first\n", "second", "third, csv", "fourth", "fifth", "sixth", "seventh"])

    def test_loader_applies_chinese_english_ascii_digit_and_punctuation_whitelist(self):
        allowed = "  EnglishAZ az 0129 中文〇㐀𠀀𰀀，。？！—…「」【】 !@#$%^&*()  \n\t"
        removed = "РусскийΕλληνικάالعربيةかなカナ한글हिन्दीไทยéＡ１２٣🦊😀"
        path = self.records("languages.json", [allowed, "中文" + removed + "English", "012345",
                                               removed, "...，。！？ \n\t", "🦊", "１２٣"])
        with self.assertWarns(UserWarning):
            self.assertEqual(data.load_texts(path), [allowed, "中文 English", "012345"])
        # strict controls malformed-record errors, not the language whitelist.
        with self.assertWarns(UserWarning):
            self.assertEqual(data.load_texts(path, strict=True), [allowed, "中文 English", "012345"])

    def test_cleaning_precedes_recursive_global_deduplication(self):
        self.records("z/last.json", ["leftЖright", "last"])
        self.records("a/first.json", ["left🦊right", "left😀😀right", "first"])
        self.assertEqual(data.load_texts(self.root), ["left right", "first", "last"])
        self.assertEqual(data.load_texts(self.root, workers=1), ["left right", "first", "last"])

    def test_cleaning_precedes_cross_split_overlap_removal(self):
        train = self.records("train.json", ["左🦊右", "only training"])
        evaluation = self.records("eval.json", ["左Русский右", "only evaluation"])
        with self.assertWarnsRegex(UserWarning, "训练集移除 1"):
            training, validation = data.load_train_eval_texts(train, evaluation)
        self.assertEqual(training, ["only training"])
        self.assertEqual(set(validation), {"左 右", "only evaluation"})
        self.assertFalse(set(training) & set(validation))

    def test_unreadable_file_does_not_cancel_other_files(self):
        bad = self.records("a/unreadable.json", ["not accessible"])
        self.records("z/good.json", ["one", "two"])
        original = Path.open

        def opened(path, *args, **kwargs):
            if path == bad:
                raise PermissionError("test unreadable corpus")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", new=opened), self.assertWarns(UserWarning):
            self.assertEqual(data.load_texts(self.root), ["one", "two"])

    def test_no_valid_documents_still_raises_in_tolerant_mode(self):
        path = self.write("bad.jsonl", '{bad}\n{"messages":[]}\n')
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "有效文档"):
            data.load_texts(path, min_documents=1)

    def test_defaults_resolve_to_two_directories_and_default_loader_does_not_resplit(self):
        self.assertEqual(data.resolve_data_paths(), (DEFAULT_TRAIN_DATA, DEFAULT_EVAL_DATA))
        self.assertEqual(DEFAULT_TRAIN_DATA.name, "train")
        self.assertEqual(DEFAULT_EVAL_DATA.name, "eval")
        self.records("defaults/train/one.json", ["training"])
        self.records("defaults/eval/one.json", ["evaluation"])
        train, evaluation = self.root / "defaults/train", self.root / "defaults/eval"
        with mock.patch.object(data, "DEFAULT_TRAIN_DATA", train), \
                mock.patch.object(data, "DEFAULT_EVAL_DATA", evaluation):
            self.assertEqual(data.load_train_eval_texts(), (["training"], ["evaluation"]))

    def test_explicit_training_only_ignores_both_defaults_and_splits_deterministically(self):
        texts = [f"document {index}" for index in range(10)]
        path = self.records("explicit.json", texts)
        self.assertEqual(data.resolve_data_paths(str(path)), (path, None))
        state = random.getstate()
        with mock.patch.object(data, "DEFAULT_TRAIN_DATA", self.root / "missing-train"), \
                mock.patch.object(data, "DEFAULT_EVAL_DATA", self.root / "missing-eval"), \
                mock.patch.object(data, "load_texts", wraps=data.load_texts) as loader:
            first = data.load_train_eval_texts(path, seed=31, valid_ratio=0.3)
            second = data.load_train_eval_texts(path, seed=31, valid_ratio=0.3)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), state)
        self.assertEqual((len(first[0]), len(first[1])), (7, 3))
        self.assertEqual(set(first[0]) | set(first[1]), set(texts))
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual([call.args[0] for call in loader.call_args_list], [path, path])
        one = self.records("single.json", ["one"])
        with self.assertRaisesRegex(ValueError, "至少需要两个"):
            data.load_train_eval_texts(one)

    def test_eval_only_is_rejected_before_loading_or_using_default_training(self):
        path = self.records("evaluation.json", ["evaluation"])
        with mock.patch.object(data, "load_texts") as loader:
            with self.assertRaisesRegex(ValueError, "--train-data"):
                data.load_train_eval_texts(eval_data=path)
            loader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "--train-data"):
            data.resolve_data_paths(eval_data=path)

    def test_two_directories_load_concurrently_without_second_split(self):
        train_texts, eval_texts = [f"train {i}" for i in range(5)], ["eval one", "eval two"]
        self.records("train/deep/documents.json", train_texts)
        self.records("eval/deep/documents.json", eval_texts)
        train, evaluation = self.root / "train", self.root / "eval"
        original, barrier, threads, lock = data.load_texts, threading.Barrier(2, timeout=10), set(), threading.Lock()

        def observed(path, *args, **kwargs):
            with lock:
                threads.add(threading.get_ident())
            barrier.wait()
            return original(path, *args, **kwargs)

        state = random.getstate()
        with mock.patch.object(data, "load_texts", side_effect=observed) as loader:
            training, validation = data.load_train_eval_texts(train, evaluation, seed=7, valid_ratio=0.9, workers=2)
        self.assertEqual(len(threads), 2)
        self.assertEqual(set(training), set(train_texts))
        self.assertEqual(set(validation), set(eval_texts))
        self.assertEqual(random.getstate(), state)
        self.assertEqual({call.args[0] for call in loader.call_args_list}, {train, evaluation})
        for call in loader.call_args_list:
            self.assertEqual(call.kwargs["min_documents"], 1)
            self.assertEqual(call.kwargs["workers"], 2)
        self.assertEqual((training, validation), data.load_train_eval_texts(train, evaluation, seed=7))

    def test_cross_split_duplicates_belong_only_to_evaluation(self):
        train = self.records("train.json", ["train only", "shared", "shared", " train only "])
        evaluation = self.records("eval.json", ["shared", "eval only", "shared"])
        with self.assertWarnsRegex(UserWarning, "训练集移除 1"):
            training, validation = data.load_train_eval_texts(train, evaluation)
        self.assertEqual(set(training), {"train only", " train only "})
        self.assertEqual(set(validation), {"shared", "eval only"})
        self.assertFalse(set(training) & set(validation))

    def test_empty_train_eval_and_training_emptied_by_overlap_raise(self):
        train = self.records("train.json", ["shared"])
        evaluation = self.records("eval.json", ["shared"])
        empty = self.write("empty.txt", " \n\n")
        for paths in ((empty, evaluation), (train, empty)):
            with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "有效文档"):
                data.load_train_eval_texts(*paths)
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "训练集为空"):
            data.load_train_eval_texts(train, evaluation)

    def test_poetry_separate_splits_accept_one_valid_poem_each(self):
        first = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
        second = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
        train = self.records("train.json", [first, "not a poem"])
        evaluation = self.records("eval.json", [second])
        with mock.patch("builtins.print"):
            self.assertEqual(data.load_train_eval_texts(train, evaluation, poetry=True), ([first], [second]))
            with self.assertRaisesRegex(ValueError, "至少需要 2"):
                data.load_poems(train)

    def test_custom_field_is_forwarded_to_both_independent_inputs(self):
        train = self.records("train.json", ["training"], field="body")
        evaluation = self.records("eval.json", ["evaluation"], field="body")
        self.assertEqual(data.load_train_eval_texts(train, evaluation, text_field="body"),
                         (["training"], ["evaluation"]))

    def test_max_documents_caps_total_and_preserves_nonempty_disjoint_splits(self):
        train_texts, eval_texts = [f"train {i}" for i in range(9)], [f"eval {i}" for i in range(4)]
        train, evaluation = self.records("train.json", train_texts), self.records("eval.json", eval_texts)
        for cap in (0, 2, 3, 5, 12, 13, 50):
            for separate in (False, True):
                with self.subTest(cap=cap, separate=separate):
                    arguments = (train, evaluation) if separate else (train,)
                    training, validation = data.load_train_eval_texts(*arguments, max_documents=cap, seed=9)
                    total = len(train_texts) + (len(eval_texts) if separate else 0)
                    self.assertEqual(len(training) + len(validation), min(cap, total) if cap else total)
                    self.assertTrue(training)
                    self.assertTrue(validation)
                    self.assertFalse(set(training) & set(validation))
                    self.assertTrue(set(training).issubset(train_texts))
                    self.assertTrue(set(validation).issubset(eval_texts if separate else train_texts))
                    self.assertEqual((training, validation),
                                     data.load_train_eval_texts(*arguments, max_documents=cap, seed=9))

    def test_invalid_max_documents_and_ratios_are_rejected(self):
        for cap in (-1, 1, True, 1.5, None):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "max_documents"):
                data.load_train_eval_texts(max_documents=cap)
        for ratio in (0, 1, -0.1, 1.1, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "valid_ratio"):
                data.load_train_eval_texts(valid_ratio=ratio)

    def test_cli_alias_defaults_resolution_and_worker_forwarding(self):
        args = build_parser().parse_args(["preTrain"])
        self.assertIsNone(args.train_data)
        self.assertIsNone(args.eval_data)
        self.assertEqual(args.data_workers, 8)
        self.assertFalse(hasattr(args, "data"))
        for alias in ("--train-data", "--data"):
            args = build_parser().parse_args(["preTrain", alias, "train", "--eval-data", "eval", "--data-workers", "3"])
            self.assertEqual((args.train_data, args.eval_data, args.data_workers), (Path("train"), Path("eval"), 3))
        for arguments, expected in (([], (DEFAULT_TRAIN_DATA, DEFAULT_EVAL_DATA)),
                                    (["--train-data", "explicit"], (Path("explicit"), None)),
                                    (["--data", "explicit", "--eval-data", "heldout"],
                                     (Path("explicit"), Path("heldout")))):
            with self.subTest(arguments=arguments), mock.patch("sft_gpt.cli.pre_train") as run, \
                    mock.patch("builtins.print"):
                main(["preTrain", "--device", "cpu"] + arguments)
                actual = run.call_args.args[0]
                self.assertEqual((actual.train_data, actual.eval_data), expected)
        for arguments in (["--eval-data", "heldout"], ["--data-workers", "0"], ["--data-workers", "-1"]):
            with self.subTest(arguments=arguments), mock.patch("sys.stderr"), \
                    mock.patch("sft_gpt.cli.pre_train") as run, self.assertRaises(SystemExit) as error:
                main(["preTrain"] + arguments)
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


    def test_independent_eval_training_and_checkpoint_resume_never_reread_or_leak(self):
        training_texts, validation_texts = ["甲甲甲", "乙乙"], ["验证专属"]
        train = self.records("train/input.json", training_texts + validation_texts)
        evaluation = self.records("eval/input.json", validation_texts)
        output = self.root / "checkpoints"
        arguments = ["preTrain", "--train-data", str(train.parent), "--eval-data", str(evaluation.parent),
                     "--epochs", "1", "--seq-len", "8", "--batch-size", "2", "--min-freq", "1",
                     "--data-workers", "3", "--valid-ratio", "0.9", "--device", "cpu",
                     "--output-dir", str(output)]
        expected_build = HybridTokenizer.build
        with mock.patch("sft_gpt.pre_training.trainer.MiniGPT", TinyGPT), \
                mock.patch("sft_gpt.pre_training.checkpoint.MiniGPT", TinyGPT), \
                mock.patch("sft_gpt.pre_training.trainer.DEFAULT_TEXT_VOCAB", self.root / "vocab.json"), \
                mock.patch("builtins.print"):
            with mock.patch.object(HybridTokenizer, "build", wraps=expected_build) as build, \
                    mock.patch("sft_gpt.pre_training.trainer.load_train_eval_texts",
                               wraps=data.load_train_eval_texts) as loader, \
                    self.assertWarnsRegex(UserWarning, "训练集移除 1"):
                main(arguments)
            loader.assert_called_once_with(train.parent, evaluation.parent, text_field="text", seed=42,
                                           valid_ratio=0.9, max_documents=0, workers=3, poetry=False)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(set(build.call_args.args[0]), set(training_texts))
            _, tokenizer, original = load_checkpoint(output / "last.pt", torch.device("cpu"))
            self.assertEqual(set(original["train_texts"]), set(training_texts))
            self.assertEqual(original["valid_texts"], validation_texts)
            self.assertEqual(tokenizer.tokens, expected_build(original["train_texts"], min_freq=1).tokens)
            for character in "验证专属":
                self.assertNotIn(character, tokenizer.token_ids)
            for text in validation_texts:
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
                self.assertNotIn(tokenizer.unk, tokenizer.encode(text))
            # No new inputs are supplied. Any attempt to resolve, read or rebuild
            # a tokenizer on resume must still fail immediately.
            with mock.patch("sft_gpt.cli.resolve_data_paths", side_effect=AssertionError("resume resolved inputs")) as resolve, \
                    mock.patch("sft_gpt.pre_training.trainer.load_train_eval_texts",
                               side_effect=AssertionError("resume loaded inputs")) as split_loader, \
                    mock.patch.object(data, "load_texts", side_effect=AssertionError("resume reread corpus")) as text_loader, \
                    mock.patch.object(HybridTokenizer, "build", side_effect=AssertionError("resume rebuilt tokenizer")) as build:
                main(["preTrain", "--resume", str(output / "last.pt"), "--epochs", "2",
                      "--tokenizer", "char",
                      "--seq-len", "3", "--device", "cpu", "--output-dir", str(output)])
                resolve.assert_not_called()
                split_loader.assert_not_called()
                text_loader.assert_not_called()
                build.assert_not_called()
            _, restored_tokenizer, resumed = load_checkpoint(output / "last.pt", torch.device("cpu"))
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(len(resumed["history"]), 2)
            for key in ("train_texts", "valid_texts", "train_settings", "train_eval_indices", "tokens", "tokenizer"):
                self.assertEqual(resumed[key], original[key])
            self.assertEqual(restored_tokenizer.tokens, tokenizer.tokens)
            self.assertEqual(resumed["train_settings"]["seq_len"], 8)
            self.assertTrue(any(not torch.equal(original["model_state"][name], value)
                                for name, value in resumed["model_state"].items()))


class TinyGPT(torch.nn.Module):
    """Cheap real gradients/checkpoints; patched only at trainer/load boundaries."""
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.output = torch.nn.Linear(8, vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, token_ids):
        return self.output(self.embedding(token_ids))
