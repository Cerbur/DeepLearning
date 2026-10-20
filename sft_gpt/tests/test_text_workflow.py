"""通用文本训练、变长损失、分词器绑定与精确恢复。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from sft_gpt import HybridTokenizer, CharTokenizer, MiniGPT
from sft_gpt.cli import main, build_parser
from sft_gpt.config import MAX_SEQ_LEN
from sft_gpt.pre_training import load_checkpoint, save_checkpoint, run_epoch
from sft_gpt.tests.test_workflow import assert_nested_equal
from sft_gpt.tokenization import tokenizer_spec


class TextWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_general_cli_defaults_and_validation(self):
        args = build_parser().parse_args(["preTrain"])
        self.assertEqual(args.tokenizer, "hybrid")
        self.assertEqual(args.seq_len, MAX_SEQ_LEN)
        self.assertFalse(args.poetry)
        self.assertEqual(build_parser().parse_args(["preTrain", "--max-poems", "4"]).max_documents, 4)
        for option, value in (("--seq-len", "0"), ("--seq-len", str(MAX_SEQ_LEN + 1)),
                              ("--vocab-size", "259"), ("--max-documents", "1")):
            with self.subTest(option=option, value=value), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                main(["preTrain", option, value])

    def test_hybrid_checkpoint_roundtrip_and_spec_mismatch(self):
        tokenizer = HybridTokenizer.build(["你好你好 Hello"], min_freq=2)
        model = MiniGPT(len(tokenizer.tokens))
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, tokenizer, optimizer)
            restored_model, restored, state = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(state["format_version"], 2)
            self.assertEqual(state["training_stage"], "pre_train")
            self.assertEqual(state["tokenizer"], tokenizer_spec(tokenizer))
            self.assertEqual(restored.encode("稀有字🦊\n Hello"), tokenizer.encode("稀有字🦊\n Hello"))
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(restored_model.state_dict()[name], tensor, rtol=0, atol=0)
            self.assertIs(restored_model.output_proj.weight, restored_model.embedding.embedding.weight)
            self.assertEqual(json.loads((path.parent / "vocab.json").read_text(encoding="utf-8"))["type"], "HybridTokenizer")
            bad = copy.deepcopy(state)
            bad["tokens"][-1], bad["tokens"][-2] = bad["tokens"][-2], bad["tokens"][-1]
            torch.save(bad, path)
            with self.assertRaisesRegex(ValueError, "ID 顺序"):
                load_checkpoint(path, torch.device("cpu"))
            bad = copy.deepcopy(state)
            bad.pop("tokenizer")
            torch.save(bad, path)
            with self.assertRaisesRegex(ValueError, "缺少 tokenizer"):
                load_checkpoint(path, torch.device("cpu"))

    def test_padding_loss_ignores_labels_and_weights_real_tokens(self):
        model = MiniGPT(32).eval()
        sequences = torch.tensor([[0, 5, 6, 7, 1], [0, 8, 1, -100, -100]])
        expected = torch.nn.functional.cross_entropy(
            model(sequences[:, :-1].clamp_min(0)).reshape(-1, 32),
            sequences[:, 1:].reshape(-1), ignore_index=-100).item()
        # Batch size one must still aggregate by real tokens, not equally by row.
        for batch_size in (1, 2):
            actual = run_epoch(model, sequences, batch_size, torch.device("cpu"))
            self.assertAlmostEqual(actual, expected, places=6)
        optimizer = torch.optim.AdamW(model.parameters())
        loss = run_epoch(model, sequences, 2, torch.device("cpu"), optimizer, 0.1)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_generic_char_vocabulary_does_not_require_poetry_punctuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, vocab = root / "input.jsonl", root / "char.json"
            data.write_text(' {"text":"hello world"}\n{"text":"hi there"}\n', encoding="utf-8")
            CharTokenizer(["<BOS>", "<EOS>", "<UNK>"] + sorted(set("hello worldhi there"))).save(vocab)
            before = vocab.read_bytes()
            with mock.patch("builtins.print"):
                main(["preTrain", "--data", str(data), "--vocab", str(vocab), "--epochs", "1",
                      "--seq-len", "8", "--batch-size", "2", "--device", "cpu",
                      "--output-dir", str(root / "out")])
            self.assertEqual(vocab.read_bytes(), before)
            _, restored, state = load_checkpoint(root / "out" / "last.pt", torch.device("cpu"))
            self.assertIsInstance(restored, CharTokenizer)
            self.assertEqual(state["train_settings"]["sequence_mode"], "packed")

    def test_packed_resume_is_exact_and_does_not_build_from_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dialogs.jsonl"
            documents = ["你好，你好。\nHello world! " * 3, "user: 今天吃什么？\nassistant: 面条🍜",
                         "中文和English 保留  两个空格", "罕见字符𠮷与emoji🦊"]
            data.write_text("\n".join(json.dumps({"text": text}, ensure_ascii=False)
                                      for text in documents), encoding="utf-8")
            arguments = ["preTrain", "--data", str(data), "--epochs", "2", "--batch-size", "2",
                         "--seq-len", "16", "--min-freq", "1", "--valid-ratio", "0.5", "--device", "cpu"]
            full_dir, interrupted_dir = root / "full", root / "interrupted"
            real_save = save_checkpoint

            def interrupt_after_first(path, *args, **metadata):
                real_save(path, *args, **metadata)
                if Path(path).name == "last.pt" and metadata["epoch"] == 1:
                    raise RuntimeError("test interruption")

            with mock.patch("sft_gpt.pre_training.trainer.DEFAULT_TEXT_VOCAB", root / "text_vocab.json"), \
                    mock.patch("builtins.print"):
                main(arguments + ["--output-dir", str(full_dir)])
                with mock.patch("sft_gpt.pre_training.trainer.save_checkpoint", side_effect=interrupt_after_first), \
                        self.assertRaisesRegex(RuntimeError, "test interruption"):
                    main(arguments + ["--output-dir", str(interrupted_dir)])
                # Different CLI sequence/tokenizer settings must not alter saved encoding.
                main(["preTrain", "--resume", str(interrupted_dir / "last.pt"), "--epochs", "2",
                      "--output-dir", str(interrupted_dir), "--seq-len", "7", "--tokenizer", "char",
                      "--device", "cpu"])
            _, tokenizer, full = load_checkpoint(full_dir / "last.pt", torch.device("cpu"))
            _, _, resumed = load_checkpoint(interrupted_dir / "last.pt", torch.device("cpu"))
            assert_nested_equal(self, full, resumed)
            self.assertEqual(set(full["train_texts"]) & set(full["valid_texts"]), set())
            expected = HybridTokenizer.build(full["train_texts"], min_freq=1)
            self.assertEqual(tokenizer.tokens, expected.tokens)
            self.assertEqual(full["train_settings"]["seq_len"], 16)
            for text in full["valid_texts"]:
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
                self.assertNotIn(tokenizer.unk, tokenizer.encode(text))
            self.assertEqual(json.loads((root / "text_vocab.json").read_text(encoding="utf-8"))["tokens"], expected.tokens)
