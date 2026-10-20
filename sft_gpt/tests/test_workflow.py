"""训练、恢复、旧检查点兼容及词表绑定的集成测试。"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from sft_gpt.cli import main
from sft_gpt.config import DEFAULT_DATA
from sft_gpt.data import load_poems, make_sequences
from sft_gpt.inference import generate
from sft_gpt.tokenization import CharTokenizer
from sft_gpt.pre_training import load_checkpoint
from sft_gpt.tests.test_model_preservation import load_original


def assert_nested_equal(case, expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        case.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            assert_nested_equal(case, expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        case.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            assert_nested_equal(case, left, right)
    else:
        case.assertEqual(expected, actual)


def assert_pretrain_checkpoint_matches_original(case, expected, actual):
    case.assertEqual(actual.get("training_stage"), "pre_train")
    original_payload = dict(actual)
    original_payload.pop("training_stage")
    case.assertEqual(original_payload.pop("tokenizer")["type"], "CharTokenizer")
    original_payload["train_settings"].pop("sequence_mode", None)
    original_payload["train_settings"].pop("seq_len", None)
    # 精度策略是新增元数据；CPU auto 仍与原 FP32 轨迹逐位一致。
    case.assertEqual(original_payload["train_settings"].pop("precision"), "auto")
    assert_nested_equal(case, expected, original_payload)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = load_original()
        cls.poems = ["床前明月光，疑是地上霜。举头望明月，低头思故乡。",
                     "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"]

    def test_dataset_copy_matches_original(self):
        self.assertEqual(DEFAULT_DATA.read_bytes(), self.original.DEFAULT_DATA.read_bytes())
        self.assertGreater(len(load_poems(DEFAULT_DATA)), 16000)

    def test_old_checkpoint_load_and_generation(self):
        tokenizer = self.original.CharTokenizer.build(self.poems, 1)
        model = self.original.MiniGPT(len(tokenizer.tokens)).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            self.original.save_checkpoint(path, model, tokenizer, optimizer)
            new, restored, _ = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(tokenizer.tokens, restored.tokens)
            self.assertFalse((path.parent / "vocab.json").exists())
            inputs = make_sequences(self.poems, restored)[:, :-1]
            with torch.no_grad():
                torch.testing.assert_close(new(inputs), model(inputs), rtol=0, atol=0)
            self.assertEqual(generate(new, restored, "春", top_k=1, structured=True),
                             self.original.generate(model, tokenizer, "春", top_k=1))
            self.assertEqual(generate(new, restored, " 春 ", top_k=1, structured=True),
                             self.original.generate(model, tokenizer, " 春 ", top_k=1))

    def test_missing_punctuation_is_rejected_without_changing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "poems.tsv"
            data.write_text("text\n" + "\n".join(self.poems), encoding="utf-8")
            vocab = root / "vocab.json"
            for punctuation in ([], ["，"], ["。"]):
                with self.subTest(punctuation=punctuation):
                    tokens = ["<BOS>", "<EOS>", "<UNK>", "春"] + punctuation
                    tokenizer = CharTokenizer(tokens)
                    tokenizer.save(vocab)
                    before = vocab.read_bytes()
                    with mock.patch("builtins.print"), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main(["preTrain", "--poetry", "--data", str(data), "--vocab", str(vocab),
                              "--epochs", "1", "--device", "cpu", "--output-dir", str(root / "out")])
                    self.assertEqual(vocab.read_bytes(), before)
                    self.assertFalse((root / "out").exists())
                    model = self.original.MiniGPT(len(tokens))
                    with self.assertRaisesRegex(ValueError, "中文标点"):
                        generate(model, tokenizer, structured=True)
                    self.assertEqual(tokenizer.tokens, tokens)

    def test_training_and_resume_match_original_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "poems.tsv"
            data.write_text("text\n" + "\n".join(self.poems), encoding="utf-8")
            old_dir, new_dir = root / "old", root / "new"
            arguments = ["train", "--data", str(data), "--epochs", "2",
                         "--batch-size", "1", "--min-freq", "1", "--device", "cpu"]
            # Do not leave test vocabularies in the actual .data directory.
            with mock.patch("sft_gpt.pre_training.trainer.DEFAULT_VOCAB", root / "vocab.json"), \
                    mock.patch("builtins.print"):
                self.original.main(arguments + ["--output-dir", str(old_dir)])
                main(["preTrain", "--poetry"] + arguments[1:] + ["--output-dir", str(new_dir)])
                for name in ("best.pt", "last.pt"):
                    old = torch.load(old_dir / name, weights_only=False)
                    new = torch.load(new_dir / name, weights_only=False)
                    assert_pretrain_checkpoint_matches_original(self, old, new)
                vocabulary = CharTokenizer.load(root / "vocab.json")
                self.assertEqual(vocabulary.tokens, new["tokens"])
                self.assertEqual(CharTokenizer.load(new_dir / "vocab.json").tokens, new["tokens"])
                for runner, output, command in ((self.original.main, old_dir, "train"),
                                                (main, new_dir, "preTrain")):
                    runner([command, "--resume", str(output / "last.pt"),
                            "--output-dir", str(output), "--epochs", "3", "--device", "cpu"])
                assert_pretrain_checkpoint_matches_original(self, torch.load(old_dir / "last.pt", weights_only=False),
                                    torch.load(new_dir / "last.pt", weights_only=False))
                # Reuse an explicit vocabulary, then reject a different ID order on resume.
                explicit = root / "explicit.json"
                vocabulary.save(explicit)
                output = root / "explicit"
                real_save = CharTokenizer.save

                def save_output_only(tokenizer, path=None):
                    self.assertNotEqual(Path(path).resolve(), explicit.resolve())
                    return real_save(tokenizer, path)

                before = explicit.read_bytes()
                with mock.patch.object(CharTokenizer, "save", autospec=True, side_effect=save_output_only):
                    main(["preTrain", "--poetry"] + arguments[1:] + ["--vocab", str(explicit), "--output-dir", str(output)])
                self.assertEqual(explicit.read_bytes(), before)
                wrong = vocabulary.tokens.copy()
                wrong[-1], wrong[-2] = wrong[-2], wrong[-1]
                CharTokenizer(wrong).save(explicit)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(["preTrain", "--resume", str(output / "last.pt"),
                          "--output-dir", str(output), "--epochs", "3", "--device", "cpu",
                          "--vocab", str(explicit)])
                self.assertEqual(error.exception.code, 2)
