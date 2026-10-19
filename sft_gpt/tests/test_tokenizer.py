"""Tokenizer tests; package initialization also imports the existing torch models."""

import ast
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sft_gpt.config import DEFAULT_VOCAB, MAX_TOKEN_FREQ
from sft_gpt.tokenization import CharTokenizer
from sft_gpt.tokenization import char_tokenizer as tokenizer_module


class TestCharTokenizer(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "vocab.json"
        self.tokens = ["<BOS>", "<EOS>", "<UNK>", "春", "江", "。", "，"]
        self.tokenizer = CharTokenizer(self.tokens)

    def test_original_api_and_ids(self):
        # Extract only the original class so compatibility checks need no torch.
        original_path = Path(__file__).resolve().parents[2] / "mini_gpt" / "mini_gpt.py"
        tree = ast.parse(original_path.read_text(encoding="utf-8"))
        definition = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CharTokenizer"
        )
        namespace = {"Counter": Counter, "MAX_TOKEN_FREQ": MAX_TOKEN_FREQ}
        exec(compile(ast.Module(body=[definition], type_ignores=[]),
                     str(original_path), "exec"), namespace)
        original = namespace["CharTokenizer"]
        poems = ["春江花月夜，春江。", "夜江春。"]
        for min_freq in (1, 2, MAX_TOKEN_FREQ):
            with self.subTest(min_freq=min_freq):
                expected = original.build(poems, min_freq=min_freq)
                actual = CharTokenizer.build(poems, min_freq=min_freq)
                self.assertEqual(actual.tokens, expected.tokens)
                self.assertEqual(actual.token_ids, expected.token_ids)
                self.assertEqual((actual.bos, actual.eos, actual.unk), (0, 1, 2))
                for special_tokens in (True, False):
                    ids = actual.encode("春江未知，。", special_tokens=special_tokens)
                    self.assertEqual(ids, expected.encode(
                        "春江未知，。", special_tokens=special_tokens))
                    self.assertEqual(actual.decode(ids), expected.decode(ids))

    def test_build_has_deterministic_sorted_order(self):
        poems = ["月春月", "江春", "春😀"]
        first = CharTokenizer.build(poems, min_freq=2)
        second = CharTokenizer.build(reversed(poems), min_freq=2)
        expected = ["<BOS>", "<EOS>", "<UNK>"] + sorted("春月，。")
        self.assertEqual(first.tokens, expected)
        self.assertEqual(second.token_ids, first.token_ids)
        self.assertEqual(CharTokenizer.build([], min_freq=100).tokens,
                         ["<BOS>", "<EOS>", "<UNK>"] + sorted("，。"))

    def test_default_frequency_comes_from_config(self):
        self.assertEqual(CharTokenizer.build.__func__.__defaults__, (MAX_TOKEN_FREQ,))
        actual = CharTokenizer.build(["春" * MAX_TOKEN_FREQ, "江"])
        self.assertIn("春", actual.token_ids)
        if MAX_TOKEN_FREQ > 1:
            self.assertNotIn("江", actual.token_ids)
        self.assertEqual(tokenizer_module.DEFAULT_VOCAB, DEFAULT_VOCAB)

    def test_unknown_and_empty_text(self):
        self.assertEqual(self.tokenizer.encode("春陌😀"), [0, 3, 2, 2, 1])
        self.assertEqual(self.tokenizer.encode("春陌😀", special_tokens=False), [3, 2, 2])
        self.assertEqual(self.tokenizer.decode([0, 3, 2, 2, 1]), "春<UNK><UNK>")
        self.assertEqual(self.tokenizer.encode(""), [0, 1])
        self.assertEqual(self.tokenizer.encode("", special_tokens=False), [])
        self.assertEqual(self.tokenizer.decode([0, 1]), "")
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_save_load_preserves_exact_order_and_ids(self):
        self.assertEqual(self.tokenizer.save(self.path), self.path)
        loaded = CharTokenizer.load(str(self.path))
        self.assertEqual(loaded.tokens, self.tokens)
        self.assertEqual(loaded.token_ids, self.tokenizer.token_ids)
        self.assertEqual((loaded.bos, loaded.eos, loaded.unk), (0, 1, 2))
        self.assertEqual(loaded.encode("春江陌，。"),
                         self.tokenizer.encode("春江陌，。"))
        self.assertEqual(loaded.decode(loaded.encode("春江陌，。")), "春江<UNK>，。")
        # Do not sort an existing vocabulary, even when it was not built here.
        self.assertNotEqual(loaded.tokens[3:], sorted(loaded.tokens[3:]))

    def test_saved_json_is_utf8_with_schema_and_complete_tokens(self):
        self.tokenizer.save(self.path)
        raw = self.path.read_bytes()
        self.assertIn("春".encode("utf-8"), raw)
        self.assertNotIn(b"\\u6625", raw)
        self.assertEqual(json.loads(raw.decode("utf-8")),
                         {"version": 1, "type": "CharTokenizer", "tokens": self.tokens})

    def test_default_save_and_load_path(self):
        default = self.directory / "nested" / "tokenizer" / "vocab.json"
        with patch.object(tokenizer_module, "DEFAULT_VOCAB", default):
            self.assertEqual(self.tokenizer.save(), default)
            self.assertEqual(CharTokenizer.load().tokens, self.tokens)
            self.assertEqual(CharTokenizer.load(None).token_ids, self.tokenizer.token_ids)
        self.assertEqual(tokenizer_module.DEFAULT_VOCAB, DEFAULT_VOCAB)

    def test_missing_file_does_not_rebuild(self):
        with patch.object(CharTokenizer, "build", side_effect=AssertionError("no rebuilding")):
            with self.assertRaises(FileNotFoundError):
                CharTokenizer.load(self.path)
        self.assertFalse(self.path.exists())
        with patch.object(tokenizer_module, "DEFAULT_VOCAB", self.path):
            with self.assertRaises(FileNotFoundError):
                CharTokenizer.load()

    def test_constructor_rejects_invalid_tokens(self):
        invalid = [
            [], ["<EOS>", "<BOS>", "<UNK>"],
            ["<BOS>", "<EOS>", "unknown"],
            self.tokens + ["春"], self.tokens + ["<BOS>"],
            self.tokens + [""], self.tokens + ["春江"],
            self.tokens + [1], self.tokens + [None], self.tokens + [[]],
        ]
        for tokens in invalid:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                CharTokenizer(tokens)

    def test_load_rejects_invalid_schema_versions_and_tokens(self):
        valid = {"version": 1, "type": "CharTokenizer", "tokens": self.tokens}
        invalid = [None, [], "text", {},
                   {"tokens": self.tokens},
                   dict(valid, extra=True)]
        for version in (0, 2, "1", "1.0", 1.0, True, None, [], {}):
            invalid.append(dict(valid, version=version))
        for kind in ("OtherTokenizer", "chartokenizer", "", None, 1, []):
            invalid.append(dict(valid, type=kind))
        for tokens in (None, "春江", {}, [], self.tokens + ["春"],
                       self.tokens + [""], self.tokens + ["春江"],
                       self.tokens + [False], self.tokens + [{}],
                       ["<EOS>", "<BOS>", "<UNK>"]):
            invalid.append(dict(valid, tokens=tokens))
        for payload in invalid:
            with self.subTest(payload=payload):
                self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                with patch.object(CharTokenizer, "build", side_effect=AssertionError("no rebuilding")):
                    with self.assertRaises(ValueError):
                        CharTokenizer.load(self.path)

    def test_load_rejects_malformed_and_ambiguous_json(self):
        invalid = ["", "{", "not json",
                   '{"version":1,"version":1,"type":"CharTokenizer","tokens":[]}',
                   '{"version":NaN,"type":"CharTokenizer","tokens":[]}',
                   '{"version":Infinity,"type":"CharTokenizer","tokens":[]}']
        for raw in invalid:
            with self.subTest(raw=raw):
                self.path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    CharTokenizer.load(self.path)
        self.path.write_bytes(b"\xff\xfe")
        with self.assertRaises(UnicodeDecodeError):
            CharTokenizer.load(self.path)

    def test_save_atomically_replaces_existing_file(self):
        self.path.write_text("old contents", encoding="utf-8")
        original_replace = os.replace

        def checked_replace(source, destination):
            self.assertEqual(Path(source).parent, self.path.parent)
            self.assertNotEqual(Path(source), self.path)
            self.assertEqual(Path(destination), self.path)
            self.assertEqual(self.path.read_text(encoding="utf-8"), "old contents")
            self.assertEqual(json.loads(Path(source).read_text(encoding="utf-8"))["tokens"],
                             self.tokens)
            original_replace(source, destination)

        with patch.object(tokenizer_module.os, "replace", side_effect=checked_replace) as replace:
            self.tokenizer.save(self.path)
        replace.assert_called_once()
        self.assertEqual(CharTokenizer.load(self.path).tokens, self.tokens)
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_replace_failure_preserves_old_file_and_cleans_temporary(self):
        self.path.write_bytes(b"old vocabulary")
        with patch.object(tokenizer_module.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                self.tokenizer.save(self.path)
        self.assertEqual(self.path.read_bytes(), b"old vocabulary")
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_write_failure_preserves_old_file_and_cleans_temporary(self):
        self.path.write_bytes(b"old vocabulary")

        def fail_after_partial_write(payload, stream, **kwargs):
            stream.write("partial JSON")
            raise OSError("write failed")

        with patch.object(tokenizer_module.json, "dump", side_effect=fail_after_partial_write):
            with self.assertRaisesRegex(OSError, "write failed"):
                self.tokenizer.save(self.path)
        self.assertEqual(self.path.read_bytes(), b"old vocabulary")
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_fsync_failure_cleans_temporary_without_creating_destination(self):
        with patch.object(tokenizer_module.os, "fsync", side_effect=OSError("fsync failed")):
            with self.assertRaisesRegex(OSError, "fsync failed"):
                self.tokenizer.save(self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_mutated_invalid_vocabulary_is_not_saved(self):
        self.tokenizer.tokens.append("春")
        with self.assertRaises(ValueError):
            self.tokenizer.save(self.path)
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
