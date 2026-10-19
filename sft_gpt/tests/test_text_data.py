"""General text loading and split-local next-token packing contracts."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from sft_gpt.config import MAX_SEQ_LEN
from sft_gpt.data import load_texts, make_pretrain_sequences
from sft_gpt.tokenization import CharTokenizer


class TextDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write_text(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return path

    def write_json(self, name, payload):
        return self.write_text(name, json.dumps(payload, ensure_ascii=False))

    def write_table(self, name, field, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
            writer.writerow([field, "metadata"])
            writer.writerows((text, "ignored") for text in rows)
        return path

    def labels(self, sequences):
        return [token for row in sequences.tolist() for token in row[1:] if token != -100]

    def test_txt_preserves_nonblank_lines_and_original_line_endings(self):
        path = self.write_text("raw.txt", "\ufeff\r\n  first  \r\n\tsecond \r\n \t\r\n\r\n third \t")
        self.assertEqual(load_texts(path), ["  first  \r\n\tsecond \r\n", " third \t"])

    def test_mixed_recursive_formats_are_sorted_and_globally_deduplicated(self):
        # Deliberately create files in reverse order; no per-file minimum of two.
        self.write_json("z/single.json", {"text": "last"})
        self.write_json("c/list.json", [{"text": "csv value"}, {"text": "json list"}])
        self.write_text("b/rows.jsonl", '\n{"text":"jsonl"}\n{"text":"alpha\\n"}\n')
        self.write_table("a/values.tsv", "text", ["tsv value", "   "])
        self.write_table("a/values.csv", "text", ["csv value", ""])
        self.write_text("a/0.TXT", "alpha\n\nbeta")
        self.write_text("ignored.md", "not a supported document")
        self.assertEqual(load_texts(self.root),
                         ["alpha\n", "beta", "csv value", "tsv value", "jsonl", "json list", "last"])

    def test_csv_and_tsv_configurable_text_column_preserve_whitespace(self):
        expected = ['  non-poetry, text  \nsecond line\t', "other 文本"]
        for suffix in (".csv", ".tsv"):
            with self.subTest(suffix=suffix):
                path = self.write_table("custom" + suffix, "body", expected)
                self.assertEqual(load_texts(path, text_field="body"), expected)
                with self.assertRaisesRegex(ValueError, "custom.*:1:"):
                    load_texts(path)

    def test_json_configurable_text_field_and_single_record(self):
        self.write_json("a.json", {"body": "  one \n"})
        self.write_json("b.json", [{"body": "two"}, {"body": "  one \n"}])
        self.write_text("c.jsonl", '{"body":"three"}\n')
        self.assertEqual(load_texts(self.root, text_field="body"), ["  one \n", "two", "three"])

    def test_conversations_are_one_verbatim_document_with_fixed_template(self):
        messages = [{"role": "system", "content": "  rules \nnext "},
                    {"role": "user", "content": "hello\n\n world"},
                    {"role": "assistant", "content": " answer "}]
        expected = "system:   rules \nnext \nuser: hello\n\n world\nassistant:  answer "
        path = self.write_json("chat.json", [{"messages": messages}, {"text": "plain"}])
        self.assertEqual(load_texts(path), [expected, "plain"])
        self.write_text("chat.jsonl", json.dumps({"messages": messages}) + '\n{"text":"plain"}\n')
        self.assertEqual(load_texts(self.root), [expected, "plain"])

    def test_bad_conversation_and_record_schemas_report_path_and_line(self):
        invalid = [None, "string", [], {}, {"text": None}, {"text": 3},
                   {"messages": None}, {"messages": {}}, {"messages": ["user"]},
                   {"messages": [{"content": "hello"}]},
                   {"messages": [{"role": "tool", "content": "hello"}]},
                   {"messages": [{"role": 1, "content": "hello"}]},
                   {"messages": [{"role": "user"}]},
                   {"messages": [{"role": "user", "content": ["hello"]}]}]
        for record in invalid:
            with self.subTest(record=record):
                path = self.write_text("bad.jsonl", '\n{"text":"valid"}\n' + json.dumps(record) + '\n')
                with self.assertRaises(ValueError) as raised:
                    load_texts(path)
                self.assertIn(str(path) + ":3:", str(raised.exception))

    def test_json_schema_error_reports_physical_record_line(self):
        path = self.write_text("bad.json", '[\n  {"text": "valid"},\n\n  {"messages": [{"role": "tool", "content": "x"}]}\n]')
        with self.assertRaises(ValueError) as raised:
            load_texts(path)
        self.assertIn(str(path) + ":4:", str(raised.exception))
        for payload in (None, 1, "text"):
            with self.subTest(payload=payload):
                self.write_json("bad.json", payload)
                with self.assertRaisesRegex(ValueError, "bad.json:1:"):
                    load_texts(path)

    def test_invalid_json_syntax_reports_source_line(self):
        for name, source, line in (("syntax.jsonl", '\n{"text": "ok"}\n{bad}\n', 3),
                                   ("syntax.json", '[\n{"text": "ok"},\n{bad}\n]', 3)):
            with self.subTest(name=name):
                path = self.write_text(name, source)
                with self.assertRaises(ValueError) as raised:
                    load_texts(path)
                self.assertIn(str(path) + f":{line}:", str(raised.exception))

    def test_missing_table_values_and_malformed_csv_are_not_silently_skipped(self):
        for source, line in (("other,text\nvalue\n", 2),
                             ('text\n"unterminated\n', 2)):
            with self.subTest(source=source):
                path = self.write_text("bad.csv", source)
                with self.assertRaises(ValueError) as raised:
                    load_texts(path)
                self.assertIn(str(path) + f":{line}:", str(raised.exception))

    def test_duplicate_table_headers_and_extra_columns_are_rejected(self):
        for suffix, delimiter in ((".csv", ","), (".tsv", "\t")):
            sources = [("text" + delimiter + "text\noriginal" + delimiter + "replaced\n", 1),
                       ("text\nhello" + delimiter + "world\nbye\n", 2)]
            for source, line in sources:
                with self.subTest(suffix=suffix, source=source):
                    path = self.write_text("ambiguous" + suffix, source)
                    with self.assertRaises(ValueError) as raised:
                        load_texts(path)
                    self.assertIn(str(path) + f":{line}:", str(raised.exception))

    def test_empty_documents_removed_and_only_full_text_duplicates_removed(self):
        path = self.write_json("texts.json", [{"text": ""}, {"text": " \n\t"},
                                              {"messages": []}, {"text": "same"},
                                              {"text": "same"}, {"text": " same "}])
        self.assertEqual(load_texts(path), ["same", " same "])
        self.write_json("texts.json", [{"text": "same"}, {"text": "same"}])
        with self.assertRaisesRegex(ValueError, "至少需要两个"):
            load_texts(path)
        self.write_text("empty.txt", "\n \t\n")
        with self.assertRaisesRegex(ValueError, "至少需要两个"):
            load_texts(self.root / "empty.txt")

    def test_unsupported_files_or_directory_raise_explicit_error(self):
        path = self.write_text("ignored.bin", "x")
        for target in (path, self.root):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "没有支持的文本文件"):
                    load_texts(target)
        with self.assertRaises(FileNotFoundError):
            load_texts(self.root / "missing.txt")
        for field in (None, 1, ""):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "text_field"):
                load_texts(self.root, text_field=field)

    def test_packing_predicts_every_stream_target_exactly_once(self):
        texts = ["abcdef" * 23, " 文本\nwith spaces ", "final"]
        tokenizer = CharTokenizer.build(texts, min_freq=1)
        stream = [token for text in texts for token in tokenizer.encode(text)]
        for length in (1, 2, 3, 7, 16, MAX_SEQ_LEN):
            with self.subTest(seq_len=length):
                sequences = make_pretrain_sequences(texts, tokenizer, seq_len=length)
                self.assertEqual(sequences.dtype, torch.long)
                self.assertEqual(sequences.shape[1], length + 1)
                self.assertEqual(self.labels(sequences), stream[1:])
                self.assertEqual(sequences.shape[0], (len(stream) - 2) // length + 1)
                for left, right in zip(sequences[:-1], sequences[1:]):
                    self.assertEqual(left[-1].item(), right[0].item())

    def test_packing_preserves_bos_eos_for_each_document(self):
        tokenizer = CharTokenizer.build(["a", "b"], min_freq=1)
        with mock.patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode:
            sequences = make_pretrain_sequences(["a", "b"], tokenizer, seq_len=8)
        self.assertEqual(encode.call_args_list,
                         [mock.call("a", special_tokens=True), mock.call("b", special_tokens=True)])
        self.assertEqual(sequences.tolist(),
                         [tokenizer.encode("a") + tokenizer.encode("b") + [-100] * 3])

    def test_tail_padding_and_one_token_tail(self):
        tokenizer = CharTokenizer.build(["abcd"], min_freq=1)
        sequences = make_pretrain_sequences(["abc"], tokenizer, seq_len=3)
        encoded = tokenizer.encode("abc")
        self.assertEqual(sequences.tolist(), [encoded[:4], encoded[3:] + [-100, -100]])
        self.assertTrue(torch.all(sequences[:, :-1].clamp_min(0) >= 0))
        # Exactly seq_len+1 tokens: the overlapping final token is not a new row.
        exact = make_pretrain_sequences(["ab"], tokenizer, seq_len=3)
        self.assertEqual(exact.tolist(), [tokenizer.encode("ab")])
        self.assertEqual(make_pretrain_sequences(["a"], tokenizer, seq_len=MAX_SEQ_LEN).shape,
                         (1, MAX_SEQ_LEN + 1))

    def test_padding_targets_use_cross_entropy_ignore_index(self):
        tokenizer = CharTokenizer.build(["abc"], min_freq=1)
        sequences = make_pretrain_sequences(["abc"], tokenizer, seq_len=3)
        targets = sequences[:, 1:].reshape(-1)
        logits = torch.arange(targets.numel() * len(tokenizer.tokens), dtype=torch.float32)
        logits = logits.reshape(targets.numel(), -1) / 10
        padded_loss = torch.nn.functional.cross_entropy(logits, targets)
        keep = targets != -100
        expected_loss = torch.nn.functional.cross_entropy(logits[keep], targets[keep])
        torch.testing.assert_close(padded_loss, expected_loss)

    def test_split_then_packing_never_includes_validation_document(self):
        path = self.write_json("split.json", [{"text": "training only 甲"},
                                              {"text": "validation only 乙"},
                                              {"text": "training only 甲"}])
        texts = load_texts(path)
        tokenizer = CharTokenizer.build(texts, min_freq=1)
        train = make_pretrain_sequences(texts[:1], tokenizer, seq_len=4)
        valid = make_pretrain_sequences(texts[1:], tokenizer, seq_len=4)
        self.assertEqual(self.labels(train), tokenizer.encode(texts[0])[1:])
        self.assertEqual(self.labels(valid), tokenizer.encode(texts[1])[1:])
        self.assertNotIn(tokenizer.token_ids["乙"], train.tolist()[0])
        self.assertFalse(torch.any(train == tokenizer.token_ids["乙"]).item())
        self.assertFalse(torch.any(valid == tokenizer.token_ids["甲"]).item())

    def test_seq_len_rejects_invalid_type_and_range(self):
        tokenizer = CharTokenizer.build(["x"], min_freq=1)
        for length in (0, -1, MAX_SEQ_LEN + 1, True, False, 1.0, "1", None):
            with self.subTest(seq_len=length), self.assertRaisesRegex(ValueError, "seq_len"):
                make_pretrain_sequences(["x"], tokenizer, seq_len=length)
        self.assertEqual(make_pretrain_sequences(["x"], tokenizer, seq_len=1).shape, (2, 2))
        self.assertEqual(make_pretrain_sequences(["x"], tokenizer).shape, (1, MAX_SEQ_LEN + 1))

    def test_empty_texts_or_insufficient_encoded_tokens_raise_value_error(self):
        tokenizer = CharTokenizer.build(["x"], min_freq=1)
        for texts in ([], [""], [" \n\t"], None, "x", [None]):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                make_pretrain_sequences(texts, tokenizer)
        for tokens in ([], [0]):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "足够 tokens"):
                make_pretrain_sequences(["x"], SimpleNamespace(encode=lambda *a, **kw: tokens))
        sequences = make_pretrain_sequences(iter(["x"]), tokenizer, seq_len=2)
        self.assertEqual(sequences.tolist(), [tokenizer.encode("x")])
