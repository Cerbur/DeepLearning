"""Lossless hybrid tokenization, strict specs and constrained UTF-8 generation."""

import codecs
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from sft_gpt.config import DEFAULT_TEXT_VOCAB
from sft_gpt.tokenization import (
    CharTokenizer, HybridTokenizer, load_tokenizer, tokenizer_from_dict, tokenizer_spec,
)
from sft_gpt.tokenization import hybrid_tokenizer as hybrid_module
from sft_gpt.tokenization import registry, serialization


class TestHybridTokenizer(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "nested" / "vocab.json"
        self.tokenizer = HybridTokenizer.build(["你好你好，世界！", "你好 😀😀"])

    @staticmethod
    def byte_ids(*values):
        return [4 + value for value in values]

    def test_fixed_ids_and_complete_byte_alphabet(self):
        tokenizer = self.tokenizer
        self.assertEqual((tokenizer.bos, tokenizer.eos, tokenizer.unk, tokenizer.pad),
                         (0, 1, 2, 3))
        self.assertEqual(tokenizer.tokens[:4], ["<BOS>", "<EOS>", "<UNK>", "<PAD>"])
        self.assertEqual(tokenizer.tokens[4:260], [f"<0x{b:02X}>" for b in range(256)])
        self.assertEqual(tokenizer.token_ids,
                         {token: i for i, token in enumerate(tokenizer.tokens)})
        self.assertEqual(HybridTokenizer.build([]).tokens, tokenizer.tokens[:260])

    def test_build_is_train_only_frequency_selected_and_unicode_sorted(self):
        texts = ["甲甲乙乙丙丙丁丁丁丁", "ASCII ASCII 稀"]
        first = HybridTokenizer.build(iter(texts), max_vocab_size=262)
        second = HybridTokenizer.build(reversed(texts), max_vocab_size=262)
        tied = min("甲乙丙")
        self.assertEqual(first.tokens[260:], sorted(["丁", tied]))
        self.assertEqual(first.tokens, second.tokens)
        self.assertNotIn("稀", first.token_ids)
        self.assertNotIn("驗", first.token_ids)
        self.assertEqual(first.decode(first.encode("驗稀")), "驗稀")
        self.assertEqual(HybridTokenizer.build(texts, max_vocab_size=260).tokens,
                         first.tokens[:260])
        self.assertEqual(HybridTokenizer.build(texts, max_vocab_size=None).tokens[260:],
                         sorted("甲乙丙丁"))
        self.assertEqual(HybridTokenizer.build(["稀"], min_freq=1).tokens[260:], ["稀"])

    def test_default_max_vocab_size_includes_base_items(self):
        chars = "".join(chr(i) for i in range(128, 128 + 8200))
        tokenizer = HybridTokenizer.build([chars, chars])
        self.assertEqual(len(tokenizer.tokens), 8192)
        self.assertEqual(tokenizer.tokens[260:], list(chars[:8192 - 260]))

    def test_ascii_always_uses_byte_ids(self):
        text = "".join(chr(i) for i in range(128))
        tokenizer = HybridTokenizer.build([text * 10], min_freq=1)
        self.assertEqual(len(tokenizer.tokens), 260)
        self.assertEqual(tokenizer.encode(text, special_tokens=False), list(range(4, 132)))
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_roundtrip_preserves_whitespace_multilingual_emoji_and_normalization(self):
        samples = [
            "", " \t\r\n  Hello, 世界!\n", "你好 😀🧑🏽‍💻🚀 未见字符𠀀",
            "é e\u0301 Å A\u030a ﬁ ffi", "العربية русский 日本語 한국어",
            "\x00\x01\x7f\u0080\u07ff\u0800\ud7ff\ue000\uffff\U0010ffff",
        ]
        for text in samples:
            for special in (True, False):
                with self.subTest(text=text, special=special):
                    ids = self.tokenizer.encode(text, special_tokens=special)
                    self.assertNotIn(self.tokenizer.unk, ids)
                    self.assertNotIn(self.tokenizer.pad, ids)
                    self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertNotEqual(self.tokenizer.encode("é"), self.tokenizer.encode("e\u0301"))

    def test_random_unicode_roundtrips(self):
        rng = random.Random(71)
        for _ in range(100):
            points = [rng.randrange(0x110000) for _ in range(30)]
            text = "".join(chr(p) for p in points if not 0xD800 <= p <= 0xDFFF)
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_learned_characters_precede_fallback(self):
        self.assertEqual(self.tokenizer.encode("你", special_tokens=False),
                         [self.tokenizer.token_ids["你"]])
        self.assertEqual(self.tokenizer.encode("陌", special_tokens=False),
                         [4 + b for b in "陌".encode("utf-8")])
        ids = self.tokenizer.encode("陌你😀陌")
        self.assertEqual(self.tokenizer.decode(ids), "陌你😀陌")

    def test_literal_special_spellings_do_not_inject_control_ids(self):
        text = "<BOS><EOS><UNK><PAD><0xFF>"
        ids = self.tokenizer.encode(text, special_tokens=False)
        self.assertTrue(all(4 <= token_id < 132 for token_id in ids))
        self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertEqual(self.tokenizer.encode(""), [0, 1])
        self.assertEqual(self.tokenizer.encode("", special_tokens=False), [])

    def test_decode_skips_controls_prints_unk_and_replaces_malformed_bytes(self):
        self.assertEqual(self.tokenizer.decode([0, 3, 2, 1]), "<UNK>")
        self.assertEqual(self.tokenizer.decode(self.byte_ids(0xF0, 0x9F, 0x98, 0x80)), "😀")
        self.assertEqual(self.tokenizer.decode(self.byte_ids(0xFF, 0xC2)), "��")
        learned = self.tokenizer.token_ids["你"]
        self.assertEqual(self.tokenizer.decode(self.byte_ids(0xC2) + [learned]), "�你")
        for value in (-1, len(self.tokenizer.tokens), True, 1.2, "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.tokenizer.decode([value])

    def test_constructor_and_specs_copy_token_lists(self):
        tokens = self.tokenizer.tokens[:260] + ["😀", "你"]
        tokenizer = HybridTokenizer(tokens)
        tokens.append("好")
        self.assertNotIn("好", tokenizer.token_ids)
        spec = tokenizer.to_dict()
        restored = HybridTokenizer.from_dict(spec)
        self.assertEqual(restored.tokens[260:], ["😀", "你"])
        spec["tokens"].append("好")
        self.assertNotIn("好", restored.token_ids)
        self.assertNotIn("好", tokenizer.token_ids)

    def test_save_load_exact_complete_spec_without_rebuild(self):
        tokenizer = HybridTokenizer(self.tokenizer.tokens[:260] + ["😀", "你"])
        self.assertEqual(tokenizer.save(self.path), self.path)
        self.assertEqual(serialization.load_json(self.path), tokenizer.to_dict())
        self.assertIn("你".encode("utf-8"), self.path.read_bytes())
        with patch.object(HybridTokenizer, "build", side_effect=AssertionError("no rebuild")):
            loaded = HybridTokenizer.load(self.path)
        self.assertEqual(loaded.tokens, tokenizer.tokens)
        self.assertEqual(loaded.token_ids, tokenizer.token_ids)
        self.assertEqual(loaded.encode("你好!"), tokenizer.encode("你好!"))

    def test_default_text_vocab_path(self):
        self.assertEqual(hybrid_module.DEFAULT_TEXT_VOCAB, DEFAULT_TEXT_VOCAB)
        with patch.object(hybrid_module, "DEFAULT_TEXT_VOCAB", self.path):
            self.assertEqual(self.tokenizer.save(), self.path)
            self.assertEqual(HybridTokenizer.load().tokens, self.tokenizer.tokens)
        with patch.object(registry, "DEFAULT_TEXT_VOCAB", self.path):
            self.assertIsInstance(load_tokenizer(), HybridTokenizer)

    def test_load_missing_is_error_and_never_rebuilds(self):
        for loader in (HybridTokenizer.load, load_tokenizer):
            with self.subTest(loader=loader), self.assertRaises(FileNotFoundError):
                loader(self.path)

    def test_reject_invalid_envelopes(self):
        valid = self.tokenizer.to_dict()
        invalid = [None, [], {}, {**valid, "extra": 1},
                   {key: value for key, value in valid.items() if key != "version"}]
        invalid += [{**valid, "version": value} for value in (True, 1.0, "1", 2, None)]
        invalid += [{**valid, "type": value} for value in (None, 1, [], "CharTokenizer")]
        invalid += [{**valid, "tokens": value} for value in (None, {}, "abc", tuple(valid["tokens"]))]
        for payload in invalid:
            with self.subTest(payload=repr(payload)[:100]), self.assertRaises(ValueError):
                HybridTokenizer.from_dict(payload)

    def test_reject_invalid_token_tables(self):
        base = self.tokenizer.tokens[:260]
        invalid = [[], base[:-1], base + ["a"], base + ["ab"], base + [""],
                   base + ["你", "你"], base + ["\ud800"], base + ["\udfff"], base + [None]]
        for index in (0, 3, 4, 100, 259):
            changed = base[:]
            changed[index] = "wrong"
            invalid.append(changed)
        swapped = base[:]
        swapped[4], swapped[5] = swapped[5], swapped[4]
        invalid.append(swapped)
        for tokens in invalid:
            with self.subTest(tokens=repr(tokens)[-100:]), self.assertRaises(ValueError):
                HybridTokenizer.from_dict({"version": 1, "type": "HybridTokenizer", "tokens": tokens})

    def test_reject_invalid_build_arguments_and_non_scalar_text(self):
        for min_freq in (0, -1, True, 1.0, "2"):
            with self.subTest(min_freq=min_freq), self.assertRaises(ValueError):
                HybridTokenizer.build([], min_freq=min_freq)
        for maximum in (259, -1, True, 8192.0, "8192"):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                HybridTokenizer.build([], max_vocab_size=maximum)
        with self.assertRaises(TypeError):
            HybridTokenizer.build([123])
        with self.assertRaises(TypeError):
            self.tokenizer.encode(123)
        with self.assertRaises(UnicodeEncodeError):
            self.tokenizer.encode("\ud800")
        with self.assertRaises(UnicodeEncodeError):
            HybridTokenizer.build(["\ud800"])

    def test_registry_supports_legacy_char_and_hybrid_specs(self):
        for tokenizer in (self.tokenizer, CharTokenizer(["<BOS>", "<EOS>", "<UNK>", "春", "江"])):
            with self.subTest(kind=type(tokenizer).__name__):
                spec = tokenizer_spec(tokenizer)
                restored = tokenizer_from_dict(spec)
                self.assertIs(type(restored), type(tokenizer))
                self.assertEqual(restored.tokens, tokenizer.tokens)
                tokenizer.save(self.path)
                self.assertEqual(load_tokenizer(self.path).tokens, tokenizer.tokens)
                spec["tokens"].append("新")
                self.assertNotIn("新", tokenizer.token_ids)
        with self.assertRaises(TypeError):
            tokenizer_spec(object())
        for spec in ({"type": "Other"}, {"type": []}, None,
                     {"version": True, "type": "CharTokenizer", "tokens": ["<BOS>", "<EOS>", "<UNK>"]},
                     {"version": 1, "type": "CharTokenizer", "tokens": ["<BOS>", "<EOS>", "<UNK>", "春春"]}):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                tokenizer_from_dict(spec)

    def test_strict_json_rejects_duplicate_fields_and_nonfinite_numbers(self):
        self.path.parent.mkdir(parents=True)
        for text in ('{"version":1,"version":1}', '{"nested":{"x":1,"x":2}}',
                     '{"number":NaN}', '{"number":Infinity}', '{"number":-Infinity}',
                     '{"number":1e999}', '{not json'):
            self.path.write_text(text, encoding="utf-8")
            for loader in (serialization.load_json, HybridTokenizer.load, load_tokenizer):
                with self.subTest(text=text, loader=loader), self.assertRaises(ValueError):
                    loader(self.path)

    def test_atomic_save_preserves_destination_and_cleans_temporary_on_failure(self):
        self.tokenizer.save(self.path)
        before = self.path.read_bytes()
        for failure in ("replace", "dump", "fsync"):
            target = {"replace": "os.replace", "dump": "json.dump", "fsync": "os.fsync"}[failure]
            with patch("sft_gpt.tokenization.serialization." + target,
                       side_effect=OSError("simulated " + failure)):
                with self.assertRaises(OSError):
                    self.tokenizer.save(self.path)
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        with self.assertRaises(ValueError):
            serialization.save_json_atomic(self.path, {"number": float("nan")})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_valid_next_at_character_boundary(self):
        tokenizer = self.tokenizer
        allowed = tokenizer.valid_next_ids([tokenizer.bos])
        self.assertEqual(allowed, sorted(set(allowed)))
        self.assertEqual(allowed, tokenizer.valid_next_ids([]))
        self.assertIn(tokenizer.eos, allowed)
        for token_id in (tokenizer.bos, tokenizer.unk, tokenizer.pad):
            self.assertNotIn(token_id, allowed)
        for byte in range(256):
            self.assertEqual(4 + byte in allowed, byte < 128 or 0xC2 <= byte <= 0xF4)
        for token_id in range(260, len(tokenizer.tokens)):
            self.assertIn(token_id, allowed)
        self.assertEqual(tokenizer.valid_next_ids([0, 1]), [])

    def test_utf8_special_continuation_ranges_and_pending_count(self):
        for lead, low, high, count in ((0xC2, 0x80, 0xBF, 1), (0xE0, 0xA0, 0xBF, 2),
                                       (0xED, 0x80, 0x9F, 2), (0xF0, 0x90, 0xBF, 3),
                                       (0xF4, 0x80, 0x8F, 3)):
            prefix = [0] + self.byte_ids(lead)
            with self.subTest(lead=lead):
                self.assertEqual(self.tokenizer.valid_next_ids(prefix),
                                 self.byte_ids(*range(low, high + 1)))
                self.assertEqual(self.tokenizer.pending_byte_count(prefix), count)
                prefix += self.byte_ids(low)
                if count > 1:
                    self.assertEqual(self.tokenizer.valid_next_ids(prefix),
                                     self.byte_ids(*range(0x80, 0xC0)))
                self.assertEqual(self.tokenizer.pending_byte_count(prefix), count - 1)
        self.assertEqual(self.tokenizer.pending_byte_count(self.tokenizer.encode("陌😀")), 0)

    def test_all_encoded_prefixes_are_allowed_and_end_on_character_boundary(self):
        for text in ("你好陌😀🧑🏽‍💻", "é\n\t<BOS>", "\u0800\ud7ff\U0010ffff"):
            ids = self.tokenizer.encode(text)
            prefix = [ids[0]]
            for token_id in ids[1:]:
                self.assertIn(token_id, self.tokenizer.valid_next_ids(prefix))
                prefix.append(token_id)
            self.assertEqual(self.tokenizer.valid_next_ids(prefix), [])
            self.assertEqual(self.tokenizer.pending_byte_count(prefix), 0)

    def test_utf8_mask_matches_python_strict_incremental_decoder(self):
        prefixes = [b"", b"A", b"\xc2", b"\xdf", b"\xe0", b"\xe0\xa0", b"\xed",
                    b"\xef", b"\xf0", b"\xf0\x90", b"\xf0\x90\x80", b"\xf4", b"\xf4\x8f"]
        for prefix in prefixes:
            allowed = set(self.tokenizer.valid_next_ids([0] + [4 + b for b in prefix]))
            for byte in range(256):
                decoder = codecs.getincrementaldecoder("utf-8")("strict")
                try:
                    candidate = prefix + bytes([byte])
                    decoder.decode(candidate, final=False)
                    pending = decoder.getstate()[0]
                    valid = not pending
                    if pending:
                        # CPython can delay rejecting ED A0 until byte 3. Check
                        # that an actual strict completion exists, not just that
                        # the decoder has buffered a not-yet-rejected prefix.
                        width = 2 if pending[0] < 0xE0 else 3 if pending[0] < 0xF0 else 4
                        missing = width - len(pending)
                        for fill in (0x80, 0xBF):
                            try:
                                (candidate + bytes([fill]) * missing).decode("utf-8", "strict")
                                valid = True
                            except UnicodeDecodeError:
                                pass
                except UnicodeDecodeError:
                    valid = False
                with self.subTest(prefix=prefix, byte=byte):
                    self.assertEqual(4 + byte in allowed, valid)

    def test_invalid_generation_prefixes_raise(self):
        invalid = [[2], [3], [0, 0], [1, 4], self.byte_ids(0x80), self.byte_ids(0xC0),
                   self.byte_ids(0xF5), self.byte_ids(0xE0, 0x80), self.byte_ids(0xED, 0xA0),
                   self.byte_ids(0xF0, 0x80), self.byte_ids(0xF4, 0x90),
                   self.byte_ids(0xC2) + [1], self.byte_ids(0xC2) + [260], [-1]]
        for prefix in invalid:
            for method in (self.tokenizer.valid_next_ids, self.tokenizer.pending_byte_count):
                with self.subTest(prefix=prefix, method=method), self.assertRaises(ValueError):
                    method(prefix)


if __name__ == "__main__":
    unittest.main()
