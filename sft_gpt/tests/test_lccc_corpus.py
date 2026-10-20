"""Offline regression tests for LCCC streaming, conversion and default isolation."""
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sft_gpt.corpus import download as corpus
from sft_gpt.config import DEFAULT_TRAIN_DATA, DEFAULT_EVAL_DATA, DEFAULT_OUTPUT
from sft_gpt.data import load_train_eval_texts, make_pretrain_sequences
from sft_gpt.tokenization import HybridTokenizer


class LcccCorpusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"

    def prepare(self, source, **kwargs):
        options = dict(output_dir=self.output, target_docs=20, min_doc_len=2,
                       max_docs_per_file=3, eval_ratio=0.25)
        options.update(kwargs)
        return corpus.build_corpus(source, **options)

    def texts(self, files):
        return [json.loads(line)["text"] for path in files
                for line in path.read_text(encoding="utf-8").splitlines()]

    def write_gzip(self, rows, name="source.jsonl.gz"):
        path = self.root / name
        path.write_bytes(gzip.compress("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                              for row in rows).encode("utf-8")))
        return path

    def test_defaults_are_lccc_in_original_corpus_directory(self):
        self.assertEqual(DEFAULT_TRAIN_DATA.parent, corpus.TARGET_DIR)
        self.assertEqual(DEFAULT_EVAL_DATA, corpus.TARGET_DIR / "eval")
        self.assertEqual(DEFAULT_TRAIN_DATA, corpus.TARGET_DIR / "train")
        self.assertEqual(DEFAULT_OUTPUT.parts[-2:], ("pre_train", "lccc"))
        self.assertIn("lccc_base_train.jsonl.gz", corpus.URL)
        self.assertEqual(corpus.MIN_DOC_LEN, 2)
        self.assertEqual(corpus.TARGET_DIR.name, "corpus")
        self.assertEqual(corpus.TARGET_DOCS, 52000)
        self.assertEqual(corpus.EVAL_RATIO, 2000 / 52000)

    def test_character_tokenization_preserves_turn_order_without_role_labels(self):
        turns = ["你 好 ， 今 天 怎 么 样 ？", "挺 好 的", "h e l l o", "Hello world !"]
        self.assertEqual(corpus._clean_text(turns, 2),
                         "你好，今天怎么样？\n挺好的\nh e l l o\nHello world!")
        self.assertEqual(corpus._clean_text({"dialog": turns}, 2), corpus._clean_text(turns, 2))
        self.assertEqual(corpus._clean_text(["你好", "今天很好"], 2),
                         "你好\n今天很好")

    def test_character_mode_is_explicit_and_preserves_default_ascii_boundaries(self):
        turns = ["h e l l o", "1 2 3"]
        self.assertEqual(corpus._clean_text(turns, 2), "h e l l o\n1 2 3")
        self.assertEqual(corpus._clean_text(turns, 2, "character"), "hello\n123")
        source = self.write_gzip([turns, ["hello", "123"]])
        result = self.prepare(source)
        self.assertEqual(result.total_docs, 2)  # Distinct numbers/words must not collide.
        with mock.patch.object(corpus.urllib.request, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "tokenization"):
                self.prepare(None, tokenization="guess")
            download.assert_not_called()

    def test_word_segmented_chinese_and_english_spacing(self):
        self.assertEqual(corpus._clean_text(["我 今天 想 吃 火锅 。", "Hello world ， 吃完 打 game 吧 20 元"], 2),
                         "我今天想吃火锅。\nHello world，吃完打game吧20元")

    def test_invalid_turn_rejects_whole_dialogue_without_bridging_context(self):
        for turns in ([], ["一句话"], ["你好", ""], ["你好", None], ["你好", "😀!?"],
                      ["你好", "Русский", "继续"], {"dialog": "不是列表"}):
            with self.subTest(turns=turns):
                self.assertIsNone(corpus._clean_text(turns, 2))
        self.assertIsNone(corpus._clean_text(["甲", "乙"], 3))  # No role-label length inflation.
        self.assertEqual(corpus._clean_text(["甲", "乙"], 2), "甲\n乙")

    def test_gzip_jsonl_clean_deduplicate_split_and_manifest(self):
        rows = [[f"你 好 {i}", f"回 答 {i}"] for i in range(12)]
        rows += [{"dialog": ["你好0", "回答0"]}, ["无效", "😀"], ["只有一句"]]
        source = self.write_gzip(rows)
        original = source.read_bytes()
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("network")):
            result = self.prepare(source, chunk_size=1)
        self.assertEqual((result.total_docs, result.train_docs, result.eval_docs), (12, 9, 3))
        train, evaluation = self.texts(result.train_files), self.texts(result.eval_files)
        self.assertFalse(set(train) & set(evaluation))
        self.assertEqual(len(set(train + evaluation)), 12)
        self.assertTrue(all("\n" in text and "user:" not in text and "assistant:" not in text
                            for text in train + evaluation))
        self.assertEqual(source.read_bytes(), original)
        metadata = json.loads((self.output / corpus.MANIFEST).read_text(encoding="utf-8"))["preparation"]
        self.assertEqual(metadata["total_docs"], 12)
        self.assertEqual(metadata["source"], str(source.resolve()))

    def test_network_gzip_stream_closes_response_and_honors_cap(self):
        payload = gzip.compress("".join(json.dumps([f"问题{i}", f"回答{i}"]) + "\n"
                                         for i in range(10)).encode())
        response = io.BytesIO(payload)
        with mock.patch.object(corpus.urllib.request, "urlopen", return_value=response) as download:
            result = self.prepare(None, target_docs=4, chunk_size=1)
        self.assertEqual(result.total_docs, 4)
        self.assertTrue(response.closed)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(set(self.texts(result.train_files + result.eval_files)),
                         {f"问题{i}\n回答{i}" for i in range(4)})

    def test_truncated_gzip_preserves_existing_published_generation(self):
        source = self.write_gzip([[f"问题{i}", f"回答{i}"] for i in range(8)])
        self.prepare(source)
        before = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        source.write_bytes(source.read_bytes()[:-6])
        with self.assertRaises((EOFError, OSError)):
            self.prepare(source)
        self.assertEqual({p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}, before)

    def test_json_array_of_dialogues_and_compressed_directory_input(self):
        source = self.root / "inputs"
        source.mkdir()
        (source / "a.json.gz").write_bytes(gzip.compress(json.dumps(
            [["你好", "你好呀"], ["晚安", "好梦"]], ensure_ascii=False).encode()))
        self.write_gzip([["早安", "早上好"]], name="inputs/b.jsonl.gz")
        result = self.prepare(source)
        self.assertEqual(result.total_docs, 3)

    def test_single_json_dialogue_uses_explicit_dialog_wrapper(self):
        source = self.root / "single.json"
        source.write_text(json.dumps({"dialog": ["你好", "你好呀"]}), encoding="utf-8")
        with source.open("rb") as stream:
            records = list(corpus.iter_json_records(stream, chunk_size=1))
        self.assertEqual(len(records), 1)
        self.assertEqual(corpus._clean_text(records[0], 2), "你好\n你好呀")

    def test_lccc_rows_load_tokenize_and_pack_in_existing_training_pipeline(self):
        source = self.write_gzip([[f"问题{i}怎么样", f"回答{i}很好"] for i in range(12)])
        self.prepare(source)
        train, evaluation = load_train_eval_texts(self.output / "train", self.output / "eval")
        tokenizer = HybridTokenizer.build(train, min_freq=1, max_vocab_size=512)
        for texts in (train, evaluation):
            sequences = make_pretrain_sequences(texts, tokenizer, seq_len=32)
            self.assertEqual(sequences.shape[1], 33)
            self.assertGreater(sequences.shape[0], 0)
            self.assertTrue(all(tokenizer.decode(tokenizer.encode(text)) == text for text in texts))
        self.assertFalse(set(train) & set(evaluation))

    def test_old_manifest_can_be_migrated_without_leaving_owned_wiki_shards(self):
        (self.output / "train").mkdir(parents=True)
        old = self.output / "train" / "wiki_zh_part_001.jsonl"
        old.write_text('{"text":"旧Wiki"}\n', encoding="utf-8")
        (self.output / corpus.MANIFEST).write_text(json.dumps({
            "version": 1, "files": ["train/wiki_zh_part_001.jsonl"]}), encoding="utf-8")
        source = self.write_gzip([["你好", "你好呀"], ["晚安", "好梦"]])
        result = self.prepare(source)
        self.assertFalse(old.exists())
        self.assertTrue(all(path.name.startswith("lccc_part_") for path in result.train_files + result.eval_files))
