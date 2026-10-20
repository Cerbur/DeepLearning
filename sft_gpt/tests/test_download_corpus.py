"""Offline corpus preparation contracts; no torch or network is required.

Run from sft_gpt: python -m unittest discover -s tests -p test_download_corpus.py -v
"""
import contextlib
import importlib.util
import io
import inspect
import json
from pathlib import Path
import random
import runpy
import sys
import tempfile
import unittest
from unittest import mock

# Loading the standalone script avoids sft_gpt.__init__ (which imports torch).
SCRIPT = Path(__file__).resolve().parents[1] / "corpus" / "download.py"
SPEC = importlib.util.spec_from_file_location("_test_download_corpus", SCRIPT)
corpus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = corpus
with mock.patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(corpus)


class DownloadCorpusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"

    def source(self, records, name="input.jsonl"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
        else:
            content = json.dumps(records, ensure_ascii=False, indent=2)
        # Preserve ordinary Unicode bytes, but serialize lone-surrogate fixtures as escapes.
        path.write_bytes(content.encode("utf-8", errors="backslashreplace"))
        return path

    def prepare(self, input_path, **kwargs):
        options = dict(output_dir=self.output, target_docs=20, min_doc_len=1,
                       max_docs_per_file=3, eval_ratio=0.25)
        options.update(kwargs)
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("unexpected network")):
            return corpus.build_corpus(input_path, **options)

    def snapshot(self, directory=None):
        directory = self.output if directory is None else directory
        return {path.relative_to(directory).as_posix(): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()} if directory.exists() else {}

    def texts(self, paths):
        return [json.loads(line)["text"] for path in paths
                for line in path.read_text(encoding="utf-8").splitlines()]

    def test_import_is_silent_and_has_no_io(self):
        module = importlib.util.module_from_spec(SPEC)
        with mock.patch.object(corpus.urllib.request, "urlopen") as download, \
                mock.patch.object(Path, "mkdir") as mkdir, \
                mock.patch.object(Path, "open") as open_file, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            SPEC.loader.exec_module(module)
        download.assert_not_called()
        mkdir.assert_not_called()
        open_file.assert_not_called()
        self.assertEqual(stdout.getvalue(), "")

    def test_normalize_filter_deduplicate_whole_documents_and_cap(self):
        expected = {f"article {index}\n\nparagraph {index}" for index in range(12)}
        records = [{"completion": f"  article {index}\r\n\r\n\r\nparagraph {index}  "}
                   for index in range(12)]
        records += [{"text": next(iter(expected))}, None, [], "scalar", {}, {"text": 12},
                    {"text": None}, {"text": "   "}, {"text": "x"}, {"text": "\x00"},
                    {"text": "\x01"}, {"text": "\x7f"}, {"text": "\ud800"},
                    {"text": "Русский 한국어 かな 😀"}, {"text": "！？,."}]
        source = self.source(records)
        with source.open("a", encoding="utf-8") as stream:
            stream.write("not valid json\n\n")
        result = self.prepare(source, min_doc_len=2)
        self.assertEqual((result.total_docs, result.train_docs, result.eval_docs), (12, 9, 3))
        train, evaluation = self.texts(result.train_files), self.texts(result.eval_files)
        self.assertEqual(set(train) | set(evaluation), expected)
        self.assertFalse(set(train) & set(evaluation))
        self.assertEqual([len(self.texts([path])) for path in result.train_files], [3, 3, 3])
        self.assertTrue(all(set(json.loads(line)) == {"text"} for path in result.train_files
                            for line in path.read_text(encoding="utf-8").splitlines()))

    def test_shared_cleaner_filters_languages_before_min_length_and_deduplication(self):
        source = self.source([{"text": "中文😀English"}, {"completion": "中文РусскийEnglish"},
                              {"text": "中文 English"}, {"text": "第二篇中文"},
                              {"text": "Third English 123!"}, {"text": "Русский 한국어 かな 😀"},
                              {"text": "Greek Ελληνικά"}, {"text": "xРусский"}, {"text": "！？, ."}])
        result = self.prepare(source, eval_ratio=0.34, min_doc_len=2)
        self.assertEqual(result.total_docs, 4)
        self.assertEqual(set(self.texts(result.train_files + result.eval_files)),
                         {"中文 English", "第二篇中文", "Third English 123!", "Greek"})

    def test_seeded_determinism_and_different_seeds(self):
        source = self.source([{"text": f"document {index}"} for index in range(20)])
        first = self.prepare(source, seed=123)
        original = self.snapshot()
        self.prepare(source, seed=123)
        self.assertEqual(self.snapshot(), original)
        self.prepare(source, seed=124)
        self.assertNotEqual(self.snapshot(), original)
        self.assertEqual(first.total_docs, 20)

    def test_default_cap_reuses_100_root_shards_without_mutating_inputs(self):
        source_dir = self.root / "corpus"
        source_dir.mkdir()
        for index in range(100):
            self.source([{"text": f"article {index}-{part} " + "中文完整文档" * 12}
                         for part in range(2)], f"corpus/wiki_zh_part_{index + 1:03d}.jsonl")
        originals = self.snapshot(source_dir)
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("network")):
            result = corpus.build_corpus(source_dir, source_dir, min_doc_len=1)
        self.assertEqual((len(result.train_files), len(result.eval_files)), (1, 1))
        self.assertEqual((result.total_docs, result.train_docs, result.eval_docs), (200, 193, 7))
        self.assertEqual(len(list((source_dir / "train").glob("*.jsonl"))), 1)
        self.assertEqual(len(list((source_dir / "eval").glob("*.jsonl"))), 1)
        self.assertTrue(all(path.stat().st_size > 0 for path in result.train_files + result.eval_files))
        self.assertEqual({name: (source_dir / name).read_bytes() for name in originals}, originals)
        # Ignore pre-existing output records when reusing the same root again.
        before = self.snapshot(source_dir)
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("network")):
            corpus.build_corpus(source_dir, source_dir, min_doc_len=1)
        self.assertEqual(self.snapshot(source_dir), before)

    def test_target_is_a_limit_not_a_requirement_to_exhaust_source(self):
        source = self.source([{"text": f"article {index}"} for index in range(50)])
        result = self.prepare(source, target_docs=10)
        self.assertEqual(result.total_docs, 10)
        self.assertEqual(set(self.texts(result.train_files + result.eval_files)),
                         {f"article {index}" for index in range(10)})

    def test_json_arrays_single_records_and_recursive_sorted_directory(self):
        self.source([{"completion": f"array {index}"} for index in range(10)], "inputs/z/data.json")
        self.source({"text": "first single document"}, "inputs/a.json")
        (self.root / "inputs" / "notes.txt").write_text("ignored", encoding="utf-8")
        result = self.prepare(self.root / "inputs", target_docs=4)
        self.assertEqual(set(self.texts(result.train_files + result.eval_files)),
                         {"first single document", "array 0", "array 1", "array 2"})

    def test_incremental_utf8_at_every_small_chunk_boundary(self):
        records = [{"completion": "中文😀𠀀边界\n完整文档"}, {"text": "第二篇\"引号\\斜线"},
                   {"text": "第三篇", "nested": [{"value": [1, 2]}]}]
        encoded = ("\ufeff" + json.dumps(records, ensure_ascii=False)).encode("utf-8")
        for size in range(1, 12):
            with self.subTest(chunk_size=size):
                self.assertEqual(list(corpus.iter_json_records(io.BytesIO(encoded), size)), records)
        source = self.source(records)
        result = self.prepare(source, target_docs=3, eval_ratio=0.34, chunk_size=1)
        self.assertEqual(set(self.texts(result.train_files + result.eval_files)),
                         {corpus.clean_corpus_text(record.get("text", record.get("completion"))) for record in records})

    def test_stream_parser_handles_non_record_values_and_rejects_bad_structure(self):
        values = [None, True, False, 1.5, -12.5, 1e30, [], {"text": "valid"}]
        data = json.dumps(values).encode("utf-8")
        for size in range(1, 8):
            with self.subTest(size=size):
                self.assertEqual(list(corpus.iter_json_records(io.BytesIO(data), size)), values)
        for data in (b"", b"[", b"[1,]", b"[1 2]", b"{} {}", b"[1.2"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                list(corpus.iter_json_records(io.BytesIO(data), 1))

    def test_mocked_network_is_streamed_without_real_download(self):
        records = [{"completion": f"第{index}篇😀完整文档"} for index in range(12)]
        response = io.BytesIO(json.dumps(records, ensure_ascii=False).encode("utf-8"))
        with mock.patch.object(corpus.urllib.request, "urlopen", return_value=response) as download:
            result = corpus.build_corpus(output_dir=self.output, target_docs=12, min_doc_len=1,
                                         max_docs_per_file=3, eval_ratio=0.25, chunk_size=1,
                                         url="https://example.test/documents.json")
        self.assertEqual(result.total_docs, 12)
        self.assertEqual(download.call_count, 1)
        self.assertTrue(response.closed)

    def test_insufficient_unique_documents_preserve_existing_data(self):
        source = self.source([{"text": f"document {index}"} for index in range(12)])
        self.prepare(source)
        before = self.snapshot()
        insufficient = self.source([{"text": "same"}] * 30, "too-small.jsonl")
        with self.assertRaisesRegex(ValueError, "Not enough valid unique"):
            self.prepare(insufficient)
        self.assertEqual(self.snapshot(), before)
        with self.assertRaises(ValueError):
            self.prepare(insufficient, output_dir=self.root / "new-output")
        self.assertFalse((self.root / "new-output").exists())

    def test_bad_json_and_utf8_preserve_existing_data(self):
        source = self.source([{"text": f"document {index}"} for index in range(12)])
        self.prepare(source)
        before = self.snapshot()
        broken = self.root / "broken.json"
        for content in (b'[{"text":"one"},', b'[{"text":"bad\xff"}]', b'[{"text":"one"},]',
                        b'[{"text":"one"}] garbage', b'[{"text":"one"} {"text":"two"}]'):
            with self.subTest(content=content):
                broken.write_bytes(content)
                with self.assertRaises(ValueError):
                    self.prepare(broken, chunk_size=1)
                self.assertEqual(self.snapshot(), before)

    def test_transport_failure_preserves_existing_data(self):
        source = self.source([{"text": f"document {index}"} for index in range(12)])
        self.prepare(source)
        before = self.snapshot()
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=OSError("offline")):
            with self.assertRaisesRegex(OSError, "offline"):
                corpus.build_corpus(output_dir=self.output)
        self.assertEqual(self.snapshot(), before)

    def test_only_manifest_owned_obsolete_files_are_removed(self):
        source = self.source([{"text": f"document {index}"} for index in range(20)])
        previous = self.prepare(source, max_docs_per_file=3)
        user_files = [self.output / "train" / "notes.txt", self.output / "train" / "custom.jsonl",
                      self.output / "train" / "wiki_zh_part_999.jsonl", self.output / "eval" / "keep.json"]
        for path in user_files:
            path.write_bytes(b"user content")
        current = self.prepare(source, max_docs_per_file=10)
        self.assertEqual((len(previous.train_files), len(previous.eval_files)), (5, 2))
        self.assertEqual((len(current.train_files), len(current.eval_files)), (2, 1))
        obsolete = set(previous.train_files + previous.eval_files) - set(current.train_files + current.eval_files)
        self.assertTrue(obsolete)
        self.assertTrue(all(not path.exists() for path in obsolete))
        self.assertTrue(all(path.read_bytes() == b"user content" for path in user_files))
        manifest = json.loads((self.output / corpus.MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["files"]),
                         {path.relative_to(self.output).as_posix()
                          for path in current.train_files + current.eval_files})

    def test_unowned_name_collision_is_not_overwritten(self):
        source = self.source([{"text": f"document {index}"} for index in range(12)])
        collision = self.output / "train" / "lccc_part_001.jsonl"
        collision.parent.mkdir(parents=True)
        collision.write_bytes(b"user supplied data")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "unowned"):
            self.prepare(source)
        self.assertEqual(self.snapshot(), before)

    def test_publication_rename_failure_rolls_back_previous_generation(self):
        source = self.source([{"text": f"document {index}"} for index in range(12)])
        self.prepare(source)
        before = self.snapshot()
        replace = corpus.os.replace
        failed = False

        def fail_once(source, destination):
            nonlocal failed
            if "new" in Path(source).parts and not failed:
                failed = True
                raise OSError("simulated publication error")
            return replace(source, destination)

        with mock.patch.object(corpus.os, "replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "simulated publication error"):
                self.prepare(source, seed=99)
        self.assertTrue(failed)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(list(self.root.glob(".corpus-stage-*")))

    def test_invalid_manifest_cannot_delete_arbitrary_user_file(self):
        source = self.source([{"text": f"document {index}"} for index in range(12)])
        self.output.mkdir()
        user_file = self.root / "user.jsonl"
        user_file.write_bytes(b"precious")
        (self.output / corpus.MANIFEST).write_text(
            json.dumps({"version": 1, "files": ["../user.jsonl"]}), encoding="utf-8")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "ownership manifest"):
            self.prepare(source)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(user_file.read_bytes(), b"precious")

    def test_invalid_settings_fail_before_network(self):
        cases = [dict(target_docs=0), dict(target_docs=1), dict(max_docs_per_file=0),
                 dict(max_docs_per_file=-1), dict(eval_ratio=0), dict(eval_ratio=1),
                 dict(eval_ratio=float("nan")), dict(eval_ratio=float("inf")),
                 dict(min_doc_len=0), dict(chunk_size=0)]
        for options in cases:
            with self.subTest(options=options), \
                    mock.patch.object(corpus.urllib.request, "urlopen") as download:
                with self.assertRaises(ValueError):
                    corpus.build_corpus(output_dir=self.output, **options)
                download.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cli_defaults_local_dispatch_and_package_entry(self):
        defaults = corpus.build_parser().parse_args([])
        self.assertEqual((defaults.target_docs, defaults.max_docs_per_file,
                          defaults.eval_ratio, defaults.seed, defaults.min_doc_len),
                         (52000, 5000, 2000 / 52000, 42, 2))
        source = self.source([{"text": f"document {index}"} for index in range(12)])
        arguments = ["--input", str(source), "--output-dir", str(self.output), "--target-docs", "12",
                     "--max-docs-per-file", "4", "--min-doc-len", "1", "--eval-ratio", "0.25"]
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("network")), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(corpus.main(arguments), 0)
        self.assertIn("train=9 (3 shards), eval=3 (1 shards)", stdout.getvalue())
        with mock.patch.object(sys, "path", [str(SCRIPT.parent.parent), *sys.path]), \
                mock.patch.dict(sys.modules, {"corpus.download": corpus}), \
                mock.patch.object(corpus, "main", return_value=0) as dispatch, \
                self.assertRaises(SystemExit) as error:
            runpy.run_module("corpus", run_name="__main__")
        self.assertEqual(error.exception.code, 0)
        dispatch.assert_called_once_with()


    def test_default_52000_documents_produce_50000_train_and_2000_eval(self):
        source = self.source([{"text": f"d{index}"} for index in range(52000)])
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("network")):
            result = corpus.build_corpus(source, self.output, min_doc_len=1)
        self.assertEqual((result.total_docs, result.train_docs, result.eval_docs), (52000, 50000, 2000))
        self.assertEqual((len(result.train_files), len(result.eval_files)), (10, 1))
        self.assertEqual([len(self.texts([path])) for path in result.train_files], [5000] * 10)
        self.assertEqual([len(self.texts([path])) for path in result.eval_files], [2000])
        actual = self.texts(result.train_files + result.eval_files)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), {f"d{index}" for index in range(52000)})
        self.assertEqual(set(self.output.rglob("*.jsonl")), set(result.train_files + result.eval_files))

    def test_cap_boundaries_keep_contiguous_documents_and_no_empty_files(self):
        cases = [(4, 5, [4]), (5, 5, [5]), (6, 5, [5, 1]), (3, 1, [1, 1, 1])]
        for split_size, cap, sizes in cases:
            with self.subTest(split_size=split_size, cap=cap):
                records = [{"text": f"document {index}"} for index in range(split_size * 2)]
                source = self.source(records)
                output = self.root / f"boundary-{split_size}-{cap}"
                result = self.prepare(source, output_dir=output, eval_ratio=0.5, max_docs_per_file=cap)
                shuffled = [record["text"] for record in records]
                random.Random(42).shuffle(shuffled)
                for files, expected in ((result.eval_files, shuffled[:split_size]),
                                        (result.train_files, shuffled[split_size:])):
                    self.assertEqual([len(self.texts([path])) for path in files], sizes)
                    self.assertEqual([path.name for path in files],
                                     [f"lccc_part_{index + 1:03d}.jsonl" for index in range(len(sizes))])
                    for index, path in enumerate(files):
                        self.assertEqual(self.texts([path]), expected[index * cap:(index + 1) * cap])
                        self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(set(output.rglob("*.jsonl")), set(result.train_files + result.eval_files))

    def test_two_unique_documents_are_enough_for_nonempty_train_and_eval(self):
        source = self.source([{"text": "a"}, {"text": "b"}])
        for target in (2, 50000):
            with self.subTest(target_docs=target):
                result = self.prepare(source, target_docs=target, eval_ratio=0.1, max_docs_per_file=5000)
                self.assertEqual((result.total_docs, result.train_docs, result.eval_docs), (2, 1, 1))
                self.assertEqual((len(result.train_files), len(result.eval_files)), (1, 1))
                self.assertEqual([len(self.texts([path])) for path in result.train_files + result.eval_files], [1, 1])
                self.assertEqual(set(self.texts(result.train_files + result.eval_files)), {"a", "b"})
                self.assertEqual(set(self.output.rglob("*.jsonl")), set(result.train_files + result.eval_files))

    def test_large_eval_uses_same_cap_and_only_remainder_in_last_file(self):
        source = self.source([{"text": f"document {index}"} for index in range(20)])
        result = self.prepare(source, max_docs_per_file=4, eval_ratio=0.75)
        self.assertEqual((result.train_docs, result.eval_docs), (5, 15))
        self.assertEqual([len(self.texts([path])) for path in result.train_files], [4, 1])
        self.assertEqual([len(self.texts([path])) for path in result.eval_files], [4, 4, 4, 3])
        self.assertEqual(set(self.output.rglob("*.jsonl")), set(result.train_files + result.eval_files))

    def test_fixed_shard_settings_are_absent_from_cli_and_api(self):
        parameters = inspect.signature(corpus.build_corpus).parameters
        self.assertEqual(parameters["max_docs_per_file"].default, 5000)
        defaults = corpus.build_parser().parse_args([])
        for name in ("train_shards", "eval_shards"):
            with self.subTest(name=name):
                self.assertNotIn(name, parameters)
                self.assertFalse(hasattr(defaults, name))
                with mock.patch.object(corpus.urllib.request, "urlopen") as download:
                    with self.assertRaises(TypeError):
                        corpus.build_corpus(output_dir=self.output, **{name: 1})
                    download.assert_not_called()
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    corpus.build_parser().parse_args(["--" + name.replace("_", "-"), "1"])
                self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_api_size_settings_require_positive_integers_before_network(self):
        for name in ("max_docs_per_file", "target_docs", "min_doc_len", "chunk_size"):
            for value in (0, -1, 1.5, True, False, "5", None):
                with self.subTest(setting=name, value=value), \
                        mock.patch.object(corpus.urllib.request, "urlopen") as download:
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        corpus.build_corpus(output_dir=self.output, **{name: value})
                    download.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cli_rejects_invalid_max_docs_per_file_before_network(self):
        for value in ("0", "-1", "1.5", "abc"):
            with self.subTest(value=value), \
                    mock.patch.object(corpus.urllib.request, "urlopen") as download, \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                corpus.main(["--output-dir", str(self.output), "--max-docs-per-file", value])
            self.assertNotEqual(error.exception.code, 0)
            download.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
