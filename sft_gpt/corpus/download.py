"""Download/reuse documents and publish seeded train/eval shards (stdlib only).

Example (project directory): python -m corpus --max-docs-per-file 5000
Importing this module never downloads or writes anything. Original inputs remain intact.
"""
from __future__ import annotations

import argparse
import codecs
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import BinaryIO, Iterator
import urllib.request
from urllib.parse import urlsplit

if __package__:
    from .cleaning import clean_corpus_text
else:
    from cleaning import clean_corpus_text

# This is the train archive referenced by thu-coai/lccc's official HF loader.
# Only its training split is used; upstream validation/test remain untouched.
URL = "https://hf-mirror.com/datasets/silver/lccc/resolve/main/lccc_base_train.jsonl.gz"
TARGET_DIR = Path(__file__).resolve().parents[1] / ".data" / "corpus"
TRAIN_DOCS = 50000
EVAL_DOCS = 2000
TARGET_DOCS = TRAIN_DOCS + EVAL_DOCS
MIN_DOC_LEN = 2
MAX_DOCS_PER_FILE = 5000
EVAL_RATIO = EVAL_DOCS / TARGET_DOCS
SEED = 42
MANIFEST = ".download_corpus_manifest.json"
_GENERATED = re.compile(r"(?:train|eval)/(?:wiki_zh|lccc)_part_[0-9]{3,}\.jsonl\Z")


@dataclass(frozen=True)
class CorpusResult:
    total_docs: int
    train_docs: int
    eval_docs: int
    train_files: tuple[Path, ...]
    eval_files: tuple[Path, ...]


def _decoded_chunks(stream: BinaryIO, chunk_size: int) -> Iterator[str]:
    """Strict incremental UTF-8 decoding, even across codepoint boundaries."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            tail = decoder.decode(b"", final=True)
            if tail:
                yield tail
            return
        yield decoder.decode(chunk, final=False)


def iter_json_records(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    """Stream a JSON array or one record, without splitting documents.

    Broken JSON/UTF-8 is an input failure. Record-schema errors are filtered later.
    """
    chunks = iter(_decoded_chunks(stream, chunk_size))
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False
    mode = None
    state = "value_or_end"
    while True:
        buffer = buffer.lstrip()
        if mode is None and buffer:
            if buffer.startswith("["):
                mode, buffer = "array", buffer[1:]
            else:
                mode = "single"
            continue
        if mode == "array" and buffer:
            if state in ("value_or_end", "comma_or_end") and buffer.startswith("]"):
                buffer, mode = buffer[1:], "done"
                continue
            if state == "comma_or_end":
                if not buffer.startswith(","):
                    raise ValueError("Invalid JSON array: expected comma or closing bracket")
                buffer, state = buffer[1:], "value"
                continue
        if mode == "done":
            if buffer:
                raise ValueError("Unexpected data after JSON document")
            if eof:
                return
        elif buffer and mode is not None:
            try:
                record, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError("Invalid or truncated JSON input") from error
            else:
                # JSONDecoder may decode 1 out of an incomplete 1.5 or 1e+3.
                number = isinstance(record, (int, float)) and not isinstance(record, bool)
                complete = end < len(buffer) and (not number or buffer[end] in " \t\r\n,]")
                if complete or eof:
                    buffer = buffer[end:]
                    if mode == "single":
                        mode = "done"
                    else:
                        state = "comma_or_end"
                    yield record
                    continue
        if eof:
            raise ValueError("Empty or truncated JSON input")
        try:
            buffer += next(chunks)
        except StopIteration:
            eof = True


def _iter_jsonl(stream: BinaryIO, chunk_size: int) -> Iterator[object]:
    buffer = ""
    for chunk in _decoded_chunks(stream, chunk_size):
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Malformed/blank JSONL records are invalid records, not documents.
                continue
    if buffer.strip():
        try:
            yield json.loads(buffer)
        except json.JSONDecodeError:
            return


@contextmanager
def _record_stream(stream: BinaryIO, name: str, chunk_size: int):
    """Read JSON/JSONL, optionally gzip, without downloading a full archive."""
    compressed = name.lower().endswith(".gz")
    plain_name = name[:-3] if compressed else name
    reader = _iter_jsonl if plain_name.lower().endswith(".jsonl") else iter_json_records
    if compressed:
        with gzip.GzipFile(fileobj=stream, mode="rb") as decoded:
            iterator = reader(decoded, chunk_size)
            try:
                yield iterator
            finally:
                iterator.close()
    else:
        iterator = reader(stream, chunk_size)
        try:
            yield iterator
        finally:
            iterator.close()


@contextmanager
def _source_records(input_path: Path | None, output_dir: Path, url: str, chunk_size: int):
    if input_path is None:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with _record_stream(response, urlsplit(url).path, chunk_size) as records:
                yield records
        return
    if input_path.is_dir():
        same_root = input_path.resolve() == output_dir.resolve()
        files = sorted(
            (path for path in input_path.rglob("*")
             if path.is_file() and path.name.lower().endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz"))
             and not any(part.startswith(".") for part in path.relative_to(input_path).parts)
             and not (same_root and path.relative_to(input_path).parts[0] in ("train", "eval"))),
            key=lambda path: path.relative_to(input_path).as_posix(),
        )
    elif input_path.is_file():
        files = [input_path]
    else:
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if not files:
        raise ValueError(f"No JSON/JSONL input files found: {input_path}")

    def records():
        for path in files:
            with path.open("rb") as stream:
                with _record_stream(stream, path.name, chunk_size) as iterator:
                    yield from iterator

    iterator = records()
    try:
        yield iterator
    finally:
        iterator.close()


def _dialogue_text(record: object, min_doc_len: int, tokenization: str = "word") -> str | None:
    """LCCC JSONL contains utterance lists; HF rows may wrap them in dialog."""
    turns = record.get("dialog") if isinstance(record, dict) else record
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    cleaned = []
    for turn in turns:
        if not isinstance(turn, str):
            return None
        # The default HF archive is word-segmented. Character-spaced releases
        # need an explicit mode: ASCII word/number spacing cannot be inferred.
        tokens = turn.split()
        if tokenization == "character":
            turn = "".join(tokens)
        else:
            turn = "".join((" " if index and re.fullmatch(r"[A-Za-z0-9]", tokens[index - 1][-1])
                            and re.fullmatch(r"[A-Za-z0-9]", token[0]) else "") + token
                           for index, token in enumerate(tokens))
        content = clean_corpus_text(turn).strip()
        if not content:
            return None  # Keep the complete dialogue, never bridge across a bad turn.
        cleaned.append(content)
    if sum(len(content) for content in cleaned) < min_doc_len:
        return None
    # Plain causal-LM pretraining: no synthetic speaker labels or chat template.
    return "\n".join(cleaned)


def _clean_text(record: object, min_doc_len: int, tokenization: str = "word") -> str | None:
    if isinstance(record, list) or isinstance(record, dict) and "dialog" in record:
        return _dialogue_text(record, min_doc_len, tokenization)
    if not isinstance(record, dict):
        return None
    text = record.get("text", record.get("completion"))
    if not isinstance(text, str):
        return None
    text = clean_corpus_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < min_doc_len or not text:
        return None
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return text


def _owned_files(output_dir: Path) -> set[str]:
    path = output_dir / MANIFEST
    if not path.exists():
        return set()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or not isinstance(manifest.get("files"), list)
            or any(not isinstance(name, str) or not _GENERATED.fullmatch(name)
                   for name in manifest["files"])):
        raise ValueError(f"Invalid corpus ownership manifest: {path}")
    return set(manifest["files"])


def _publish(output_dir: Path, documents: dict[str, list[str]], max_docs_per_file: int,
             metadata: dict | None = None) -> dict[str, tuple[Path, ...]]:
    """Stage first, replace only manifest-owned files, roll back on exceptions.

    This is not a multi-process transaction or power-loss recovery protocol.
    """
    owned = _owned_files(output_dir)
    names = {split: tuple(f"{split}/lccc_part_{index + 1:03d}.jsonl"
                         for index in range((len(texts) + max_docs_per_file - 1) // max_docs_per_file))
             for split, texts in documents.items()}
    new_files = {name for paths in names.values() for name in paths}
    for split in documents:
        if (output_dir / split).is_symlink():
            raise ValueError(f"Refusing symlink output directory: {output_dir / split}")
    for name in new_files | owned:
        path = output_dir / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Not a regular output file: {path}")
        if name in new_files and path.exists() and name not in owned:
            raise ValueError(f"Refusing to overwrite an unowned file: {path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".corpus-stage-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        for split, texts in documents.items():
            for index, name in enumerate(names[split]):
                offset = index * max_docs_per_file
                path = stage / "new" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    for text in texts[offset:offset + max_docs_per_file]:
                        stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        manifest = stage / "new" / MANIFEST
        manifest.write_text(json.dumps({"version": 1, "files": sorted(new_files),
                                        "preparation": metadata or {}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        backed_up = []
        installed = []
        try:
            for name in sorted(owned | {MANIFEST}):
                destination = output_dir / name
                if destination.exists():
                    backup = stage / "backup" / name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(destination, backup)
                    backed_up.append((destination, backup))
            for name in sorted(new_files) + [MANIFEST]:
                destination = output_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / "new" / name, destination)
                installed.append(destination)
        except BaseException:
            for destination in reversed(installed):
                destination.unlink()
            for destination, backup in reversed(backed_up):
                os.replace(backup, destination)
            raise
    return {split: tuple(output_dir / name for name in paths) for split, paths in names.items()}


def build_corpus(input_path: str | Path | None = None, output_dir: str | Path = TARGET_DIR,
                 *, target_docs: int = TARGET_DOCS, eval_ratio: float = EVAL_RATIO,
                 max_docs_per_file: int = MAX_DOCS_PER_FILE,
                 seed: int = SEED, min_doc_len: int = MIN_DOC_LEN,
                 url: str = URL, chunk_size: int = 1024 * 1024,
                 tokenization: str = "word") -> CorpusResult:
    """Collect up to target_docs valid unique documents, then publish train/eval.

    LCCC utterance lists / dialog rows become newline-joined plain text.
    Local JSON arrays/records and JSONL directories (also gzip) avoid network entirely.
    Apply the shared Chinese/ASCII-English/digits/punctuation whitelist, normalize,
    deduplicate, seeded-shuffle whole documents, then split. EOF below target_docs
    is allowed when both splits are nonempty. Each file holds at most
    max_docs_per_file records; file counts follow split sizes, with no fixed count.
    Broken JSONL records are
    skipped; broken JSON arrays or UTF-8 fail safely. Inputs are never deleted.
    """
    if tokenization not in ("word", "character"):
        raise ValueError("tokenization must be word or character")
    sizes = (target_docs, max_docs_per_file, min_doc_len, chunk_size)
    if any(type(value) is not int or value < 1 for value in sizes):
        raise ValueError("target_docs, max_docs_per_file, min_doc_len and chunk_size must be positive integers")
    if target_docs < 2:
        raise ValueError("target_docs must be at least 2 for nonempty train/eval splits")
    if not math.isfinite(eval_ratio) or not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1 (exclusive)")
    output_dir = Path(output_dir)
    texts = []
    seen = set()
    with _source_records(None if input_path is None else Path(input_path), output_dir, url, chunk_size) as records:
        for record in records:
            text = _clean_text(record, min_doc_len, tokenization)
            if text is None or text in seen:
                continue
            seen.add(text)
            texts.append(text)
            if len(texts) >= target_docs:
                break
    eval_docs = max(1, int(len(texts) * eval_ratio))
    train_docs = len(texts) - eval_docs
    if train_docs < 1 or eval_docs < 1:
        raise ValueError(f"Not enough valid unique documents ({len(texts)}): "
                         "need at least two documents for nonempty train/eval splits")
    random.Random(seed).shuffle(texts)
    files = _publish(output_dir, {"train": texts[eval_docs:], "eval": texts[:eval_docs]},
                     max_docs_per_file, metadata={
                         "source": str(Path(input_path).resolve()) if input_path is not None else url,
                         "target_docs": target_docs, "seed": seed, "eval_ratio": eval_ratio,
                         "tokenization": tokenization,
                         "min_doc_len": min_doc_len, "max_docs_per_file": max_docs_per_file,
                         "total_docs": len(texts), "train_docs": train_docs, "eval_docs": eval_docs,
                         "dialogue_template": "plain text, newline-separated turns, no role labels",
                     })
    return CorpusResult(len(texts), train_docs, eval_docs, files["train"], files["eval"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CDial-GPT LCCC-base conversations and create whole-dialogue train/eval shards")
    parser.add_argument("--input", type=Path, help="Local LCCC JSON/JSONL (optionally .gz) file or directory; no network when supplied")
    parser.add_argument("--output-dir", type=Path, default=TARGET_DIR)
    parser.add_argument("--target-docs", type=int, default=TARGET_DOCS, help="Maximum valid unique conversations in total (default: 52000 = 50000 train + 2000 eval)")
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO,
                        help="Evaluation fraction (default: 2000/52000); split whole conversations")
    parser.add_argument("--max-docs-per-file", type=int, default=MAX_DOCS_PER_FILE,
                        help="Maximum records in each train/eval file (default: 5000); no fixed file count")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-doc-len", type=int, default=MIN_DOC_LEN,
                        help="Minimum total cleaned content characters (excluding role labels); default: 2")
    parser.add_argument("--tokenization", choices=("word", "character"), default="word",
                        help="LCCC segmentation: default word preserves ASCII word/number boundaries; character joins all tokens")
    parser.add_argument("--url", default=URL, help="Network source, used only when --input is omitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = build_corpus(input_path=args.input, output_dir=args.output_dir,
                              target_docs=args.target_docs, eval_ratio=args.eval_ratio,
                              max_docs_per_file=args.max_docs_per_file,
                              seed=args.seed, min_doc_len=args.min_doc_len, url=args.url,
                              tokenization=args.tokenization)
    except (OSError, ValueError, EOFError) as error:
        parser.exit(1, f"Corpus preparation failed: {error}\n")
    print(f"Prepared {result.total_docs} unique documents: "
          f"train={result.train_docs} ({len(result.train_files)} shards), "
          f"eval={result.eval_docs} ({len(result.eval_files)} shards) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
