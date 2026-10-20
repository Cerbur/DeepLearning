"""通用文本读取和预训练 packing，兼容旧五言绝句数据流程。"""
import csv
import json
import re
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import torch

from .config import MAX_SEQ_LEN, DEFAULT_TRAIN_DATA, DEFAULT_EVAL_DATA
from .corpus.cleaning import clean_corpus_text


def load_poems(path, *, min_documents=2, workers=8, strict=False):
    """通用文件/目录读取后筛选五绝，保留旧去空白与去重规则。"""
    texts = load_texts(path, min_documents=1, workers=workers, strict=strict)
    poems, seen, skipped = [], set(), 0
    for raw in texts:
        text = "".join(raw.split())
        if not re.fullmatch(r"(?:[^，。]{5}，[^，。]{5}。){2}", text):
            skipped += 1
            continue
        if text not in seen:
            seen.add(text)
            poems.append(text)
    if len(poems) < min_documents:
        raise ValueError(f"至少需要 {min_documents} 首有效五言绝句")
    print(f"有效且去重诗歌: {len(poems)}，格式不符跳过: {skipped}")
    return poems

def make_sequences(poems, tokenizer):
    # 所有有效诗歌均为 24 字符，加 BOS/EOS 为 26，无需 PAD 或改变注意力。
    return torch.tensor([tokenizer.encode(poem) for poem in poems], dtype=torch.long)


def select_train_eval_indices(train_count, sample_size, seed, saved_indices=None):
    """固定抽样，不消耗全局随机状态；恢复时校验并复用原索引。"""
    if (type(train_count) is not int or train_count < 1
            or type(sample_size) is not int or sample_size < 0):
        raise ValueError("训练集须非空，train-eval-size 须为非负整数")
    count = min(sample_size, train_count) if sample_size else train_count
    if saved_indices is not None:
        if (not isinstance(saved_indices, list) or len(saved_indices) != count
                or any(type(i) is not int or not 0 <= i < train_count for i in saved_indices)
                or len(set(saved_indices)) != count):
            raise ValueError("检查点中的训练评估索引无效或与评估大小不匹配")
        return list(saved_indices)
    if count == train_count:
        return list(range(train_count))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(train_count, generator=generator, device="cpu")[:count].tolist()


_TEXT_SUFFIXES = frozenset({".txt", ".tsv", ".csv", ".jsonl", ".json"})
_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})


def _text_error(path, line, message):
    return ValueError(f"{path}:{line}: {message}")


def _record_text(record, text_field, path, line):
    """Validate a whole record before rendering; never split a conversation."""
    if not isinstance(record, dict):
        raise _text_error(path, line, "记录必须是对象")
    if text_field in record:
        text = record[text_field]
        if not isinstance(text, str):
            raise _text_error(path, line, f"{text_field} 字段必须是字符串")
        return text
    if "messages" not in record:
        raise _text_error(path, line, f"记录必须包含 {text_field} 字段或 messages 列表")
    messages = record["messages"]
    if not isinstance(messages, list):
        raise _text_error(path, line, "messages 必须是列表")
    rendered = []
    for index, message in enumerate(messages, 1):
        if not isinstance(message, dict):
            raise _text_error(path, line, f"messages[{index}] 必须是对象")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or role not in _MESSAGE_ROLES:
            raise _text_error(path, line, f"messages[{index}].role 只能是 system/user/assistant")
        if not isinstance(content, str):
            raise _text_error(path, line, f"messages[{index}].content 必须是字符串")
        content = clean_corpus_text(content)
        if not content.strip():
            raise _text_error(path, line, f"messages[{index}].content 清洗后不能为空")
        rendered.append(role + ": " + content)
    return "\n".join(rendered)


def _json_records(source, path):
    """Yield top-level JSON records with their physical starting line numbers."""
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as error:
        raise _text_error(path, error.lineno, f"JSON 解析失败: {error.msg}") from error
    start = len(source) - len(source.lstrip())
    if isinstance(payload, dict):
        yield payload, source.count("\n", 0, start) + 1
    elif isinstance(payload, list):
        # raw_decode locates records without guessing from repeated field values.
        decoder = json.JSONDecoder()
        position = start + 1
        line, line_position = 1, 0
        for record in payload:
            while source[position].isspace() or source[position] == ",":
                position += 1
            line += source.count("\n", line_position, position)
            line_position = position
            _, position = decoder.raw_decode(source, position)
            yield record, line
    else:
        raise _text_error(path, source.count("\n", 0, start) + 1,
                          "JSON 顶层必须是记录对象或记录列表")


def _file_texts(path, text_field, reject):
    suffix = path.suffix.lower()
    # newline="" preserves original CRLF/LF and whitespace inside documents.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if suffix == ".txt":
            lines = []
            for line in handle:
                if line.strip():
                    lines.append(line)
                elif lines:
                    yield "".join(lines)
                    lines = []
            if lines:
                yield "".join(lines)
        elif suffix in {".csv", ".tsv"}:
            reader = csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",",
                                    strict=True)
            try:
                if not reader.fieldnames or text_field not in reader.fieldnames:
                    raise _text_error(path, 1, f"必须包含 {text_field} 列")
                if len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise _text_error(path, 1, "CSV/TSV 表头不能重复")
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        reject(_text_error(path, reader.line_num,
                                           "CSV/TSV 数据列数与表头不符；含分隔符的文本须正确引用"))
                        continue
                    yield row[text_field]
            except csv.Error as error:
                line_number = max(reader.line_num, reader.reader.line_num, 1)
                raise _text_error(path, line_number, f"CSV/TSV 解析失败: {error}") from error
        elif suffix == ".jsonl":
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    text = _record_text(record, text_field, path, line_number)
                except json.JSONDecodeError as error:
                    reject(_text_error(path, line_number, f"JSON 解析失败: {error.msg}"))
                    continue
                except ValueError as error:
                    reject(error)
                    continue
                yield text
        else:
            for record, line_number in _json_records(handle.read(), path):
                try:
                    text = _record_text(record, text_field, path, line_number)
                except ValueError as error:
                    reject(error)
                    continue
                yield text


def _load_text_file(path, text_field, strict):
    """Workers never print; merge errors and results in path order on the caller."""
    skipped, examples = 0, []

    def reject(error):
        nonlocal skipped
        if strict:
            raise error
        skipped += 1
        if len(examples) < 3:
            examples.append(str(error))

    texts = []
    try:
        for index, text in enumerate(_file_texts(path, text_field, reject), 1):
            if not text.strip():
                skipped += 1
                continue
            if any((ord(char) < 32 and char not in "\t\r\n")
                   or 0xD800 <= ord(char) <= 0xDFFF or ord(char) == 127 for char in text):
                reject(_text_error(path, index, "文本含二进制控制字符或无效 Unicode"))
                continue
            text = clean_corpus_text(text)
            if not text.strip():
                skipped += 1
                continue
            texts.append(text)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        if strict:
            raise
        # A corrupt/unreadable file must not leave a partially decoded corpus.
        reject(error)
        texts = []
    return texts, skipped, examples


def load_texts(path, text_field="text", *, min_documents=2, workers=8, strict=False):
    """并发递归读取支持格式，按相对路径顺序合并并去重，保留文档原始空白。

    默认跳过坏文件及坏记录并汇总警告；strict=True 用于路径/行号诊断。
    已独立划分的集合可指定 min_documents=1，不要求每个文件有两个文档。
    """
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field 必须是非空字符串")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers 必须是正整数")
    if type(min_documents) is not int or min_documents < 1:
        raise ValueError("min_documents 必须是正整数")
    path = Path(path)
    if path.is_dir():
        files = sorted((item for item in path.rglob("*")
                        if item.is_file() and item.suffix.lower() in _TEXT_SUFFIXES),
                       key=lambda item: item.relative_to(path).as_posix())
    elif not path.exists():
        raise FileNotFoundError(path)
    else:
        files = [path] if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES else []
    if not files:
        raise ValueError(f"{path}: 没有支持的文本文件 (.txt/.tsv/.csv/.jsonl/.json)")
    texts, seen, examples, skipped = [], set(), [], 0
    loader = partial(_load_text_file, text_field=text_field, strict=strict)
    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
        # executor.map preserves submission order even when files finish out of order.
        for documents, count, errors in pool.map(loader, files):
            skipped += count
            examples.extend(errors[:max(0, 5 - len(examples))])
            for text in documents:
                if text not in seen:
                    seen.add(text)
                    texts.append(text)
    if skipped:
        detail = "; ".join(examples)
        warnings.warn(f"{path}: 跳过 {skipped} 个异常文件/记录或空文档"
                      + (f"；示例: {detail}" if detail else ""), UserWarning, stacklevel=2)
    if len(texts) < min_documents:
        required = "两个" if min_documents == 2 else str(min_documents) + " 个"
        raise ValueError(f"{path}: 至少需要{required}独立且非空的有效文档，当前 {len(texts)} 个")
    return texts


def resolve_data_paths(train_data=None, eval_data=None):
    """Only use both defaults when neither input was explicitly supplied."""
    if train_data is None and eval_data is None:
        return DEFAULT_TRAIN_DATA, DEFAULT_EVAL_DATA
    if train_data is None:
        raise ValueError("单独指定 --eval-data 时也必须指定 --train-data；不会使用默认训练目录")
    return Path(train_data), Path(eval_data) if eval_data is not None else None


def load_train_eval_texts(train_data=None, eval_data=None, *, text_field="text", seed=42,
                         valid_ratio=0.1, max_documents=0, workers=8, poetry=False):
    """Load both splits concurrently, or split only explicitly supplied training data.

    Evaluation documents win exact cross-split duplicates: remove those from train,
    so tokenizer fitting and training never see an evaluation document.
    max_documents caps total documents; with two inputs reserve at least one per split.
    """
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio 必须在 (0,1)")
    if type(max_documents) is not int or max_documents < 0 or max_documents == 1:
        raise ValueError("max_documents 必须为 0 或 >=2")
    train_path, eval_path = resolve_data_paths(train_data, eval_data)
    loader = partial(load_poems, min_documents=1, workers=workers) if poetry else partial(
        load_texts, text_field=text_field, min_documents=1, workers=workers)
    if eval_path is None:
        texts = loader(train_path)
        random.Random(seed).shuffle(texts)
        if max_documents:
            texts = texts[:max_documents]
        if len(texts) < 2:
            raise ValueError("仅训练输入至少需要两个有效文档才能拆分验证集")
        count = max(1, int(len(texts) * valid_ratio))
        return texts[count:], texts[:count]
    with ThreadPoolExecutor(max_workers=2) as pool:
        train_future = pool.submit(loader, train_path)
        eval_future = pool.submit(loader, eval_path)
        train_texts, eval_texts = train_future.result(), eval_future.result()
    eval_set = set(eval_texts)
    filtered = [text for text in train_texts if text not in eval_set]
    if len(filtered) != len(train_texts):
        warnings.warn(f"训练集移除 {len(train_texts) - len(filtered)} 个与验证集完全重复的文档",
                      UserWarning, stacklevel=2)
    train_texts = filtered
    if not train_texts:
        raise ValueError("去除与验证集重复的文档后训练集为空；请提供独立训练语料")
    # Do not re-split an already-separated corpus; shuffle using a local RNG only.
    generator = random.Random(seed)
    generator.shuffle(train_texts)
    generator.shuffle(eval_texts)
    if max_documents and len(train_texts) + len(eval_texts) > max_documents:
        count = max(1, min(len(eval_texts), max_documents - 1,
                           int(max_documents * len(eval_texts) / (len(train_texts) + len(eval_texts)))))
        eval_texts = eval_texts[:count]
        train_texts = train_texts[:max_documents - count]
    return train_texts, eval_texts

def make_pretrain_sequences(texts, tokenizer, seq_len=MAX_SEQ_LEN):
    """Pack only one already-separated split into next-token prediction rows.

    Each document is encoded with BOS/EOS. Windows contain seq_len + 1 tokens
    and advance by seq_len, sharing one boundary token so every stream target
    appears exactly once. Right padding is -100 for CrossEntropyLoss's default
    ignore_index; callers must clamp input IDs to zero, not the target labels.
    A one-token tail has no next-token target and is intentionally omitted.
    """
    if type(seq_len) is not int or not 1 <= seq_len <= MAX_SEQ_LEN:
        raise ValueError(f"seq_len 必须是 1 到 {MAX_SEQ_LEN} 的正整数")
    if texts is None or isinstance(texts, (str, bytes)):
        raise ValueError("texts 必须是非空文档集合")
    stream = []
    for text in texts:
        if not isinstance(text, str):
            raise ValueError("texts 中的每个文档必须是字符串")
        if text.strip():
            stream.extend(tokenizer.encode(text, special_tokens=True))
    if len(stream) < 2:
        raise ValueError("数据为空或编码后没有足够 tokens 进行下一个 token 预测")
    sequences = []
    for start in range(0, len(stream) - 1, seq_len):
        chunk = stream[start:start + seq_len + 1]
        sequences.append(chunk + [-100] * (seq_len + 1 - len(chunk)))
    return torch.tensor(sequences, dtype=torch.long)
