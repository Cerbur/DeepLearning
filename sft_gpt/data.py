"""通用文本读取和预训练 packing，兼容旧五言绝句数据流程。"""
import csv
import json
import re
from pathlib import Path

import torch

from .config import MAX_SEQ_LEN


def load_poems(path):
    """只读取 text 列，去空白、过滤非标准五绝并去重，避免跨集合重复。"""
    poems, seen, skipped = [], set(), 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("TSV 必须包含 text 列")
        for row in reader:
            text = "".join((row.get("text") or "").split())
            if not re.fullmatch(r"(?:[^，。]{5}，[^，。]{5}。){2}", text):
                skipped += 1
                continue
            if text not in seen:
                seen.add(text)
                poems.append(text)
    if len(poems) < 2:
        raise ValueError("至少需要两首有效五言绝句才能划分训练集和验证集")
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


def _file_texts(path, text_field):
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
                    if None in row:
                        raise _text_error(path, reader.line_num,
                                          "CSV/TSV 数据列数超过表头；含分隔符的文本须正确引用")
                    text = row.get(text_field)
                    if not isinstance(text, str):
                        raise _text_error(path, reader.line_num,
                                          f"{text_field} 列必须是字符串，不能缺失")
                    yield text
            except csv.Error as error:
                line_number = max(reader.line_num, reader.reader.line_num, 1)
                raise _text_error(path, line_number, f"CSV/TSV 解析失败: {error}") from error
        elif suffix == ".jsonl":
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise _text_error(path, line_number,
                                      f"JSON 解析失败: {error.msg}") from error
                yield _record_text(record, text_field, path, line_number)
        else:
            for record, line_number in _json_records(handle.read(), path):
                yield _record_text(record, text_field, path, line_number)


def load_texts(path, text_field="text"):
    """Load unique intact documents, before any split or tokenization.

    Directories are visited recursively in relative-path order; unsupported files
    are ignored. TXT blank lines delimit documents; all nonblank lines, including
    their original line endings and leading/trailing spaces, are kept verbatim.
    CSV/TSV and JSON text fields are also kept verbatim. Conversations use the
    deterministic role: content template joined by a single newline.
    """
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field 必须是非空字符串")
    path = Path(path)
    if path.is_dir():
        files = sorted((item for item in path.rglob("*")
                        if item.is_file() and item.suffix.lower() in _TEXT_SUFFIXES),
                       key=lambda item: item.relative_to(path).as_posix())
    elif not path.exists():
        raise FileNotFoundError(path)
    else:
        files = [path] if path.suffix.lower() in _TEXT_SUFFIXES else []
    if not files:
        raise ValueError(f"{path}: 没有支持的文本文件 (.txt/.tsv/.csv/.jsonl/.json)")
    texts, seen = [], set()
    for file in files:
        for text in _file_texts(file, text_field):
            if text.strip() and text not in seen:
                seen.add(text)
                texts.append(text)
    if len(texts) < 2:
        raise ValueError(f"{path}: 至少需要两个独立且非空的文档才能划分训练集和验证集")
    return texts


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
