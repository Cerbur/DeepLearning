"""共享的轻量字符白名单清洗；下载脚本无需导入 PyTorch。"""
from functools import lru_cache
import string
import unicodedata


# CJK unified/compatibility ideographs, including supplementary extensions.
_HAN_RANGES = (
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0x2EBF0, 0x2EE5F),
    (0x2F800, 0x2FA1F), (0x30000, 0x3134F), (0x31350, 0x323AF),
)


@lru_cache(maxsize=65536)
def _character_kind(char):
    """0: unwanted, 1: punctuation/whitespace, 2: meaningful zh/en/number."""
    if "a" <= char <= "z" or "A" <= char <= "Z" or "0" <= char <= "9":
        return 2
    if char in " \t\r\n" or char in string.punctuation:
        return 1
    code = ord(char)
    if code == 0x3007 or any(low <= code <= high for low, high in _HAN_RANGES):
        return 2
    if unicodedata.category(char).startswith("P"):
        return 1
    return 0


def clean_corpus_text(text):
    """Keep Han, ASCII English/digits, punctuation and space/tab/CR/LF only.

    Replace each unwanted run with a space rather than joining neighboring words.
    Original allowed whitespace stays verbatim; punctuation-only output is empty.
    This is a character policy, not semantic language detection: Han shared by
    Chinese and Japanese cannot be distinguished without additional NLP tooling.
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    output, rejected, meaningful = [], False, False
    for char in text:
        kind = _character_kind(char)
        if kind:
            output.append(char)
            rejected = False
            meaningful = meaningful or kind == 2
        elif not rejected:
            output.append(" ")
            rejected = True
    return "".join(output) if meaningful else ""
