"""Character vocabulary compatible with the original mini-GPT tokenizer."""

from collections import Counter
import json
import os
from pathlib import Path
import tempfile

from ..config import DEFAULT_VOCAB, MAX_TOKEN_FREQ


_SPECIAL_TOKENS = ("<BOS>", "<EOS>", "<UNK>")
_FORMAT_VERSION = 1
_TOKENIZER_TYPE = "CharTokenizer"


def _validate_tokens(tokens):
    if not isinstance(tokens, list):
        raise ValueError("tokens must be an ordered JSON array")
    if any(not isinstance(token, str) for token in tokens):
        raise ValueError("every token must be a string")
    if tokens[:3] != list(_SPECIAL_TOKENS):
        raise ValueError("special tokens must be <BOS>, <EOS>, <UNK> at IDs 0, 1, 2")
    if len(set(tokens)) != len(tokens):
        raise ValueError("duplicate tokens are not allowed")
    if any(len(token) != 1 for token in tokens[3:]):
        raise ValueError("ordinary tokens must contain exactly one character")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field: " + key)
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError("invalid JSON constant: " + value)


class CharTokenizer:
    """Map characters to stable IDs, with BOS=0, EOS=1 and UNK=2.

    The JSON schema is {"version": 1, "type": "CharTokenizer", "tokens": [...]} .
    Loading preserves token order exactly; only build() sorts ordinary characters.
    """

    def __init__(self, tokens):
        self.tokens = list(tokens)
        _validate_tokens(self.tokens)
        self.token_ids = {token: i for i, token in enumerate(self.tokens)}
        self.bos, self.eos, self.unk = 0, 1, 2

    @classmethod
    def build(cls, poems, min_freq=MAX_TOKEN_FREQ):
        """Build in Unicode sort order, always retaining Chinese comma/full stop."""
        counts = Counter("".join(poems))
        chars = {c for c, n in counts.items() if n >= min_freq} | set("，。")
        return cls(list(_SPECIAL_TOKENS) + sorted(chars))

    def encode(self, text, special_tokens=True):
        ids = [self.token_ids.get(c, self.unk) for c in text]
        return [self.bos] + ids + [self.eos] if special_tokens else ids

    def decode(self, ids):
        return "".join(self.tokens[i] for i in ids if i not in (self.bos, self.eos))

    def save(self, path=None):
        """Atomically write UTF-8 JSON and return its Path.

        Parent directories are created as needed. A temporary file in the same
        directory is flushed and closed before os.replace; failures remove it
        and leave an existing destination intact.
        """
        _validate_tokens(self.tokens)
        destination = Path(DEFAULT_VOCAB if path is None else path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                dir=destination.parent, prefix="." + destination.name + ".",
                suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    {"version": _FORMAT_VERSION, "type": _TOKENIZER_TYPE,
                     "tokens": self.tokens},
                    stream, ensure_ascii=False, indent=2, allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path=None):
        """Read and validate a vocabulary, never rebuilding on missing/bad data.

        I/O errors propagate. Malformed JSON/schema/token data raise ValueError.
        """
        source = Path(DEFAULT_VOCAB if path is None else path)
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(
                stream, object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        if not isinstance(payload, dict) or set(payload) != {"version", "type", "tokens"}:
            raise ValueError("vocabulary must contain exactly version, type and tokens")
        if type(payload["version"]) is not int or payload["version"] != _FORMAT_VERSION:
            raise ValueError("unsupported vocabulary format version")
        if payload["type"] != _TOKENIZER_TYPE:
            raise ValueError("incompatible tokenizer type")
        _validate_tokens(payload["tokens"])
        return cls(payload["tokens"])
