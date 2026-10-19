"""Dispatch complete tokenizer specifications without rebuilding vocabularies."""

from ..config import DEFAULT_TEXT_VOCAB
from .char_tokenizer import CharTokenizer
from .hybrid_tokenizer import HybridTokenizer
from .serialization import load_json, validate_tokenizer_spec


def tokenizer_from_dict(payload):
    """Restore a HybridTokenizer or legacy CharTokenizer from an exact spec."""
    if not isinstance(payload, dict) or type(payload.get("type")) is not str:
        raise ValueError("a tokenizer spec must specify its type")
    if payload["type"] == "HybridTokenizer":
        return HybridTokenizer.from_dict(payload)
    if payload["type"] == "CharTokenizer":
        validate_tokenizer_spec(payload, "CharTokenizer")
        return CharTokenizer(payload["tokens"])
    raise ValueError("unsupported tokenizer type: " + payload["type"])


def load_tokenizer(path=None):
    """Load either tokenizer format; missing or invalid files are never rebuilt."""
    return tokenizer_from_dict(load_json(DEFAULT_TEXT_VOCAB if path is None else path))


def tokenizer_spec(tokenizer):
    """Return an independent complete JSON-compatible spec for a checkpoint."""
    if isinstance(tokenizer, HybridTokenizer):
        return tokenizer.to_dict()
    if isinstance(tokenizer, CharTokenizer):
        payload = {"version": 1, "type": "CharTokenizer", "tokens": list(tokenizer.tokens)}
        tokenizer_from_dict(payload)
        return payload
    raise TypeError("unsupported tokenizer instance")
