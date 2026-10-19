"""Strict JSON I/O shared by tokenizer files and embedded tokenizer specs."""

import json
import math
import os
from pathlib import Path
import tempfile


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("invalid JSON constant: " + value)


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number: " + value)
    return number


def load_json(path):
    """Read UTF-8 JSON, rejecting duplicate keys and non-finite numbers."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant, parse_float=_finite_float)


def save_json_atomic(path, payload):
    """Atomically save JSON; a failed write leaves the old destination intact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix="." + destination.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def validate_tokenizer_spec(payload, tokenizer_type):
    """Validate the exact common envelope without changing token order."""
    if not isinstance(payload, dict) or set(payload) != {"version", "type", "tokens"}:
        raise ValueError("vocabulary must contain exactly version, type and tokens")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("unsupported vocabulary format version")
    if type(payload["type"]) is not str or payload["type"] != tokenizer_type:
        raise ValueError("incompatible tokenizer type")
    if not isinstance(payload["tokens"], list):
        raise ValueError("tokens must be an ordered JSON array")
    if any(not isinstance(token, str) for token in payload["tokens"]):
        raise ValueError("every token must be a string")
