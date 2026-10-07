"""Strict JSON, content addressing, and atomic local writes. No deserialization code."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


class IntegrityError(ValueError):
    """An artifact cannot be trusted or does not satisfy its schema."""


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise IntegrityError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise IntegrityError(f"Non-finite JSON constant: {value}")


def finite_tree(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise IntegrityError(f"Non-finite value at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{location}[{index}]")


def canonical(value: Any) -> bytes:
    finite_tree(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_json(path: str | Path) -> Any:
    try:
        value = json.loads(Path(path).read_text(), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Cannot read JSON artifact {path}: {exc}") from exc
    finite_tree(value)
    return value


def atomic_write(path: str | Path, data: bytes) -> None:
    """Publish a complete file with rename, then sync the parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any) -> str:
    atomic_write(path, canonical(value))
    return digest(value)


def require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise IntegrityError(f"{label} fields must be exactly {sorted(keys)}")
