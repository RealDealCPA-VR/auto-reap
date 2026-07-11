"""Canonical hashing: config hashes and streamed artifact hashes (PRD §5 reproducibility)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(obj: Any, length: int = 12) -> str:
    """Short stable hash of any JSON-serializable object."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def file_hash(path: str | Path) -> str:
    """Streamed sha256 of one file (safe for multi-GB GGUFs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def dir_hash(path: str | Path) -> str:
    """Order-independent hash of a directory tree: sorted relative paths + content hashes."""
    root = Path(path)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode("utf-8"))
            h.update(bytes.fromhex(file_hash(p)))
    return h.hexdigest()


def artifact_hash(path: str | Path) -> str:
    """Hash a file or a checkpoint directory uniformly."""
    p = Path(path)
    return file_hash(p) if p.is_file() else dir_hash(p)
