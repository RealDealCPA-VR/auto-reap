"""Schema-validated JSONL IO. All datasets and per-item results flow through here."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: str | Path, model: type[T]) -> list[T]:
    """Read and validate every line. Raises on the first invalid record with its line number."""
    out: list[T] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(model.model_validate(json.loads(line)))
            except Exception as e:  # noqa: BLE001 - re-raise with location
                raise ValueError(f"{path}:{lineno}: invalid {model.__name__}: {e}") from e
    return out


def iter_jsonl(path: str | Path, model: type[T]) -> Iterator[T]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield model.model_validate(json.loads(line))
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{path}:{lineno}: invalid {model.__name__}: {e}") from e


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    """Write records atomically (tmp file + replace). Returns the record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
            n += 1
    tmp.replace(path)
    return n


def append_jsonl(path: str | Path, record: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(record.model_dump_json(exclude_none=True) + "\n")
