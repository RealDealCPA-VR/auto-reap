"""Resumable job state in SQLite (PRD FR-4.1). One row per (stage, key) unit of work.

Usage:
    db = StateDB(workspace / "state.db")
    if not db.is_done("prune", "r0.50"):
        db.mark_running("prune", "r0.50")
        ...work...
        db.mark_done("prune", "r0.50", meta={"path": "..."})
Failure isolation: mark_failed records the error; the sweep moves on (PRD FR-4.2).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    stage TEXT NOT NULL,
    key TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    meta TEXT,
    PRIMARY KEY (stage, key)
);
CREATE TABLE IF NOT EXISTS metrics (
    artifact_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL,
    text_value TEXT,
    PRIMARY KEY (artifact_id, name)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> StateDB:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- job lifecycle -------------------------------------------------------

    def mark_running(self, stage: str, key: str) -> None:
        self._conn.execute(
            "INSERT INTO jobs (stage, key, status, started_at) VALUES (?, ?, 'running', ?) "
            "ON CONFLICT(stage, key) DO UPDATE SET status='running', started_at=?, error=NULL",
            (stage, key, _now(), _now()),
        )
        self._conn.commit()

    def mark_done(self, stage: str, key: str, meta: dict[str, Any] | None = None) -> None:
        self._conn.execute(
            "INSERT INTO jobs (stage, key, status, started_at, finished_at, meta) "
            "VALUES (?, ?, 'done', ?, ?, ?) "
            "ON CONFLICT(stage, key) DO UPDATE SET status='done', finished_at=?, meta=?, error=NULL",
            (stage, key, _now(), _now(), json.dumps(meta or {}), _now(), json.dumps(meta or {})),
        )
        self._conn.commit()

    def mark_failed(self, stage: str, key: str, error: str) -> None:
        self._conn.execute(
            "INSERT INTO jobs (stage, key, status, started_at, finished_at, error) "
            "VALUES (?, ?, 'failed', ?, ?, ?) "
            "ON CONFLICT(stage, key) DO UPDATE SET status='failed', finished_at=?, error=?",
            (stage, key, _now(), _now(), error[:4000], _now(), error[:4000]),
        )
        self._conn.commit()

    def status(self, stage: str, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM jobs WHERE stage=? AND key=?", (stage, key)
        ).fetchone()
        return row[0] if row else None

    def is_done(self, stage: str, key: str) -> bool:
        return self.status(stage, key) == "done"

    def meta(self, stage: str, key: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT meta FROM jobs WHERE stage=? AND key=?", (stage, key)
        ).fetchone()
        return json.loads(row[0]) if row and row[0] else {}

    def jobs(self, stage: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT stage, key, status, started_at, finished_at, error, meta FROM jobs"
        params: tuple[Any, ...] = ()
        if stage:
            q += " WHERE stage=?"
            params = (stage,)
        rows = self._conn.execute(q + " ORDER BY stage, key", params).fetchall()
        return [
            {
                "stage": r[0],
                "key": r[1],
                "status": r[2],
                "started_at": r[3],
                "finished_at": r[4],
                "error": r[5],
                "meta": json.loads(r[6]) if r[6] else {},
            }
            for r in rows
        ]

    # -- metrics (report queries) --------------------------------------------

    def record_metric(self, artifact_id: str, name: str, value: float | str) -> None:
        if isinstance(value, str):
            self._conn.execute(
                "INSERT INTO metrics (artifact_id, name, text_value) VALUES (?, ?, ?) "
                "ON CONFLICT(artifact_id, name) DO UPDATE SET text_value=?, value=NULL",
                (artifact_id, name, value, value),
            )
        else:
            self._conn.execute(
                "INSERT INTO metrics (artifact_id, name, value) VALUES (?, ?, ?) "
                "ON CONFLICT(artifact_id, name) DO UPDATE SET value=?, text_value=NULL",
                (artifact_id, name, float(value), float(value)),
            )
        self._conn.commit()

    def metrics_for(self, artifact_id: str) -> dict[str, float | str]:
        rows = self._conn.execute(
            "SELECT name, value, text_value FROM metrics WHERE artifact_id=?", (artifact_id,)
        ).fetchall()
        return {r[0]: (r[1] if r[1] is not None else r[2]) for r in rows}

    def all_artifacts(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT artifact_id FROM metrics ORDER BY artifact_id").fetchall()
        return [r[0] for r in rows]
