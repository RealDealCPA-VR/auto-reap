"""Workspace layout. Everything a sweep touches lives under one root:

workspace/
  data/                    generated datasets (calibration_v*.jsonl, eval_v*.jsonl)
  artifacts/               pruned HF checkpoints and GGUFs
  runs/<config_hash>/      per-sweep state.db, results.jsonl, manifests, logs
  reports/                 rendered markdown reports + decision pages
  cache/judge/             judgment cache keyed (item_id, artifact_hash, judge_version)
"""

from __future__ import annotations

from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- directories -----------------------------------------------------

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def judge_cache(self) -> Path:
        return self.root / "cache" / "judge"

    @property
    def archive(self) -> Path:
        return self.root / "archive"

    def run_dir(self, config_hash: str) -> Path:
        return self.root / "runs" / config_hash

    def state_db(self, config_hash: str) -> Path:
        return self.run_dir(config_hash) / "state.db"

    def results_jsonl(self, config_hash: str) -> Path:
        return self.run_dir(config_hash) / "results.jsonl"

    def logs(self, config_hash: str) -> Path:
        return self.run_dir(config_hash) / "logs"

    def ensure(self, config_hash: str | None = None) -> Workspace:
        for d in (self.data, self.artifacts, self.reports, self.judge_cache, self.archive):
            d.mkdir(parents=True, exist_ok=True)
        if config_hash:
            self.run_dir(config_hash).mkdir(parents=True, exist_ok=True)
            self.logs(config_hash).mkdir(parents=True, exist_ok=True)
        return self


def free_disk_gb(path: str | Path = ".") -> float:
    """Free space on the volume holding `path` (sweep guard, PRD FR-4.2)."""
    import shutil

    return shutil.disk_usage(Path(path).resolve().anchor or ".").free / 1e9
