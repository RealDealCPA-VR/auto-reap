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

    def data_dir(self, config_hash: str) -> Path:
        """Per-sweep dataset directory. Datasets are keyed by config hash so two specs
        sharing one workspace can never overwrite each other's calibration/eval sets
        (and a resumed sweep always reads exactly the data it started with)."""
        return self.run_dir(config_hash) / "data"

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
            self.data_dir(config_hash).mkdir(parents=True, exist_ok=True)
            self.logs(config_hash).mkdir(parents=True, exist_ok=True)
        return self


def free_disk_gb(path: str | Path = ".") -> float:
    """Free space on the volume that actually holds `path` (sweep guard, PRD FR-4.2).

    Measures the deepest existing ancestor of `path`, not the drive anchor — on
    systems where the workspace sits on a mounted volume, the anchor would report
    the wrong disk."""
    import shutil

    p = Path(path).resolve()
    while not p.exists():
        parent = p.parent
        if parent == p:
            break
        p = parent
    return shutil.disk_usage(p).free / 1e9
