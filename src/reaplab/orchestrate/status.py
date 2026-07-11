"""Human-readable sweep status for the CLI (`reap-lab status <sweep.yaml>`).

Reads only the StateDB for the spec's config hash — safe to call while a
sweep is running in another process (SQLite handles concurrent readers).
"""

from __future__ import annotations

from collections import Counter

from reaplab.core.config import SweepSpec
from reaplab.core.paths import Workspace
from reaplab.core.state import StateDB


def sweep_status(spec: SweepSpec) -> str:
    """Summarize sweep progress: stage counts, per-stage status lines, and
    per-artifact metrics recorded so far. Returns a plain-text block."""
    config_hash = spec.config_hash()
    workspace = Workspace(spec.workspace)
    db_path = workspace.state_db(config_hash)
    if not db_path.exists():
        return (
            f"No sweep state found for config {config_hash} (expected {db_path}). "
            "Run `uv run reap-lab sweep <your-sweep.yaml>` to start one."
        )

    lines: list[str] = [f"Sweep {config_hash} — {spec.model_id}"]
    with StateDB(db_path) as state:
        jobs = state.jobs()
        counts = Counter(job["status"] for job in jobs)
        lines.append(
            f"Stages: {counts.get('done', 0)} done, {counts.get('failed', 0)} failed, "
            f"{counts.get('running', 0)} running"
        )
        lines.append("")
        for job in jobs:
            line = f"  [{job['status']:>7}] {job['stage']}:{job['key']}"
            if job["status"] == "failed" and job.get("error"):
                line += f" — {job['error']}"
            lines.append(line)

        artifacts = state.all_artifacts()
        if artifacts:
            lines += ["", "Artifact metrics:"]
            for artifact_id in artifacts:
                metrics = state.metrics_for(artifact_id)
                parts = []
                for name in sorted(metrics):
                    value = metrics[name]
                    parts.append(
                        f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
                    )
                lines.append(f"  {artifact_id}: " + ", ".join(parts))
    return "\n".join(lines)
