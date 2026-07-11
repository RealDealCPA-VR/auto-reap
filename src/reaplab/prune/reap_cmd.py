"""Pure builders for the REAP invocation (PRD FR-2.1).

Ground truth: docs/RESEARCH_BRIEF.md section 1.

- REAP's entry point is the *file* ``src/reap/prune.py`` (run from the repo
  root), not ``python -m``.
- ``--compression-ratio`` is the fraction of experts REMOVED; our specs use
  *retention* (fraction kept), so ``compression_ratio = 1 - retention``.
- REAP has no jsonl flag: calibration data must be an HF ``load_dataset()``
  target. A local folder containing a jsonl file with a ``messages`` column
  loads via ``load_dataset(folder)`` -- :func:`calib_to_dataset_dir` produces
  exactly that folder from our prompt-only ``calibration.jsonl``.

Everything here is a pure function: no subprocesses, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from reaplab.core.config import SweepSpec
from reaplab.core.jsonl import iter_jsonl
from reaplab.core.records import CalibrationRecord

from reaplab.prune.errors import PruneError

#: Basename of the converted dataset file inside the dataset folder.
DATASET_FILENAME = "data.jsonl"


def compression_ratio(retention: float) -> float:
    """Convert *retention* (fraction of experts KEPT, our config convention)
    to REAP's ``--compression-ratio`` (fraction REMOVED).

    Rounded to 10 decimal places so binary-float noise never leaks into the
    command line (``1 - 0.9`` is ``0.09999999999999998`` raw).
    """
    if not 0 < retention <= 1:
        raise PruneError(
            f"retention must be in (0, 1], got {retention}. "
            "Retention is the fraction of experts to KEEP (e.g. 0.5 keeps half)."
        )
    return round(1.0 - retention, 10)


def format_ratio(retention: float) -> str:
    """``--compression-ratio`` argument string: shortest exact form.

    ``0.625 -> "0.375"``, never ``"0.37500000000000004"``.
    """
    return format(compression_ratio(retention), "g")


def retention_tag(retention: float) -> str:
    """Shared naming contract: ``r0.5``, ``r0.625``, ``r0.75`` (``:g`` format)."""
    return f"r{retention:g}"


def build_prune_command(spec: SweepSpec, retention: float, dataset_path: Path | str) -> list[str]:
    """The exact REAP invocation, as an argv list (run with cwd = reap repo root).

    ``dataset_path`` may be a real local folder (local profile) or a literal
    shell variable like ``"$DATASET"`` (remote script) -- it is passed through
    verbatim, never normalized, so bash variables survive on Windows.

    ``--record_pruning_metrics_only false`` is set explicitly: we want the
    pruned checkpoint saved, not just the saliency metrics.
    """
    seed = spec.seeds[0] if spec.seeds else 42
    return [
        "python",
        "src/reap/prune.py",
        "--model-name",
        spec.model_id,
        "--dataset-name",
        str(dataset_path),
        "--compression-ratio",
        format_ratio(retention),
        "--prune-method",
        "reap",
        "--seed",
        str(seed),
        "--distance_measure",
        "cosine",
        "--record_pruning_metrics_only",
        "false",
    ]


def calib_to_dataset_dir(calibration_jsonl: Path, out_dir: Path) -> Path:
    """Convert our prompt-only ``calibration.jsonl`` (CalibrationRecord lines)
    into a dataset *folder* REAP can consume via ``load_dataset(folder)``.

    The folder contains ``data.jsonl`` where every row has a ``messages``
    column: ``[{"role": "user", "content": <prompt>}]`` (chat-format datasets
    need a messages column per the research brief).

    Returns ``out_dir``. Raises :class:`PruneError` if the calibration file is
    missing or empty -- an empty dataset would make REAP's calibration pass
    meaningless.
    """
    calibration_jsonl = Path(calibration_jsonl)
    if not calibration_jsonl.exists():
        raise PruneError(
            f"Calibration file not found: {calibration_jsonl}\n"
            "Generate it first (reap-lab generate) or point spec.calibration at an existing JSONL."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / DATASET_FILENAME
    tmp = target.with_suffix(target.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for rec in iter_jsonl(calibration_jsonl, CalibrationRecord):
            row = {"messages": [{"role": "user", "content": rec.prompt}]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise PruneError(
            f"Calibration file is empty: {calibration_jsonl}\n"
            "REAP needs calibration prompts to observe router activations. "
            "Re-run dataset generation (reap-lab generate)."
        )
    tmp.replace(target)
    return out_dir
