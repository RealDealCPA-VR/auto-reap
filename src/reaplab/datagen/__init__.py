"""C1 — dataset generator: pack-driven calibration/eval generation (PRD FR-1.x).

Public API:

- ``generate_datasets(spec, workspace, provider=None, state=None)`` — the whole
  pipeline: plan -> generate (procedural for mock, batched LLM otherwise) ->
  near-dup/leakage filter -> JSONL emit + dedup report + human-audit sample.
  Datasets are per-sweep: they live under ``workspace.data_dir(config_hash)`` and
  are reused as-is whenever both files already exist there (contract C1).
- ``dataset_paths(spec, workspace)`` / ``audit_sample_path(spec, workspace)`` /
  ``dedup_report_path(spec, workspace)`` — where those files are, without generating.
- ``plan_counts(pack, data, seed=...)`` — the per-domain allocation plan.
- ``filter_near_duplicates(...)`` — the FR-1.3 filter, usable standalone.
- ``write_audit_sample(...)`` — the PRD M1 stratified review sample.
"""

from __future__ import annotations

from reaplab.datagen.audit import write_audit_sample
from reaplab.datagen.dedup import DedupReport, DroppedItem, filter_near_duplicates
from reaplab.datagen.pipeline import (
    AUDIT_SAMPLE_FILENAME,
    CALIBRATION_FILENAME,
    DEDUP_REPORT_FILENAME,
    EVAL_FILENAME,
    audit_sample_path,
    dataset_dir,
    dataset_paths,
    dedup_report_path,
    generate_datasets,
)
from reaplab.datagen.planning import (
    BENIGN_SENSITIVE_DOMAIN,
    SHOULD_REFUSE_COUNT,
    SHOULD_REFUSE_DOMAIN,
    DomainAllocation,
    GenerationPlan,
    benign_suite_size,
    plan_counts,
)
from reaplab.datagen.synthdoc import comparable_text, estimate_tokens

__all__ = [
    "AUDIT_SAMPLE_FILENAME",
    "BENIGN_SENSITIVE_DOMAIN",
    "CALIBRATION_FILENAME",
    "DEDUP_REPORT_FILENAME",
    "DedupReport",
    "DomainAllocation",
    "DroppedItem",
    "EVAL_FILENAME",
    "GenerationPlan",
    "SHOULD_REFUSE_COUNT",
    "SHOULD_REFUSE_DOMAIN",
    "audit_sample_path",
    "benign_suite_size",
    "comparable_text",
    "dataset_dir",
    "dataset_paths",
    "dedup_report_path",
    "estimate_tokens",
    "filter_near_duplicates",
    "generate_datasets",
    "plan_counts",
    "write_audit_sample",
]
