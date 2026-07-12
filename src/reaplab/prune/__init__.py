"""C2 -- pruning engine + GGUF conversion (PRD FR-2.1..FR-2.4).

Public surface:
- :func:`build_artifacts` -- per-retention flow: dataset -> prune -> bf16 -> quant grid.
- :func:`build_baseline` -- unpruned baseline GGUFs (``baseline-<quant>``).
- :func:`expected_baseline_ids` -- the artifact ids build_baseline WILL produce, so the
  orchestrator can pre-check resume state without guessing (a user-provided
  ``baseline_gguf`` yields one id, not one per quant).
- :mod:`reaplab.prune.reap_cmd` -- pure REAP command/dataset builders.
- :mod:`reaplab.prune.profiles` -- mock / local-offload / remote execution.
- :mod:`reaplab.prune.gguf` -- llama.cpp convert + quantize wrappers.

Everything a sweep writes is namespaced by ``spec.config_hash()``
(``artifacts/<hash>/...``, ``runs/<hash>/...``), so two specs sharing a workspace
never read or clobber each other's artifacts.

Heavy work (REAP itself, llama.cpp) always runs as an external subprocess or a
generated script; nothing GPU-shaped is ever imported.
"""

from __future__ import annotations

from reaplab.prune.baseline import build_baseline, expected_baseline_ids
from reaplab.prune.errors import (
    NeedsManualStep,
    PrerequisiteError,
    PruneError,
    ToolNotFoundError,
)
from reaplab.prune.gguf import (
    CONFIRMED_QUANTS,
    LlamaCppTools,
    convert_to_gguf,
    detect_quant_from_name,
    quantize,
    validate_quant,
    write_fake_gguf,
)
from reaplab.prune.profiles import (
    ALLOW_LOCAL_OFFLOAD_ENV,
    ExecutionProfile,
    LocalOffloadProfile,
    MockProfile,
    RemoteProfile,
    budget_timeout_seconds,
    build_remote_script,
    get_profile,
    redact,
    resolve_hf_model_dir,
)
from reaplab.prune.reap_cmd import (
    build_prune_command,
    calib_to_dataset_dir,
    compression_ratio,
    format_ratio,
    retention_tag,
)
from reaplab.prune.runner import (
    artifacts_dir,
    build_artifacts,
    ensure_calibration_dataset,
    model_slug,
    read_expert_stats,
    validated_quants,
)
from reaplab.prune.stages import (
    baseline_artifact_id,
    convert_stage,
    prune_stage,
    pruned_artifact_id,
)

__all__ = [
    "ALLOW_LOCAL_OFFLOAD_ENV",
    "CONFIRMED_QUANTS",
    "ExecutionProfile",
    "LlamaCppTools",
    "LocalOffloadProfile",
    "MockProfile",
    "NeedsManualStep",
    "PrerequisiteError",
    "PruneError",
    "RemoteProfile",
    "ToolNotFoundError",
    "artifacts_dir",
    "baseline_artifact_id",
    "budget_timeout_seconds",
    "build_artifacts",
    "build_baseline",
    "build_prune_command",
    "build_remote_script",
    "calib_to_dataset_dir",
    "compression_ratio",
    "convert_stage",
    "convert_to_gguf",
    "detect_quant_from_name",
    "ensure_calibration_dataset",
    "expected_baseline_ids",
    "format_ratio",
    "get_profile",
    "model_slug",
    "prune_stage",
    "pruned_artifact_id",
    "quantize",
    "read_expert_stats",
    "redact",
    "resolve_hf_model_dir",
    "retention_tag",
    "validate_quant",
    "validated_quants",
    "write_fake_gguf",
]
