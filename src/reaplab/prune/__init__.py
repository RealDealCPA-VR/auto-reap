"""C2 -- pruning engine + GGUF conversion (PRD FR-2.1..FR-2.4).

Public surface:
- :func:`build_artifacts` -- per-retention flow: dataset -> prune -> bf16 -> quant grid.
- :func:`build_baseline` -- unpruned baseline GGUFs (``baseline-<quant>``).
- :mod:`reaplab.prune.reap_cmd` -- pure REAP command/dataset builders.
- :mod:`reaplab.prune.profiles` -- mock / local-offload / remote execution.
- :mod:`reaplab.prune.gguf` -- llama.cpp convert + quantize wrappers.

Heavy work (REAP itself, llama.cpp) always runs as an external subprocess or a
generated script; nothing GPU-shaped is ever imported.
"""

from __future__ import annotations

from reaplab.prune.baseline import build_baseline
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
    ExecutionProfile,
    LocalOffloadProfile,
    MockProfile,
    RemoteProfile,
    budget_timeout_seconds,
    build_remote_script,
    get_profile,
    resolve_hf_model_dir,
)
from reaplab.prune.reap_cmd import (
    build_prune_command,
    calib_to_dataset_dir,
    compression_ratio,
    format_ratio,
    retention_tag,
)
from reaplab.prune.runner import build_artifacts, ensure_calibration_dataset, model_slug
from reaplab.prune.stages import (
    baseline_artifact_id,
    convert_stage,
    prune_stage,
    pruned_artifact_id,
)

__all__ = [
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
    "format_ratio",
    "get_profile",
    "model_slug",
    "prune_stage",
    "pruned_artifact_id",
    "quantize",
    "resolve_hf_model_dir",
    "retention_tag",
    "validate_quant",
    "write_fake_gguf",
]
