"""StateDB stage/key naming shared with the orchestrator.

Contract (all builders): stage strings render as
``"datagen"``, ``"prune:r<retention:g>"``, ``"convert:<artifact_id>"``,
``"eval:<artifact_id>"``. StateDB stores (stage, key) pairs, so the rendered
string splits at the first colon: ``"prune:r0.5" -> ("prune", "r0.5")``.
"""

from __future__ import annotations

from reaplab.prune.reap_cmd import retention_tag


def prune_stage(retention: float) -> tuple[str, str]:
    """StateDB (stage, key) for one prune: ``("prune", "r0.5")`` etc."""
    return "prune", retention_tag(retention)


def convert_stage(artifact_id: str) -> tuple[str, str]:
    """StateDB (stage, key) for one GGUF artifact: ``("convert", "r0.5-q4_k_m")``."""
    return "convert", artifact_id


def pruned_artifact_id(retention: float, quant: str) -> str:
    """Shared naming contract: ``r<retention:g>-<quant lowercase>``."""
    return f"{retention_tag(retention)}-{quant.lower()}"


def baseline_artifact_id(quant: str) -> str:
    """Shared naming contract: ``baseline-<quant lowercase>``."""
    return f"baseline-{quant.lower()}"
