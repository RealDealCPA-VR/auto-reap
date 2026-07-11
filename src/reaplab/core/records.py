"""Record schemas for datasets, results, manifests. JSONL-serialized via core.jsonl.

Versioned per PRD FR-1.6: every record carries schema_version.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class TaskType(str, enum.Enum):
    """Drives which scorer evaluates an item (PRD FR-3.2)."""

    EXACT = "exact"  # normalized string match against gold
    JSON_SCHEMA = "json_schema"  # response must parse and validate against item's json_schema
    OPEN_ENDED = "open_ended"  # pairwise LLM judge vs. baseline, rubric-guided
    TOOL_CALL = "tool_call"  # response must contain schema-valid tool call(s)
    REFUSAL_BENIGN = "refusal_benign"  # benign-but-sensitive: model must NOT refuse
    SHOULD_REFUSE = "should_refuse"  # control set: model MUST refuse


class Difficulty(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CalibrationRecord(BaseModel):
    """Prompt-only calibration item (PRD FR-1.2): REAP observes router gates on forward
    passes; no gold response is needed or stored."""

    id: str
    domain: str
    prompt: str
    tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    source: str = "synthetic"
    schema_version: str = SCHEMA_VERSION


class EvalRecord(BaseModel):
    """Held-out scored eval item (PRD FR-1.3). Exactly one of gold/rubric/json_schema/tools
    is typically set, depending on task_type."""

    id: str
    domain: str
    prompt: str
    task_type: TaskType
    gold: str | None = None
    rubric: str | None = None
    json_schema: dict[str, Any] | None = None  # for JSON_SCHEMA tasks
    tools: list[dict[str, Any]] | None = None  # OpenAI-format tool defs, for TOOL_CALL tasks
    expected_tool: str | None = None  # tool a correct answer must call (optional)
    max_tokens: int = 512
    tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    source: str = "synthetic"
    schema_version: str = SCHEMA_VERSION


class ItemResult(BaseModel):
    """One eval item scored against one artifact. Appended to runs/<hash>/results.jsonl."""

    item_id: str
    artifact_id: str
    domain: str
    task_type: TaskType
    response: str
    score: float  # 0..1
    passed: bool
    scorer: str
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    schema_version: str = SCHEMA_VERSION


class PerfMetrics(BaseModel):
    """Runtime performance capture for one artifact at one context size (PRD FR-3.4)."""

    context: int
    load_time_s: float | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    peak_vram_mb: float | None = None
    schema_version: str = SCHEMA_VERSION


class ArtifactManifest(BaseModel):
    """Per-artifact provenance (PRD FR-2.3): every artifact traceable to its config."""

    artifact_id: str  # short stable id, e.g. "r0.50-q4_k_m" or "baseline-q4_k_m"
    kind: str  # "baseline" | "pruned_hf" | "gguf"
    model_id: str
    retention: float | None = None  # None for unpruned baseline
    quant: str | None = None  # None for HF checkpoints
    path: str
    config_hash: str
    artifact_hash: str | None = None  # streamed content hash once materialized
    reap_commit: str | None = None
    retained_expert_map: dict[str, list[int]] | None = None
    saliency_stats: dict[str, Any] | None = None
    wall_clock_s: float | None = None
    peak_mem_gb: float | None = None
    versions: dict[str, str] = Field(default_factory=dict)  # library/tool versions
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = SCHEMA_VERSION
