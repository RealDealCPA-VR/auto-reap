"""Near-duplicate and leakage filtering (PRD FR-1.3).

Two-phase filter over the EVAL set (calibration is never modified):

1. WITHIN-EVAL  -- drop any eval item whose prompt is a near-duplicate of an
   earlier kept eval item (first occurrence wins; ordering is the stable id order).
2. LEAKAGE      -- drop any surviving eval item too similar to ANY calibration
   item (held-out guarantee: zero overlap with calibration).

Backends:

- ``fuzzy``     -- rapidfuzz ``token_set_ratio / 100 >= threshold``. No extra deps,
  fast enough for 2000 x 500 comparisons.
- ``embedding`` -- ``provider.embed`` + cosine ``>= threshold`` (PRD's cosine >= 0.90
  filter). Works offline through MockProvider.embed; online through any
  OpenAI-compatible ``/v1/embeddings`` endpoint (e.g. LM Studio).

Comparisons run on ``comparable_text`` (embedded synthetic long documents are
stripped first) so long-context items are judged on their task text, not on
shared filler paragraphs. For records that carry a ``json_schema``, the schema's
canonical JSON is also stripped from the prompt before comparison: every item of
a json_schema domain legitimately repeats the same schema block, and that shared
boilerplate must not read as similarity.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from reaplab.core.hashing import canonical_json
from reaplab.core.providers import LLMProvider
from reaplab.core.records import CalibrationRecord, EvalRecord
from reaplab.datagen.synthdoc import comparable_text

Backend = Literal["fuzzy", "embedding"]

REASON_WITHIN_EVAL = "near_dup_within_eval"
REASON_LEAKS_CALIBRATION = "leaks_calibration"


class DroppedItem(BaseModel):
    """One eval record removed by the filter, with its closest match."""

    id: str
    reason: Literal["near_dup_within_eval", "leaks_calibration"]
    similar_to: str  # id of the kept eval / calibration record it collided with
    similarity: float


class DedupReport(BaseModel):
    """Audit trail for one filter run; serialized to dedup_report_v1.json."""

    backend: str
    threshold: float
    eval_in: int
    eval_kept: int
    calibration_count: int
    dropped: list[DroppedItem] = Field(default_factory=list)


def _record_text(record: CalibrationRecord | EvalRecord) -> str:
    """Comparison text for one record: prompt minus its own known boilerplate
    (embedded long documents; the domain's repeated schema block), normalized."""
    prompt = record.prompt
    schema = getattr(record, "json_schema", None)
    if schema:
        prompt = prompt.replace(canonical_json(schema), " ")
    return comparable_text(prompt)


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db) if da and db else 0.0


def _embed_all(provider: LLMProvider, texts: list[str], chunk: int = 128) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), chunk):
        part = provider.embed(texts[i : i + chunk])
        if part is None:
            raise ValueError(
                "dedup_backend='embedding' needs an embedding-capable provider, but "
                f"{provider.name!r} cannot embed. Set data.embedding_provider to an "
                "openai-compat endpoint with an embedding model loaded (e.g. LM Studio at "
                "http://localhost:1234/v1), or switch data.dedup_backend to 'fuzzy'."
            )
        out.extend(part)
    return out


def filter_near_duplicates(
    eval_records: list[EvalRecord],
    calibration_records: list[CalibrationRecord],
    *,
    threshold: float = 0.90,
    backend: Backend = "fuzzy",
    provider: LLMProvider | None = None,
) -> tuple[list[EvalRecord], DedupReport]:
    """Filter the eval set: intra-eval near-dups first, then leakage vs. calibration.

    Returns (kept eval records in original order, report). Raises ValueError with
    a fix-it message when backend == "embedding" but no embedding-capable provider
    was supplied.
    """
    if backend not in ("fuzzy", "embedding"):
        raise ValueError(f"unknown dedup backend {backend!r}; use 'fuzzy' or 'embedding'")
    if not 0 < threshold <= 1:
        raise ValueError(f"near_dup_threshold must be in (0, 1], got {threshold}")

    eval_texts = [_record_text(r) for r in eval_records]
    cal_texts = [_record_text(r) for r in calibration_records]

    if backend == "embedding":
        if provider is None:
            raise ValueError(
                "dedup_backend='embedding' requires a provider. Set data.embedding_provider "
                "in your sweep YAML (an openai-compat endpoint with an embedding model), or "
                "switch data.dedup_backend to 'fuzzy'."
            )
        eval_vecs = _embed_all(provider, eval_texts)
        cal_vecs = _embed_all(provider, cal_texts)

        def similarity(i: int, j: int, against_cal: bool) -> float:
            other = cal_vecs[j] if against_cal else eval_vecs[j]
            return _cosine(eval_vecs[i], other)

    else:
        cutoff = threshold * 100.0

        def similarity(i: int, j: int, against_cal: bool) -> float:
            other = cal_texts[j] if against_cal else eval_texts[j]
            return fuzz.token_set_ratio(eval_texts[i], other, score_cutoff=cutoff) / 100.0

    dropped: list[DroppedItem] = []

    # Phase 1: within-eval, first occurrence wins.
    kept_idx: list[int] = []
    for i in range(len(eval_records)):
        hit: tuple[int, float] | None = None
        for j in kept_idx:
            sim = similarity(i, j, against_cal=False)
            if sim >= threshold:
                hit = (j, sim)
                break
        if hit is None:
            kept_idx.append(i)
        else:
            dropped.append(
                DroppedItem(
                    id=eval_records[i].id,
                    reason=REASON_WITHIN_EVAL,
                    similar_to=eval_records[hit[0]].id,
                    similarity=round(hit[1], 4),
                )
            )

    # Phase 2: leakage against ANY calibration item.
    final_idx: list[int] = []
    for i in kept_idx:
        leak: tuple[int, float] | None = None
        for j in range(len(calibration_records)):
            sim = similarity(i, j, against_cal=True)
            if sim >= threshold:
                leak = (j, sim)
                break
        if leak is None:
            final_idx.append(i)
        else:
            dropped.append(
                DroppedItem(
                    id=eval_records[i].id,
                    reason=REASON_LEAKS_CALIBRATION,
                    similar_to=calibration_records[leak[0]].id,
                    similarity=round(leak[1], 4),
                )
            )

    kept = [eval_records[i] for i in final_idx]
    report = DedupReport(
        backend=backend,
        threshold=threshold,
        eval_in=len(eval_records),
        eval_kept=len(kept),
        calibration_count=len(calibration_records),
        dropped=dropped,
    )
    return kept, report
