"""Near-dup + leakage filter: both backends, planted duplicates, error messages."""

from __future__ import annotations

import pytest

from reaplab.core.config import ProviderCfg
from reaplab.core.providers import LLMProvider, LLMResponse
from reaplab.core.providers.mock import MockProvider
from reaplab.core.records import CalibrationRecord, EvalRecord, TaskType
from reaplab.datagen.dedup import (
    REASON_LEAKS_CALIBRATION,
    REASON_WITHIN_EVAL,
    filter_near_duplicates,
)


def _ev(id: str, prompt: str) -> EvalRecord:
    return EvalRecord(id=id, domain="d", prompt=prompt, task_type=TaskType.OPEN_ENDED)


def _cal(id: str, prompt: str) -> CalibrationRecord:
    return CalibrationRecord(id=id, domain="d", prompt=prompt)


DISTINCT = [
    "Categorize the March payment to Blue Harbor LLC for office chairs, invoice 4471.",
    "Summarize the quarterly utility spending trend for Cascade Group in Tacoma.",
    "Draft a reminder about the September filing deadline for Ironwood Partners.",
]


class _NoEmbedProvider(LLMProvider):
    """Provider that cannot embed (embed() -> None from the base class)."""

    name = "no-embed"

    def complete(self, prompt, *, system=None, temperature=None, max_tokens=None, json_mode=False):
        return LLMResponse(text="unused")


# ---------------------------------------------------------------------------
# fuzzy backend
# ---------------------------------------------------------------------------


def test_fuzzy_drops_planted_within_eval_dupe():
    records = [
        _ev("ev-000001", DISTINCT[0]),
        # same token set, reordered -> token_set_ratio ~ 1.0
        _ev("ev-000002", "Categorize the March payment for office chairs to Blue Harbor LLC, invoice 4471."),
        _ev("ev-000003", DISTINCT[1]),
    ]
    kept, report = filter_near_duplicates(records, [], threshold=0.90, backend="fuzzy")
    assert [r.id for r in kept] == ["ev-000001", "ev-000003"]
    assert len(report.dropped) == 1
    drop = report.dropped[0]
    assert drop.id == "ev-000002"
    assert drop.reason == REASON_WITHIN_EVAL
    assert drop.similar_to == "ev-000001"
    assert drop.similarity >= 0.90


def test_fuzzy_blocks_leakage_against_any_calibration_item():
    records = [_ev("ev-000001", DISTINCT[0]), _ev("ev-000002", DISTINCT[1])]
    cal = [
        _cal("cal-000001", DISTINCT[2]),
        _cal("cal-000002", "Summarize the utility spending trend for Cascade Group in Tacoma, quarterly."),
    ]
    kept, report = filter_near_duplicates(records, cal, threshold=0.90, backend="fuzzy")
    assert [r.id for r in kept] == ["ev-000001"]
    drop = report.dropped[0]
    assert drop.id == "ev-000002"
    assert drop.reason == REASON_LEAKS_CALIBRATION
    assert drop.similar_to == "cal-000002"


def test_fuzzy_keeps_distinct_items_and_reports_counts():
    records = [_ev(f"ev-{i:06d}", p) for i, p in enumerate(DISTINCT, 1)]
    cal = [_cal("cal-000001", "Explain the estimated tax payment schedule for a new S corporation.")]
    kept, report = filter_near_duplicates(records, cal, threshold=0.90, backend="fuzzy")
    assert len(kept) == 3
    assert report.eval_in == 3 and report.eval_kept == 3
    assert report.calibration_count == 1
    assert report.backend == "fuzzy" and report.threshold == 0.90
    assert report.dropped == []


def test_kept_order_is_stable():
    records = [_ev(f"ev-{i:06d}", p) for i, p in enumerate(DISTINCT, 1)]
    kept, _ = filter_near_duplicates(records, [], threshold=0.90, backend="fuzzy")
    assert [r.id for r in kept] == ["ev-000001", "ev-000002", "ev-000003"]


# ---------------------------------------------------------------------------
# embedding backend (offline via MockProvider.embed)
# ---------------------------------------------------------------------------


def test_embedding_drops_exact_dupe_modulo_case_and_whitespace():
    provider = MockProvider(ProviderCfg(kind="mock"))
    records = [
        _ev("ev-000001", DISTINCT[0]),
        _ev("ev-000002", "  categorize the MARCH payment to Blue Harbor   LLC for office chairs, invoice 4471. "),
        _ev("ev-000003", DISTINCT[1]),
    ]
    kept, report = filter_near_duplicates(
        records, [], threshold=0.90, backend="embedding", provider=provider
    )
    assert [r.id for r in kept] == ["ev-000001", "ev-000003"]
    assert report.dropped[0].reason == REASON_WITHIN_EVAL
    assert report.dropped[0].similarity == pytest.approx(1.0)


def test_embedding_blocks_leakage():
    provider = MockProvider(ProviderCfg(kind="mock"))
    records = [_ev("ev-000001", DISTINCT[0]), _ev("ev-000002", DISTINCT[1])]
    cal = [_cal("cal-000001", DISTINCT[1].upper())]  # same text after normalization
    kept, report = filter_near_duplicates(
        records, cal, threshold=0.90, backend="embedding", provider=provider
    )
    assert [r.id for r in kept] == ["ev-000001"]
    assert report.dropped[0].reason == REASON_LEAKS_CALIBRATION


def test_embedding_requires_a_provider():
    with pytest.raises(ValueError, match="embedding_provider"):
        filter_near_duplicates([_ev("ev-000001", DISTINCT[0])], [], backend="embedding")


def test_embedding_rejects_provider_that_cannot_embed():
    provider = _NoEmbedProvider(ProviderCfg(kind="mock"))
    with pytest.raises(ValueError, match="cannot embed"):
        filter_near_duplicates(
            [_ev("ev-000001", DISTINCT[0])], [], backend="embedding", provider=provider
        )


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        filter_near_duplicates([], [], backend="cosine")  # type: ignore[arg-type]


def test_bad_threshold_rejected():
    with pytest.raises(ValueError, match="threshold"):
        filter_near_duplicates([], [], threshold=1.5)
