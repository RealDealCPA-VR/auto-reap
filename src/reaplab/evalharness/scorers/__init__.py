"""Scorer registry: TaskType -> scorer (PRD FR-3.2).

Scorer protocol: score(item: EvalRecord, response: RunnerResponse)
    -> (score 0..1, passed, detail dict).

OPEN_ENDED maps to a cheap non-refusal/length heuristic here; evaluate.py swaps in
the pairwise LLM judge (scorers.judge) whenever baseline responses are available
and the artifact under test is not itself the baseline.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from reaplab.core.records import EvalRecord, TaskType
from reaplab.evalharness.scorers.exact import ExactScorer
from reaplab.evalharness.scorers.json_schema_scorer import JsonSchemaScorer
from reaplab.evalharness.scorers.refusal import (
    RefusalBenignScorer,
    ShouldRefuseScorer,
    is_refusal,
)
from reaplab.evalharness.scorers.tool_call import ToolCallScorer


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, item: EvalRecord, response: Any) -> tuple[float, bool, dict[str, Any]]:
        """Return (score in [0,1], passed, detail)."""
        ...


class OpenEndedHeuristicScorer:
    """Judge-free fallback for open-ended items: used for the baseline itself and
    whenever no baseline responses exist yet. A non-refusal answer of plausible
    length earns 0.75 (flagged "unjudged" so reports can tell it apart from real
    judged scores); refusals and near-empty answers earn 0."""

    name = "open_ended_heuristic"
    MIN_LENGTH = 20

    def score(self, item: EvalRecord, response: Any) -> tuple[float, bool, dict[str, Any]]:
        text = (response.text or "").strip()
        if is_refusal(text):
            return 0.0, False, {"unjudged": True, "refused": True}
        if len(text) < self.MIN_LENGTH:
            return 0.0, False, {"unjudged": True, "too_short": True, "length": len(text)}
        return 0.75, True, {"unjudged": True}


SCORER_REGISTRY: dict[TaskType, Scorer] = {
    TaskType.EXACT: ExactScorer(),
    TaskType.JSON_SCHEMA: JsonSchemaScorer(),
    TaskType.TOOL_CALL: ToolCallScorer(),
    TaskType.REFUSAL_BENIGN: RefusalBenignScorer(),
    TaskType.SHOULD_REFUSE: ShouldRefuseScorer(),
    TaskType.OPEN_ENDED: OpenEndedHeuristicScorer(),
}


def get_scorer(task_type: TaskType) -> Scorer:
    """Scorer for a task type. Raises KeyError with the known types listed."""
    try:
        return SCORER_REGISTRY[task_type]
    except KeyError:
        known = ", ".join(t.value for t in SCORER_REGISTRY)
        raise KeyError(f"no scorer registered for task_type {task_type!r}; known: {known}") from None


__all__ = [
    "SCORER_REGISTRY",
    "ExactScorer",
    "JsonSchemaScorer",
    "OpenEndedHeuristicScorer",
    "RefusalBenignScorer",
    "Scorer",
    "ShouldRefuseScorer",
    "ToolCallScorer",
    "get_scorer",
    "is_refusal",
]
