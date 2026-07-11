"""C3 evaluation harness: runners, scorers, perf capture, and evaluate_artifact.

Public surface:
    evaluate_artifact(...)    -- score one artifact end to end (shared summary contract)
    MockRunner / OpenAICompatRunner / LlamaServerRunner / ModelRunner / RunnerResponse
    SCORER_REGISTRY / get_scorer -- TaskType -> scorer mapping
    capture_perf              -- FR-3.4 perf metrics for one context size
"""

from __future__ import annotations

from reaplab.evalharness.evaluate import evaluate_artifact
from reaplab.evalharness.perf import capture_perf
from reaplab.evalharness.runners import (
    LlamaServerRunner,
    MockRunner,
    ModelRunner,
    OpenAICompatRunner,
    RunnerError,
    RunnerResponse,
    runner_from_runtime,
)
from reaplab.evalharness.scorers import SCORER_REGISTRY, Scorer, get_scorer

__all__ = [
    "SCORER_REGISTRY",
    "LlamaServerRunner",
    "MockRunner",
    "ModelRunner",
    "OpenAICompatRunner",
    "RunnerError",
    "RunnerResponse",
    "Scorer",
    "capture_perf",
    "evaluate_artifact",
    "get_scorer",
    "runner_from_runtime",
]
