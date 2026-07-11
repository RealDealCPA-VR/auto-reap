"""C4 — orchestrator: sweep engine, scoring/gates, report, promotion, status.

Public surface (stable for the CLI and for tests):

    run_sweep(spec, resume=True, ..., promote=False) -> Path   # report path
    sweep_status(spec) -> str
    promote_winner(spec, manifest, report_path, workspace) -> PromotionResult
    render_report(...) -> str / write_report(workspace, config_hash, md) -> Path
    weighted_score / quality_retention / domain_regressions
    evaluate_gates -> list[GateResult] / pareto_front / select_winner
"""

from __future__ import annotations

from reaplab.orchestrate.promote import PromotionResult, promote_winner
from reaplab.orchestrate.report import ArtifactRow, render_report, write_report
from reaplab.orchestrate.scoring import (
    SPECIAL_DOMAINS,
    GateResult,
    domain_regressions,
    evaluate_gates,
    pareto_front,
    quality_retention,
    select_winner,
    weighted_score,
)
from reaplab.orchestrate.status import sweep_status
from reaplab.orchestrate.sweep import run_sweep

__all__ = [
    "SPECIAL_DOMAINS",
    "ArtifactRow",
    "GateResult",
    "PromotionResult",
    "domain_regressions",
    "evaluate_gates",
    "pareto_front",
    "promote_winner",
    "quality_retention",
    "render_report",
    "run_sweep",
    "select_winner",
    "sweep_status",
    "weighted_score",
    "write_report",
]
