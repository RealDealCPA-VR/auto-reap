"""Scoring, gates, Pareto, and winner-selection unit tests (pure functions)."""

from __future__ import annotations

import pytest

from reaplab.core.config import Gates
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

GATE_NAMES = {
    "quality_retention",
    "domain_regression",
    "vram",
    "false_refusal",
    "should_refuse",
    "tool_call_validity",
    "decode_tps",
}


def by_name(results: list[GateResult]) -> dict[str, GateResult]:
    return {g.name: g for g in results}


def assert_only_failed(results: list[GateResult], *names: str) -> None:
    failed = {g.name for g in results if not g.passed}
    assert failed == set(names), f"expected only {names} to fail, got {failed}"


# -- weighted_score ------------------------------------------------------------


def test_weighted_score_respects_weights_and_skips_specials(pack):
    scores = {
        "alpha": 1.0,
        "beta": 0.5,
        "benign_sensitive": 0.0,  # special: must be skipped
        "should_refuse": 0.0,  # special: must be skipped
        "ghost": 0.0,  # not in the pack: must be skipped
    }
    # alpha weight 3, beta weight 1 -> (3*1.0 + 1*0.5) / 4 = 0.875
    assert weighted_score(scores, pack) == pytest.approx(0.875)


def test_weighted_score_renormalizes_over_present_domains(pack):
    # only alpha measured: its weight renormalizes to 1.0
    assert weighted_score({"alpha": 0.8}, pack) == pytest.approx(0.8)


def test_weighted_score_empty_and_special_only(pack):
    assert weighted_score({}, pack) == 0.0
    assert weighted_score({"benign_sensitive": 1.0, "should_refuse": 1.0}, pack) == 0.0


def test_special_domains_are_the_contract_names():
    assert SPECIAL_DOMAINS == {"benign_sensitive", "should_refuse"}


# -- quality_retention -----------------------------------------------------------


def test_quality_retention_ratio():
    assert quality_retention(0.86, 0.875) == pytest.approx(0.86 / 0.875)


def test_quality_retention_guards_division_by_zero():
    assert quality_retention(0.5, 0.0) == 1.0  # degenerate baseline: no regression possible
    assert quality_retention(0.0, 0.0) == 1.0
    assert quality_retention(-0.1, 0.0) == 0.0


# -- domain_regressions ------------------------------------------------------------


def test_domain_regressions_points_scale_and_shared_only():
    candidate = {"alpha": 0.85, "beta": 0.90}
    baseline = {"alpha": 0.90, "beta": 0.85, "gamma": 0.5}
    regs = domain_regressions(candidate, baseline)
    assert regs["alpha"] == pytest.approx(5.0)
    assert regs["beta"] == pytest.approx(-5.0)  # improvement is negative
    assert "gamma" not in regs  # not shared


# -- evaluate_gates -----------------------------------------------------------------


@pytest.fixture
def baseline(summary_factory):
    return summary_factory("baseline-q4_k_m", alpha=0.90, beta=0.80, false_refusal=0.02)


def test_all_gates_pass(pack, gates, baseline, summary_factory):
    candidate = summary_factory("r0.75-q4_k_m", alpha=0.88, beta=0.80, false_refusal=0.01)
    results = evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack)
    assert {g.name for g in results} == GATE_NAMES
    assert all(g.passed for g in results)
    # blocking flags: decode_tps is the only advisory gate
    assert {g.name for g in results if not g.blocking} == {"decode_tps"}


def test_quality_retention_gate_flips_only_itself(pack, gates, baseline, summary_factory):
    # weighted 0.826 vs 0.875 -> 94.4% (< 95%), but no single domain drops > 5 pts
    candidate = summary_factory("r0.5-q4_k_m", alpha=0.851, beta=0.751, false_refusal=0.01)
    results = evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack)
    assert_only_failed(results, "quality_retention")
    gate = by_name(results)["quality_retention"]
    assert gate.value == pytest.approx(0.826 / 0.875, rel=1e-3)
    assert gate.blocking


def test_domain_regression_gate_flips_only_itself(pack, gates, baseline, summary_factory):
    # beta drops 6 pts (> 5), but weighted retention stays above 95%
    candidate = summary_factory("r0.5-q4_k_m", alpha=0.90, beta=0.74, false_refusal=0.01)
    results = evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack)
    assert_only_failed(results, "domain_regression")
    gate = by_name(results)["domain_regression"]
    assert gate.value == pytest.approx(6.0)
    assert "beta" in gate.note


def test_vram_gate_flips_only_itself(pack, gates, baseline, summary_factory):
    candidate = summary_factory(
        "r0.5-q4_k_m", alpha=0.88, beta=0.80, false_refusal=0.01, vram_mb=45 * 1024
    )
    results = evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack)
    assert_only_failed(results, "vram")
    gate = by_name(results)["vram"]
    assert gate.value == pytest.approx(45.0)
    assert gate.limit == pytest.approx(40.0)


def test_false_refusal_absolute_limit(pack, gates, summary_factory):
    baseline = summary_factory("baseline-q4_k_m", alpha=0.90, beta=0.80, false_refusal=0.06)
    candidate = summary_factory("r0.5-q4_k_m", alpha=0.88, beta=0.80, false_refusal=0.05)
    # 5% is under the baseline's 6% but above the 2% absolute limit -> fail
    results = evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack)
    assert_only_failed(results, "false_refusal")


def test_false_refusal_fails_above_baseline_even_under_absolute_limit(
    pack, gates, summary_factory
):
    baseline = summary_factory("baseline-q4_k_m", alpha=0.90, beta=0.80, false_refusal=0.01)
    candidate = summary_factory("r0.5-q4_k_m", alpha=0.88, beta=0.80, false_refusal=0.015)
    results = evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack)
    assert_only_failed(results, "false_refusal")
    assert "baseline" in by_name(results)["false_refusal"].note


def test_should_refuse_hard_fails_at_99_percent(pack, gates, baseline, summary_factory):
    candidate = summary_factory(
        "r0.5-q4_k_m", alpha=0.88, beta=0.80, false_refusal=0.01, should_refuse=0.99
    )
    results = evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack)
    assert_only_failed(results, "should_refuse")
    gate = by_name(results)["should_refuse"]
    assert gate.blocking
    assert gate.value == pytest.approx(0.99)
    assert gate.limit == pytest.approx(1.0)


def test_tool_call_validity_gate(pack, gates, baseline, summary_factory):
    candidate = summary_factory(
        "r0.5-q4_k_m", alpha=0.88, beta=0.80, false_refusal=0.01, tool_validity=0.97
    )
    results = evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack)
    assert_only_failed(results, "tool_call_validity")


def test_decode_tps_advisory_never_blocks(pack, baseline, summary_factory):
    gates = Gates(min_decode_tps=100.0)
    candidate = summary_factory("r0.5-q4_k_m", alpha=0.88, beta=0.80, false_refusal=0.01, tps=60.0)
    results = evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack)
    gate = by_name(results)["decode_tps"]
    assert not gate.passed
    assert not gate.blocking
    assert "advisory" in gate.note
    # every blocking gate still passes -> the candidate remains promotable
    assert all(g.passed for g in results if g.blocking)


def test_decode_tps_without_configured_minimum_passes(pack, gates, baseline, summary_factory):
    candidate = summary_factory("r0.5-q4_k_m", alpha=0.88, beta=0.80, tps=1.0)
    gate = by_name(evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack))[
        "decode_tps"
    ]
    assert gate.passed and not gate.blocking


# -- not-measured semantics ------------------------------------------------------------


def test_vram_not_measured_passes_with_note(pack, gates, baseline, summary_factory):
    candidate = summary_factory("r0.5-q4_k_m", alpha=0.88, beta=0.80, vram_mb=None)
    gate = by_name(evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack))["vram"]
    assert gate.passed
    assert gate.value is None
    assert "not measured" in gate.note


def test_vram_measured_path_reports_gb(pack, gates, baseline, summary_factory, perf_factory):
    candidate = summary_factory("r0.5-q4_k_m", alpha=0.88, beta=0.80, vram_mb=30000.0)
    gate = by_name(evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack))["vram"]
    assert gate.passed
    assert gate.value == pytest.approx(30000.0 / 1024.0)


def test_vram_missing_context_entry_is_not_measured(pack, gates, baseline, summary_factory, perf_factory):
    # perf only captured at 4k; the gate context is 32k
    candidate = summary_factory(
        "r0.5-q4_k_m", alpha=0.88, beta=0.80, perf=perf_factory(contexts=(4096,))
    )
    gate = by_name(evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack))["vram"]
    assert gate.passed
    assert "not measured" in gate.note


def test_rate_gates_not_measured(pack, gates, baseline, summary_factory):
    candidate = summary_factory(
        "r0.5-q4_k_m",
        alpha=0.88,
        beta=0.80,
        false_refusal=None,
        should_refuse=None,
        tool_validity=None,
    )
    named = by_name(evaluate_gates(gates, candidate, baseline, candidate["perf"], pack=pack))
    for name in ("false_refusal", "should_refuse", "tool_call_validity"):
        assert named[name].passed, name
        assert "not measured" in named[name].note, name


def test_no_baseline_makes_relative_gates_not_measured(pack, gates, summary_factory):
    candidate = summary_factory("r0.5-q4_k_m", alpha=0.88, beta=0.80, false_refusal=0.01)
    named = by_name(evaluate_gates(gates, candidate, None, candidate["perf"], pack=pack))
    assert named["quality_retention"].passed
    assert "not measured" in named["quality_retention"].note
    assert named["domain_regression"].passed
    assert "not measured" in named["domain_regression"].note
    # absolute-only false refusal check still applies
    assert named["false_refusal"].passed
    assert "absolute limit only" in named["false_refusal"].note


# -- pareto_front ------------------------------------------------------------------------


def test_pareto_front_excludes_dominated():
    points = [
        {"artifact_id": "a", "quality": 0.9, "vram": 30.0, "tps": 60.0},
        {"artifact_id": "b", "quality": 0.85, "vram": 25.0, "tps": 55.0},
        {"artifact_id": "c", "quality": 0.8, "vram": 35.0, "tps": 50.0},  # dominated by a
    ]
    assert pareto_front(points) == {"a", "b"}


def test_pareto_none_is_worst_but_other_axes_can_still_win():
    points = [
        {"artifact_id": "a", "quality": 0.9, "vram": 30.0, "tps": 60.0},
        {"artifact_id": "d", "quality": None, "vram": 20.0, "tps": 100.0},
    ]
    assert pareto_front(points) == {"a", "d"}


def test_pareto_all_none_point_is_dominated_by_any_measured_point():
    points = [
        {"artifact_id": "a", "quality": 0.9, "vram": 30.0, "tps": 60.0},
        {"artifact_id": "e", "quality": None, "vram": None, "tps": None},
    ]
    assert pareto_front(points) == {"a"}


def test_pareto_single_candidate_never_excluded():
    only = [{"artifact_id": "only", "quality": None, "vram": None, "tps": None}]
    assert pareto_front(only) == {"only"}
    assert pareto_front([]) == set()


# -- select_winner -------------------------------------------------------------------------


def _pass_gates() -> list[GateResult]:
    return [GateResult(name="g", passed=True)]


def _fail_blocking() -> list[GateResult]:
    return [GateResult(name="g", passed=False)]


def _fail_advisory() -> list[GateResult]:
    return [GateResult(name="g", passed=False, blocking=False)]


def test_select_winner_excludes_gate_failers_even_with_top_score():
    winner = select_winner(
        [("top-score", 0.95, _fail_blocking()), ("runner-up", 0.85, _pass_gates())]
    )
    assert winner == "runner-up"


def test_select_winner_none_when_all_fail():
    assert select_winner([("a", 0.9, _fail_blocking()), ("b", 0.8, _fail_blocking())]) is None
    assert select_winner([]) is None


def test_select_winner_ignores_advisory_failures():
    assert select_winner([("a", 0.7, _fail_advisory())]) == "a"


def test_select_winner_picks_highest_weighted_among_passing():
    winner = select_winner(
        [("low", 0.5, _pass_gates()), ("high", 0.9, _pass_gates()), ("mid", 0.7, _pass_gates())]
    )
    assert winner == "high"
