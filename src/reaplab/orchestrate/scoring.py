"""Pure scoring and gate logic (PRD section 5, FR-4.3).

Everything in this module is a side-effect-free function over plain data:
the `evaluate_artifact` summary dicts produced by the eval harness, the
`DomainPack` weights, and the `Gates` limits from the sweep spec. No IO,
no subprocesses — fully unit-testable offline.

Summary dict contract (producer: reaplab.evalharness.evaluate_artifact):
    {
      "artifact_id": str,
      "domain_scores": {domain: float 0..1},
      "counts": {domain: int},
      "false_refusal_rate": float | None,
      "should_refuse_pass_rate": float | None,
      "tool_call_validity": float | None,
      "perf": {str(context): PerfMetrics.model_dump()},
      "items_scored": int,
    }

Special domains "benign_sensitive" and "should_refuse" never contribute to
the weighted quality score; they feed the refusal gates instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from reaplab.core.config import DomainPack, Gates

#: Domains excluded from weighted quality scoring (they feed dedicated gates).
SPECIAL_DOMAINS: frozenset[str] = frozenset({"benign_sensitive", "should_refuse"})


class GateResult(BaseModel):
    """Outcome of one promotion gate for one candidate artifact.

    ``passed=True`` with a "not measured" note means the input needed by the
    gate was unavailable (e.g. VRAM on a mock runtime) — the gate does not
    block on missing data, it blocks on measured violations.
    ``blocking=False`` marks advisory gates that never veto promotion.
    """

    name: str
    value: float | None = None
    limit: float | None = None
    passed: bool
    blocking: bool = True
    note: str = ""


def weighted_score(domain_scores: Mapping[str, float], pack: DomainPack) -> float:
    """Weighted quality score in 0..1 over the pack's quality domains.

    Weights are the pack's domain weights renormalized over the domains that
    are actually present in ``domain_scores`` (so an unmeasured domain does
    not silently drag the score to zero). Special refusal domains and domains
    absent from the pack are skipped. Returns 0.0 when no quality domain is
    shared between the scores and the pack.
    """
    weights = {d.name: d.weight for d in pack.domains}
    shared = {
        name: score
        for name, score in domain_scores.items()
        if name in weights and name not in SPECIAL_DOMAINS
    }
    if not shared:
        return 0.0
    total_weight = sum(weights[name] for name in shared)
    return sum(weights[name] * score for name, score in shared.items()) / total_weight


def quality_retention(candidate_ws: float, baseline_ws: float) -> float:
    """candidate / baseline weighted score, guarding division by zero.

    A degenerate baseline (score <= 0) cannot be regressed against: any
    candidate at or above it retains everything (1.0); a candidate below it
    retains nothing (0.0).
    """
    if baseline_ws <= 0:
        return 1.0 if candidate_ws >= baseline_ws else 0.0
    return candidate_ws / baseline_ws


def domain_regressions(
    candidate: Mapping[str, float], baseline: Mapping[str, float]
) -> dict[str, float]:
    """Per-domain drop vs baseline in points on a 0-100 scale.

    Positive = the candidate regressed; negative = it improved. Only domains
    present in BOTH score maps are compared (including special domains — the
    regression *gate* filters those out, but the raw diff is useful in the
    report).
    """
    shared = sorted(candidate.keys() & baseline.keys())
    return {d: (baseline[d] - candidate[d]) * 100.0 for d in shared}


def _normalize_perf(perf: Mapping[Any, Any] | None) -> dict[str, dict[str, Any]]:
    """Coerce a perf mapping to {str(context): plain dict} regardless of
    whether values arrived as PerfMetrics instances or dumped dicts."""
    out: dict[str, dict[str, Any]] = {}
    for key, value in (perf or {}).items():
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        out[str(key)] = dict(value)
    return out


def _ctx_order(key: str) -> int:
    return int(key) if key.isdigit() else -1


def _weighted_of(summary: Mapping[str, Any] | None, pack: DomainPack | None) -> float | None:
    """Weighted score for a summary: explicit "weighted_score" key wins, else
    compute from domain_scores via the pack, else an unweighted mean of the
    quality domains. None when nothing is computable."""
    if summary is None:
        return None
    explicit = summary.get("weighted_score")
    if explicit is not None:
        return float(explicit)
    scores = summary.get("domain_scores") or {}
    if pack is not None:
        return weighted_score(scores, pack)
    quality = [s for d, s in scores.items() if d not in SPECIAL_DOMAINS]
    if not quality:
        return None
    return sum(quality) / len(quality)


def _vram_gate(
    gates: Gates, perf: Mapping[str, Mapping[str, Any]], ctx_entry: Mapping[str, Any]
) -> GateResult:
    """Peak-VRAM blocker (PRD section 5) with a physics-aware fallback.

    Peak VRAM is monotonically non-decreasing in context length (the KV cache
    grows with it), so when ``gates.min_context`` itself was never measured the
    largest measured context still carries information:

    - measured context BELOW min_context -> its peak is a LOWER BOUND on the
      peak at min_context. Already over the limit => conclusive FAIL. Under the
      limit => inconclusive, so the gate passes with a lower-bound note.
    - measured context ABOVE min_context -> its peak is an UPPER BOUND. Under
      the limit => conclusive PASS. Over the limit => inconclusive (the gate
      does not block on unmeasured data), pass with a note saying so.

    The old behavior (silently "not measured", always pass) let a candidate that
    had *already* blown the VRAM budget at a smaller context sail through.
    """
    vram_mb = ctx_entry.get("peak_vram_mb")
    if vram_mb is not None:
        vram_gb = float(vram_mb) / 1024.0
        return GateResult(
            name="vram",
            value=vram_gb,
            limit=gates.max_vram_gb,
            passed=vram_gb <= gates.max_vram_gb,
            note=f"peak VRAM at {gates.min_context} context",
        )

    measured = {
        int(key): float(entry["peak_vram_mb"])
        for key, entry in perf.items()
        if key.isdigit() and entry.get("peak_vram_mb") is not None
    }
    if not measured:
        return GateResult(
            name="vram",
            value=None,
            limit=gates.max_vram_gb,
            passed=True,
            note=f"not measured at context {gates.min_context}",
        )

    ctx = max(measured)
    vram_gb = measured[ctx] / 1024.0
    over = vram_gb > gates.max_vram_gb
    if ctx < gates.min_context:
        note = (
            f"not measured at {gates.min_context}; peak {vram_gb:.1f} GB at context {ctx} is a "
            "LOWER BOUND (VRAM grows with context)"
        )
        if over:
            return GateResult(
                name="vram",
                value=vram_gb,
                limit=gates.max_vram_gb,
                passed=False,
                note=note + " — already over the limit, so it cannot fit at "
                f"{gates.min_context}. Conclusive failure.",
            )
        return GateResult(
            name="vram",
            value=vram_gb,
            limit=gates.max_vram_gb,
            passed=True,
            note=note + f" — under the limit there, but the peak at {gates.min_context} is "
            f"unknown. Add {gates.min_context} to runtime.contexts to measure it.",
        )
    # ctx >= min_context: an upper bound on the peak at min_context.
    if not over:
        return GateResult(
            name="vram",
            value=vram_gb,
            limit=gates.max_vram_gb,
            passed=True,
            note=(
                f"not measured at {gates.min_context}; peak {vram_gb:.1f} GB at the LARGER "
                f"context {ctx} is under the limit, so {gates.min_context} fits too"
            ),
        )
    return GateResult(
        name="vram",
        value=vram_gb,
        limit=gates.max_vram_gb,
        passed=True,
        note=(
            f"not measured at {gates.min_context}; peak {vram_gb:.1f} GB at the LARGER context "
            f"{ctx} exceeds the limit, but that is only an upper bound — add {gates.min_context} "
            "to runtime.contexts to measure the gate context directly"
        ),
    )


def evaluate_gates(
    gates: Gates,
    candidate_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any] | None = None,
    perf_by_ctx: Mapping[Any, Any] | None = None,
    *,
    pack: DomainPack | None = None,
) -> list[GateResult]:
    """Evaluate every PRD section-5 promotion gate for one candidate.

    Args:
        gates: limits from the sweep spec.
        candidate_summary: evaluate_artifact summary for the candidate.
        baseline_summary: matching-quant baseline summary, or None when the
            sweep ran without a baseline (baseline-relative gates then report
            "not measured" and pass).
        perf_by_ctx: {str(context): PerfMetrics dump}; defaults to
            ``candidate_summary["perf"]`` when omitted.
        pack: domain pack used to compute weighted scores; when omitted the
            summaries' own "weighted_score" keys (or an unweighted mean) are
            used.

    Returns gate results in a stable order: quality_retention,
    domain_regression, vram, false_refusal, should_refuse,
    tool_call_validity, decode_tps (advisory, never blocking).
    """
    results: list[GateResult] = []
    cand_scores: dict[str, float] = dict(candidate_summary.get("domain_scores") or {})
    base_scores: dict[str, float] = dict((baseline_summary or {}).get("domain_scores") or {})

    # 1. Weighted quality retention >= min (blocker).
    cand_ws = _weighted_of(candidate_summary, pack)
    base_ws = _weighted_of(baseline_summary, pack) if baseline_summary else None
    if cand_ws is None or base_ws is None:
        results.append(
            GateResult(
                name="quality_retention",
                value=None,
                limit=gates.min_quality_retention,
                passed=True,
                note="not measured (no baseline weighted score)",
            )
        )
    else:
        ratio = quality_retention(cand_ws, base_ws)
        results.append(
            GateResult(
                name="quality_retention",
                value=ratio,
                limit=gates.min_quality_retention,
                passed=ratio >= gates.min_quality_retention,
                note=f"weighted {cand_ws:.4f} vs baseline {base_ws:.4f}",
            )
        )

    # 2. Max single-domain regression <= limit (blocker; quality domains only).
    shared_quality = sorted(
        d for d in cand_scores if d in base_scores and d not in SPECIAL_DOMAINS
    )
    if not shared_quality:
        results.append(
            GateResult(
                name="domain_regression",
                value=None,
                limit=gates.max_domain_regression_pts,
                passed=True,
                note="not measured (no shared quality domains with baseline)",
            )
        )
    else:
        regs = domain_regressions(cand_scores, base_scores)
        worst_domain = max(shared_quality, key=lambda d: regs[d])
        worst = regs[worst_domain]
        results.append(
            GateResult(
                name="domain_regression",
                value=worst,
                limit=gates.max_domain_regression_pts,
                passed=worst <= gates.max_domain_regression_pts,
                note=f"worst domain {worst_domain} ({worst:+.1f} pts)",
            )
        )

    # 3. Peak VRAM at gates.min_context <= max_vram_gb (blocker).
    perf = _normalize_perf(perf_by_ctx if perf_by_ctx is not None else candidate_summary.get("perf"))
    ctx_entry = perf.get(str(gates.min_context)) or {}
    results.append(_vram_gate(gates, perf, ctx_entry))

    # 4. False refusal <= absolute limit AND <= baseline rate when known (blocker).
    fr = candidate_summary.get("false_refusal_rate")
    base_fr = (baseline_summary or {}).get("false_refusal_rate")
    if fr is None:
        results.append(
            GateResult(
                name="false_refusal",
                value=None,
                limit=gates.max_false_refusal_rate,
                passed=True,
                note="not measured",
            )
        )
    else:
        fr = float(fr)
        ok_abs = fr <= gates.max_false_refusal_rate
        ok_rel = base_fr is None or fr <= float(base_fr)
        notes: list[str] = []
        if not ok_abs:
            notes.append(f"above absolute limit {gates.max_false_refusal_rate:.1%}")
        if base_fr is not None:
            notes.append(
                f"baseline {float(base_fr):.1%}" + ("" if ok_rel else " exceeded — refusal regression")
            )
        else:
            notes.append("baseline rate unknown; absolute limit only")
        results.append(
            GateResult(
                name="false_refusal",
                value=fr,
                limit=gates.max_false_refusal_rate,
                passed=ok_abs and ok_rel,
                note="; ".join(notes),
            )
        )

    # 5. Should-refuse control set at the required pass rate (hard fail).
    sr = candidate_summary.get("should_refuse_pass_rate")
    if sr is None:
        results.append(
            GateResult(
                name="should_refuse",
                value=None,
                limit=gates.should_refuse_pass_rate,
                passed=True,
                note="not measured",
            )
        )
    else:
        sr = float(sr)
        sr_ok = sr >= gates.should_refuse_pass_rate
        results.append(
            GateResult(
                name="should_refuse",
                value=sr,
                limit=gates.should_refuse_pass_rate,
                passed=sr_ok,
                note=(
                    "control set at required pass rate"
                    if sr_ok
                    else "HARD FAIL: safety behavior regressed below the required refuse rate"
                ),
            )
        )

    # 6. Tool-call schema validity >= min (blocker).
    tv = candidate_summary.get("tool_call_validity")
    if tv is None:
        results.append(
            GateResult(
                name="tool_call_validity",
                value=None,
                limit=gates.min_tool_call_validity,
                passed=True,
                note="not measured",
            )
        )
    else:
        tv = float(tv)
        results.append(
            GateResult(
                name="tool_call_validity",
                value=tv,
                limit=gates.min_tool_call_validity,
                passed=tv >= gates.min_tool_call_validity,
                note="agentic trace schema validity",
            )
        )

    # 7. Decode throughput — advisory, never blocking.
    tps = ctx_entry.get("decode_tps")
    if tps is None:
        for key in sorted(perf, key=_ctx_order, reverse=True):
            if perf[key].get("decode_tps") is not None:
                tps = perf[key]["decode_tps"]
                break
    if gates.min_decode_tps is None:
        results.append(
            GateResult(
                name="decode_tps",
                value=float(tps) if tps is not None else None,
                limit=None,
                passed=True,
                blocking=False,
                note="advisory; no minimum configured",
            )
        )
    elif tps is None:
        results.append(
            GateResult(
                name="decode_tps",
                value=None,
                limit=gates.min_decode_tps,
                passed=True,
                blocking=False,
                note="not measured",
            )
        )
    else:
        tps = float(tps)
        below = tps < gates.min_decode_tps
        results.append(
            GateResult(
                name="decode_tps",
                value=tps,
                limit=gates.min_decode_tps,
                passed=not below,
                blocking=False,
                note=(
                    "below advisory minimum — does not block promotion"
                    if below
                    else "advisory"
                ),
            )
        )

    return results


def pareto_front(points: list[dict[str, Any]]) -> set[str]:
    """Non-dominated artifact_ids over (quality max, vram min, tps max).

    Each point is ``{"artifact_id": str, "quality": float|None,
    "vram": float|None, "tps": float|None}``. None fields are treated as the
    worst possible value for domination purposes, but a lone candidate is
    never excluded. A point is dominated when another point is at least as
    good on all three axes and strictly better on at least one.
    """
    if not points:
        return set()

    def axes(p: Mapping[str, Any]) -> tuple[float, float, float]:
        q = p.get("quality")
        v = p.get("vram")
        t = p.get("tps")
        return (
            float(q) if q is not None else float("-inf"),
            float(v) if v is not None else float("inf"),
            float(t) if t is not None else float("-inf"),
        )

    front: set[str] = set()
    for a in points:
        qa, va, ta = axes(a)
        dominated = False
        for b in points:
            if b is a:
                continue
            qb, vb, tb = axes(b)
            if qb >= qa and vb <= va and tb >= ta and (qb > qa or vb < va or tb > ta):
                dominated = True
                break
        if not dominated:
            front.add(str(a["artifact_id"]))
    if not front:  # defensive: never return an empty front for a non-empty input
        front.add(str(points[0]["artifact_id"]))
    return front


def select_winner(candidates: list[tuple[str, float, list[GateResult]]]) -> str | None:
    """Highest weighted score among candidates passing ALL blocking gates.

    Advisory (non-blocking) gate failures never disqualify a candidate.
    Returns None when no candidate passes — the report then says so and the
    promotion step is skipped.
    """
    passing = [
        (artifact_id, weighted)
        for artifact_id, weighted, gate_results in candidates
        if all(g.passed for g in gate_results if g.blocking)
    ]
    if not passing:
        return None
    return max(passing, key=lambda pair: pair[1])[0]
