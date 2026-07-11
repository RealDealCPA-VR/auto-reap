"""evaluate_artifact: run the full eval suite + perf capture against one artifact.

This is C3's exported entry point (PRD FR-3.1..FR-3.5). The orchestrator calls it
once per artifact; the returned summary dict follows the shared contract:

    { "artifact_id": str, "domain_scores": {domain: float 0..1}, "counts": {domain: int},
      "false_refusal_rate": float|None, "should_refuse_pass_rate": float|None,
      "tool_call_validity": float|None, "perf": {str(context): PerfMetrics.model_dump()},
      "items_scored": int, "responses": {item_id: text} }

"responses" lets the orchestrator feed the BASELINE's answers back in as
baseline_responses when evaluating pruned candidates, which switches open-ended
scoring from the cheap heuristic to the pairwise LLM judge (FR-3.3).

Resume: per-item results are appended to runs/<config_hash>/results.jsonl; items
already scored for this artifact are skipped on re-run and their stored scores
feed the aggregates (PRD FR-4.1). Determinism: every completion runs at
temperature 0 (FR-3.5).
"""

from __future__ import annotations

from typing import Any

from reaplab.core.config import SweepSpec
from reaplab.core.jsonl import append_jsonl, read_jsonl
from reaplab.core.paths import Workspace
from reaplab.core.providers import LLMProvider, get_provider
from reaplab.core.records import ArtifactManifest, EvalRecord, ItemResult, TaskType
from reaplab.core.state import StateDB
from reaplab.evalharness.perf import capture_perf
from reaplab.evalharness.runners import ModelRunner, runner_from_runtime
from reaplab.evalharness.scorers import get_scorer
from reaplab.evalharness.scorers.judge import judge_item

_PERF_SAMPLE_COUNT = 2
_PERF_PROMPT_CHARS = 400


def _score_item(
    spec: SweepSpec,
    manifest: ArtifactManifest,
    record: EvalRecord,
    response_text: str,
    response: Any,
    workspace: Workspace,
    baseline_responses: dict[str, str] | None,
    judge_provider_cache: list[LLMProvider | None],
) -> tuple[float, bool, str, dict[str, Any]]:
    """Route one response to its scorer; open-ended items go to the judge when a
    baseline answer exists and this artifact is not itself the baseline."""
    if (
        record.task_type == TaskType.OPEN_ENDED
        and baseline_responses is not None
        and manifest.kind != "baseline"
        and record.id in baseline_responses
    ):
        if judge_provider_cache[0] is None:
            judge_provider_cache[0] = get_provider(spec.judge.provider)
        score, detail = judge_item(
            record,
            response_text,
            baseline_responses[record.id],
            judge_provider_cache[0],
            votes=spec.judge.votes,
            judge_version=spec.judge.version,
            cache_dir=workspace.judge_cache,
            artifact_hash=manifest.artifact_hash or manifest.artifact_id,
        )
        return score, score >= 0.5, "judge", detail
    if record.task_type == TaskType.OPEN_ENDED and manifest.kind == "baseline":
        # Pairwise judging scores candidates as win-rate vs. the baseline, where
        # parity = 0.5. Anchor the baseline's own open-ended score at 0.5 so the
        # candidate/baseline ratio (quality retention) lives on one scale; without
        # this a candidate tying the baseline everywhere would still read ~67%.
        return 0.5, True, "open_ended_anchor", {"anchor": True}
    scorer = get_scorer(record.task_type)
    score, passed, detail = scorer.score(record, response)
    return score, passed, scorer.name, detail


def evaluate_artifact(
    spec: SweepSpec,
    manifest: ArtifactManifest,
    eval_records: list[EvalRecord],
    workspace: Workspace,
    state: StateDB,
    runner: ModelRunner | None = None,
    baseline_responses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate one artifact end to end; returns the shared summary dict.

    Constraints: eval_records must be non-empty; the runner (picked from
    spec.runtime.kind when not injected) is started at min(spec.runtime.contexts)
    for scoring, then restarted per context for perf capture, and always stopped
    before returning. Raises RunnerError with fix-it guidance when the runtime
    cannot start.
    """
    if not eval_records:
        raise ValueError(
            "evaluate_artifact got an empty eval set. Generate one first "
            "(reap-lab generate) or point spec.eval at an existing eval JSONL."
        )
    config_hash = spec.config_hash()
    workspace.ensure(config_hash)
    results_path = workspace.results_jsonl(config_hash)

    # -- resume: load results already scored for THIS artifact ----------------
    done: dict[str, ItemResult] = {}
    if results_path.exists():
        for r in read_jsonl(results_path, ItemResult):
            if r.artifact_id == manifest.artifact_id:
                done[r.item_id] = r

    contexts = sorted(set(spec.runtime.contexts)) or [4096]
    runner = runner or runner_from_runtime(spec.runtime)
    judge_provider_cache: list[LLMProvider | None] = [None]

    results: list[ItemResult] = []
    responses: dict[str, str] = {}
    perf: dict[str, dict[str, Any]] = {}
    try:
        runner.start(manifest, contexts[0])
        for record in eval_records:
            if record.id in done:
                prior = done[record.id]
                results.append(prior)
                responses[record.id] = prior.response
                continue
            resp = runner.complete(
                record.prompt,
                tools=record.tools if record.task_type == TaskType.TOOL_CALL else None,
                max_tokens=record.max_tokens,
                temperature=0.0,
                record=record,
            )
            score, passed, scorer_name, detail = _score_item(
                spec, manifest, record, resp.text, resp, workspace,
                baseline_responses, judge_provider_cache,
            )
            item = ItemResult(
                item_id=record.id,
                artifact_id=manifest.artifact_id,
                domain=record.domain,
                task_type=record.task_type,
                response=resp.text,
                score=score,
                passed=passed,
                scorer=scorer_name,
                detail=detail,
                latency_ms=resp.latency_ms,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
            )
            append_jsonl(results_path, item)
            results.append(item)
            responses[record.id] = resp.text

        # -- perf capture per context (FR-3.4) --------------------------------
        sample_prompts = [r.prompt[:_PERF_PROMPT_CHARS] for r in eval_records[:_PERF_SAMPLE_COUNT]]
        current_ctx = contexts[0]
        for ctx in contexts:
            if ctx != current_ctx:
                runner.stop()
                runner.start(manifest, ctx)
                current_ctx = ctx
            perf[str(ctx)] = capture_perf(runner, ctx, sample_prompts).model_dump()
    finally:
        runner.stop()

    # -- aggregate -------------------------------------------------------------
    domain_scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    by_domain: dict[str, list[float]] = {}
    for r in results:
        by_domain.setdefault(r.domain, []).append(r.score)
    for domain, scores in sorted(by_domain.items()):
        domain_scores[domain] = round(sum(scores) / len(scores), 4)
        counts[domain] = len(scores)

    benign = [r for r in results if r.task_type == TaskType.REFUSAL_BENIGN]
    refuse = [r for r in results if r.task_type == TaskType.SHOULD_REFUSE]
    tool = [r for r in results if r.task_type == TaskType.TOOL_CALL]
    false_refusal_rate = (
        round(sum(1 for r in benign if not r.passed) / len(benign), 4) if benign else None
    )
    should_refuse_pass_rate = (
        round(sum(1 for r in refuse if r.passed) / len(refuse), 4) if refuse else None
    )
    tool_call_validity = (
        round(sum(1 for r in tool if r.detail.get("schema_valid")) / len(tool), 4) if tool else None
    )

    # -- raw metrics into state (report queries) --------------------------------
    aid = manifest.artifact_id
    for domain, mean in domain_scores.items():
        state.record_metric(aid, f"domain:{domain}", mean)
    if false_refusal_rate is not None:
        state.record_metric(aid, "false_refusal_rate", false_refusal_rate)
    if should_refuse_pass_rate is not None:
        state.record_metric(aid, "should_refuse_pass_rate", should_refuse_pass_rate)
    if tool_call_validity is not None:
        state.record_metric(aid, "tool_call_validity", tool_call_validity)
    for ctx_str, pm in perf.items():
        for field, metric in (
            ("decode_tps", f"decode_tps@{ctx_str}"),
            ("peak_vram_mb", f"peak_vram_mb@{ctx_str}"),
            ("load_time_s", f"load_time_s@{ctx_str}"),
        ):
            if pm.get(field) is not None:
                state.record_metric(aid, metric, pm[field])

    return {
        "artifact_id": aid,
        "domain_scores": domain_scores,
        "counts": counts,
        "false_refusal_rate": false_refusal_rate,
        "should_refuse_pass_rate": should_refuse_pass_rate,
        "tool_call_validity": tool_call_validity,
        "perf": perf,
        "items_scored": len(results),
        "responses": responses,
    }
