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

Scoring context: every item is scored at max(runtime.contexts) — the FR-1.4
long-context items (>=16k tokens) simply do not fit at the smallest configured
context. Perf is then captured from largest to smallest context, so the server
that scored the suite serves the first perf point (no extra restart).

Resume (PRD FR-4.1): per-item results are appended to runs/<config_hash>/results.jsonl.
A stored row is reused ONLY when it provably describes the same measurement:
  - it carries the same artifact_hash as the manifest under test (both non-None) —
    a rebuilt/replaced GGUF re-uses the artifact_id but never the content hash; and
  - for open_ended items, its scorer matches the scoring mode now in effect (judge
    vs. baseline anchor vs. heuristic) — those scales are not interchangeable, and
    mixing them makes quality retention meaningless.
Rows that fail either test are dropped from results.jsonl and re-scored; resume=False
drops every row of THIS artifact (other artifacts share the file) and re-scores all.
Determinism: every completion runs at temperature 0 (FR-3.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reaplab.core.config import SweepSpec
from reaplab.core.jsonl import append_jsonl, read_jsonl, write_jsonl
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

JUDGE_SCORER = "judge"
ANCHOR_SCORER = "open_ended_anchor"
HEURISTIC_SCORER = "open_ended_heuristic"


def open_ended_scorer_name(
    manifest: ArtifactManifest,
    record: EvalRecord,
    baseline_responses: dict[str, str] | None,
) -> str:
    """Which open-ended scoring mode is in effect for this (artifact, item) right now.

    The three modes live on different scales (judge = win-rate vs. baseline, anchor =
    the baseline's own 0.5 reference point, heuristic = 0.75 for any non-refusal), so a
    resumed row scored under a different mode must never be reused.
    """
    if manifest.kind == "baseline":
        return ANCHOR_SCORER
    if baseline_responses is not None and record.id in baseline_responses:
        return JUDGE_SCORER
    return HEURISTIC_SCORER


def _score_item(
    spec: SweepSpec,
    manifest: ArtifactManifest,
    record: EvalRecord,
    response_text: str,
    response: Any,
    judge_cache_dir: Path,
    baseline_responses: dict[str, str] | None,
    judge_provider_cache: list[LLMProvider | None],
) -> tuple[float, bool, str, dict[str, Any]]:
    """Route one response to its scorer; open-ended items go to the judge when a
    baseline answer exists and this artifact is not itself the baseline."""
    if record.task_type == TaskType.OPEN_ENDED:
        mode = open_ended_scorer_name(manifest, record, baseline_responses)
        if mode == JUDGE_SCORER:
            if judge_provider_cache[0] is None:
                judge_provider_cache[0] = get_provider(spec.judge.provider)
            score, detail = judge_item(
                record,
                response_text,
                baseline_responses[record.id],  # type: ignore[index]  # mode implies presence
                judge_provider_cache[0],
                votes=spec.judge.votes,
                judge_version=spec.judge.version,
                cache_dir=judge_cache_dir,
                artifact_hash=manifest.artifact_hash or manifest.artifact_id,
            )
            return score, score >= 0.5, JUDGE_SCORER, detail
        if mode == ANCHOR_SCORER:
            # Pairwise judging scores candidates as win-rate vs. the baseline, where
            # parity = 0.5. Anchor the baseline's own open-ended score at 0.5 so the
            # candidate/baseline ratio (quality retention) lives on one scale; without
            # this a candidate tying the baseline everywhere would still read ~67%.
            return 0.5, True, ANCHOR_SCORER, {"anchor": True}
    scorer = get_scorer(record.task_type)
    score, passed, detail = scorer.score(record, response)
    return score, passed, scorer.name, detail


def _reusable(
    prior: ItemResult,
    manifest: ArtifactManifest,
    record: EvalRecord,
    baseline_responses: dict[str, str] | None,
) -> bool:
    """True when a stored row still describes the measurement we would make now."""
    if manifest.artifact_hash is None or prior.artifact_hash is None:
        return False  # unverifiable identity: the file may have been rebuilt
    if prior.artifact_hash != manifest.artifact_hash:
        return False  # same id, different bytes
    if record.task_type == TaskType.OPEN_ENDED:
        return prior.scorer == open_ended_scorer_name(manifest, record, baseline_responses)
    return True


def _resume_rows(
    results_path: Path,
    manifest: ArtifactManifest,
    eval_records: list[EvalRecord],
    baseline_responses: dict[str, str] | None,
    resume: bool,
) -> dict[str, ItemResult]:
    """Prune results.jsonl to the rows we can still trust and return them by item_id.

    results.jsonl is SHARED by every artifact of the sweep, so only this artifact's
    rows are ever touched: unusable ones are rewritten away (rather than left to
    duplicate on append), everyone else's rows survive byte-for-byte.
    """
    if not results_path.exists():
        return {}
    rows = read_jsonl(results_path, ItemResult)
    by_id = {r.id: r for r in eval_records}

    keep: list[ItemResult] = []
    reusable: dict[str, ItemResult] = {}
    dropped = 0
    for row in rows:
        if row.artifact_id != manifest.artifact_id:
            keep.append(row)  # another artifact's row: never our business
            continue
        record = by_id.get(row.item_id)
        if resume and record is not None and _reusable(row, manifest, record, baseline_responses):
            keep.append(row)
            reusable[row.item_id] = row
            continue
        if not resume or record is not None:
            dropped += 1  # stale/forced: re-scored below, so drop it here
            continue
        keep.append(row)  # item is not in this eval set; leave it alone

    if dropped:
        write_jsonl(results_path, keep)
    return reusable


def evaluate_artifact(
    spec: SweepSpec,
    manifest: ArtifactManifest,
    eval_records: list[EvalRecord],
    workspace: Workspace,
    state: StateDB,
    runner: ModelRunner | None = None,
    baseline_responses: dict[str, str] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate one artifact end to end; returns the shared summary dict.

    Constraints: eval_records must be non-empty; the runner (picked from
    spec.runtime.kind when not injected) is started at max(spec.runtime.contexts)
    for scoring, then restarted per context for perf capture (largest first), and
    always stopped before returning. resume=False re-scores every item of this
    artifact, leaving other artifacts' rows in results.jsonl untouched. Raises
    RunnerError with fix-it guidance when the runtime cannot start.
    """
    if not eval_records:
        raise ValueError(
            "evaluate_artifact got an empty eval set. Generate one first "
            "(reap-lab generate) or point spec.eval at an existing eval JSONL."
        )
    config_hash = spec.config_hash()
    workspace.ensure(config_hash)
    results_path = workspace.results_jsonl(config_hash)
    judge_cache_dir = workspace.judge_cache / config_hash  # judgments are per-sweep

    done = _resume_rows(results_path, manifest, eval_records, baseline_responses, resume)

    contexts = sorted(set(spec.runtime.contexts)) or [4096]
    scoring_context = contexts[-1]  # long-context items (FR-1.4) need the biggest one
    runner = runner or runner_from_runtime(spec.runtime)
    judge_provider_cache: list[LLMProvider | None] = [None]

    results: list[ItemResult] = []
    responses: dict[str, str] = {}
    perf: dict[str, dict[str, Any]] = {}
    try:
        runner.start(manifest, scoring_context)
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
                spec, manifest, record, resp.text, resp, judge_cache_dir,
                baseline_responses, judge_provider_cache,
            )
            item = ItemResult(
                item_id=record.id,
                artifact_id=manifest.artifact_id,
                artifact_hash=manifest.artifact_hash,
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

        # -- perf capture per context (FR-3.4), largest first ------------------
        # The scoring server is already up at max(contexts), so descending order
        # captures that context without a restart: same restart count as before,
        # and every PerfMetrics is snapshotted while its server still runs (the
        # runner's peak_vram_mb is live until stop()).
        sample_prompts = [r.prompt[:_PERF_PROMPT_CHARS] for r in eval_records[:_PERF_SAMPLE_COUNT]]
        current_ctx = scoring_context
        for ctx in reversed(contexts):
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
