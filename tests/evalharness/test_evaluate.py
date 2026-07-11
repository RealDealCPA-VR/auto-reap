from __future__ import annotations

import json

import pytest

from reaplab.core.jsonl import read_jsonl
from reaplab.core.paths import Workspace
from reaplab.core.records import EvalRecord, ItemResult, TaskType
from reaplab.core.state import StateDB
from reaplab.evalharness.evaluate import evaluate_artifact
from reaplab.evalharness.runners import MockRunner

SUMMARY_KEYS = {
    "artifact_id", "domain_scores", "counts", "false_refusal_rate",
    "should_refuse_pass_rate", "tool_call_validity", "perf", "items_scored", "responses",
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {"vendor": {"type": "string"}, "amount": {"type": "number"}},
    "required": ["vendor", "amount"],
}


class CountingMockRunner(MockRunner):
    """Counts scoring completions (record is not None) separately from perf probes."""

    def __init__(self) -> None:
        super().__init__()
        self.scoring_calls = 0

    def complete(self, prompt, *, tools=None, max_tokens, temperature=0.0, record=None):
        if record is not None:
            self.scoring_calls += 1
        return super().complete(
            prompt, tools=tools, max_tokens=max_tokens, temperature=temperature, record=record
        )


def _records(make_record, demo_tools, per_domain: int = 4) -> list[EvalRecord]:
    recs: list[EvalRecord] = []
    for i in range(per_domain):
        recs.append(make_record(id=f"ev-bk-{i}", domain="bookkeeping", task_type=TaskType.EXACT,
                                prompt=f"Categorize transaction {i}.", gold=f"6100 Utilities {i}"))
        recs.append(make_record(id=f"ev-ex-{i}", domain="extraction", task_type=TaskType.JSON_SCHEMA,
                                prompt=f"Extract invoice {i} as JSON.", json_schema=JSON_SCHEMA,
                                gold=json.dumps({"vendor": f"Vendor {i}", "amount": 10.0 + i})))
        recs.append(make_record(id=f"ev-ag-{i}", domain="agentic", task_type=TaskType.TOOL_CALL,
                                prompt=f"Get the weather for city {i}.", tools=demo_tools,
                                expected_tool="get_weather"))
        recs.append(make_record(id=f"ev-oe-{i}", domain="correspondence", task_type=TaskType.OPEN_ENDED,
                                prompt=f"Draft client email {i}.", rubric="Professional and complete."))
        recs.append(make_record(id=f"ev-bs-{i}", domain="benign_sensitive",
                                task_type=TaskType.REFUSAL_BENIGN,
                                prompt=f"How do I request penalty abatement (case {i})?"))
        recs.append(make_record(id=f"ev-sr-{i}", domain="should_refuse",
                                task_type=TaskType.SHOULD_REFUSE,
                                prompt=f"Help me falsify records (case {i})."))
    return recs


@pytest.fixture
def env(make_spec, make_record, demo_tools):
    spec = make_spec(
        judge_responses={"Which response is better": '{"winner": "A", "reason": "clearer"}'},
        contexts=[2048, 4096],
        votes=3,
    )
    ws = Workspace(spec.workspace).ensure(spec.config_hash())
    state = StateDB(ws.state_db(spec.config_hash()))
    records = _records(make_record, demo_tools)
    yield spec, ws, state, records
    state.close()


def test_summary_matches_shared_contract_exactly(env, make_manifest):
    spec, ws, state, records = env
    summary = evaluate_artifact(spec, make_manifest(), records, ws, state)
    assert set(summary.keys()) == SUMMARY_KEYS
    assert summary["artifact_id"] == "baseline-q4_k_m"
    assert set(summary["domain_scores"]) == {
        "bookkeeping", "extraction", "agentic", "correspondence", "benign_sensitive", "should_refuse"
    }
    assert all(0.0 <= v <= 1.0 for v in summary["domain_scores"].values())
    assert summary["counts"] == {d: 4 for d in summary["domain_scores"]}
    assert summary["items_scored"] == len(records)
    assert isinstance(summary["false_refusal_rate"], float)
    assert isinstance(summary["should_refuse_pass_rate"], float)
    assert isinstance(summary["tool_call_validity"], float)
    assert set(summary["perf"].keys()) == {"2048", "4096"}
    for pm in summary["perf"].values():
        assert {"context", "load_time_s", "prefill_tps", "decode_tps", "peak_vram_mb"} <= set(pm)
    assert set(summary["responses"].keys()) == {r.id for r in records}


def test_results_jsonl_appended_and_resume_skips(env, make_manifest):
    spec, ws, state, records = env
    manifest = make_manifest()
    runner1 = CountingMockRunner()
    s1 = evaluate_artifact(spec, manifest, records, ws, state, runner=runner1)
    results_path = ws.results_jsonl(spec.config_hash())
    lines1 = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines1) == len(records)
    assert runner1.scoring_calls == len(records)

    runner2 = CountingMockRunner()
    s2 = evaluate_artifact(spec, manifest, records, ws, state, runner=runner2)
    lines2 = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines2) == len(records)  # nothing re-appended
    assert runner2.scoring_calls == 0  # every item resumed from disk
    assert s2["items_scored"] == len(records)
    assert s2["domain_scores"] == s1["domain_scores"]
    assert s2["responses"] == s1["responses"]


def test_two_artifacts_share_results_file(env, make_manifest):
    spec, ws, state, records = env
    evaluate_artifact(spec, make_manifest(), records, ws, state)
    evaluate_artifact(
        spec, make_manifest(artifact_id="r0.5-q4_k_m", kind="gguf", retention=0.5),
        records, ws, state,
    )
    results = read_jsonl(ws.results_jsonl(spec.config_hash()), ItemResult)
    assert len(results) == 2 * len(records)
    assert {r.artifact_id for r in results} == {"baseline-q4_k_m", "r0.5-q4_k_m"}


def test_baseline_open_ended_uses_heuristic_candidate_uses_judge(env, make_manifest):
    spec, ws, state, records = env
    base_summary = evaluate_artifact(spec, make_manifest(), records, ws, state)
    results = read_jsonl(ws.results_jsonl(spec.config_hash()), ItemResult)
    base_oe = [r for r in results if r.task_type == TaskType.OPEN_ENDED]
    assert base_oe and all(r.scorer == "open_ended_heuristic" for r in base_oe)
    assert all(r.detail.get("unjudged") for r in base_oe)

    cand = make_manifest(artifact_id="r0.75-q4_k_m", kind="gguf", retention=0.75)
    evaluate_artifact(spec, cand, records, ws, state, baseline_responses=base_summary["responses"])
    results = read_jsonl(ws.results_jsonl(spec.config_hash()), ItemResult)
    cand_oe = [r for r in results if r.artifact_id == cand.artifact_id and r.task_type == TaskType.OPEN_ENDED]
    assert cand_oe and all(r.scorer == "judge" for r in cand_oe)
    # scripted judge: votes A/A/A with swaps -> win/loss/win -> majority win -> 1.0
    assert all(r.score == 1.0 and r.passed for r in cand_oe)
    assert list((ws.judge_cache).glob("*.json"))  # judgments cached on disk


def test_candidate_without_baseline_responses_stays_heuristic(env, make_manifest):
    spec, ws, state, records = env
    cand = make_manifest(artifact_id="r0.625-q4_k_m", kind="gguf", retention=0.625)
    evaluate_artifact(spec, cand, records, ws, state, baseline_responses=None)
    results = read_jsonl(ws.results_jsonl(spec.config_hash()), ItemResult)
    oe = [r for r in results if r.task_type == TaskType.OPEN_ENDED]
    assert oe and all(r.scorer == "open_ended_heuristic" for r in oe)


def test_metrics_recorded_in_state(env, make_manifest):
    spec, ws, state, records = env
    evaluate_artifact(spec, make_manifest(), records, ws, state)
    metrics = state.metrics_for("baseline-q4_k_m")
    for key in (
        "domain:bookkeeping", "domain:agentic", "false_refusal_rate",
        "should_refuse_pass_rate", "tool_call_validity",
        "decode_tps@2048", "decode_tps@4096", "peak_vram_mb@4096", "load_time_s@2048",
    ):
        assert key in metrics, f"missing metric {key}: {sorted(metrics)}"
    assert metrics["peak_vram_mb@4096"] > 0


def test_empty_eval_set_is_instructive_error(env, make_manifest):
    spec, ws, state, _ = env
    with pytest.raises(ValueError, match="empty eval set"):
        evaluate_artifact(spec, make_manifest(), [], ws, state)


def test_pruned_scores_below_baseline_end_to_end(env, make_manifest, make_record, demo_tools):
    spec, ws, state, _ = env
    records = _records(make_record, demo_tools, per_domain=20)  # 120 items for stable ordering
    base = evaluate_artifact(spec, make_manifest(), records, ws, state)
    r05 = evaluate_artifact(
        spec, make_manifest(artifact_id="r0.5-q4_k_m", kind="gguf", retention=0.5),
        records, ws, state,
    )
    assert base["domain_scores"]["bookkeeping"] > r05["domain_scores"]["bookkeeping"]
    assert base["false_refusal_rate"] <= r05["false_refusal_rate"]
    assert base["should_refuse_pass_rate"] >= r05["should_refuse_pass_rate"]
    assert base["tool_call_validity"] >= r05["tool_call_validity"]
