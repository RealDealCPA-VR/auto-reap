"""End-to-end pipeline in procedural (mock) mode: files, counts, determinism,
ids, long-context, refusal suites, state/resume, failure marking, pre-existing sets."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from reaplab.core.config import DataCfg, DomainPack, ProviderCfg
from reaplab.core.jsonl import read_jsonl, write_jsonl
from reaplab.core.paths import Workspace
from reaplab.core.records import CalibrationRecord, EvalRecord, TaskType
from reaplab.core.state import StateDB
from reaplab.datagen import (
    AUDIT_SAMPLE_FILENAME,
    DEDUP_REPORT_FILENAME,
    estimate_tokens,
    generate_datasets,
    plan_counts,
)

OUTPUTS = ["calibration_v1.jsonl", "eval_v1.jsonl", DEDUP_REPORT_FILENAME, AUDIT_SAMPLE_FILENAME]


def _run(spec, tmp_root: Path, name: str = "ws") -> tuple[Path, Path, Workspace]:
    ws = Workspace(tmp_root / name).ensure()
    cal, ev = generate_datasets(spec, ws)
    return cal, ev, ws


def test_writes_expected_files_and_counts(make_spec, tmp_path):
    spec = make_spec()
    cal_path, eval_path, ws = _run(spec, tmp_path)
    assert cal_path == ws.data / "calibration_v1.jsonl"
    assert eval_path == ws.data / "eval_v1.jsonl"
    for f in OUTPUTS:
        assert (ws.data / f).exists(), f"missing output {f}"

    cal = read_jsonl(cal_path, CalibrationRecord)
    ev = read_jsonl(eval_path, EvalRecord)
    pack = DomainPack.from_yaml(spec.domain_pack)
    plan = plan_counts(pack, spec.data, seed=42)

    # nothing may be silently lost: procedural variety must survive the filter
    assert len(cal) == plan.calibration_total == 20
    assert len(ev) == plan.eval_total == 10 + 10 + 15

    ev_by_domain = Counter(r.domain for r in ev)
    for alloc in plan.allocations:
        assert ev_by_domain[alloc.spec.name] == alloc.eval_count, alloc.spec.name


def test_ids_unique_stable_and_well_formed(make_spec, tmp_path):
    cal_path, eval_path, _ = _run(make_spec(), tmp_path)
    cal = read_jsonl(cal_path, CalibrationRecord)
    ev = read_jsonl(eval_path, EvalRecord)
    assert all(re.fullmatch(r"cal-\d{6}", r.id) for r in cal)
    assert all(re.fullmatch(r"ev-\d{6}", r.id) for r in ev)
    assert len({r.id for r in cal}) == len(cal)
    assert len({r.id for r in ev}) == len(ev)
    assert [r.id for r in ev] == sorted(r.id for r in ev), "stable ascending order"


def test_source_field_names_the_provider(make_spec, tmp_path):
    cal_path, eval_path, _ = _run(make_spec(), tmp_path)
    for r in read_jsonl(cal_path, CalibrationRecord) + read_jsonl(eval_path, EvalRecord):
        assert r.source == "synthetic-mock"


def test_task_types_match_domains(make_spec, tmp_path):
    _, eval_path, _ = _run(make_spec(), tmp_path)
    ev = read_jsonl(eval_path, EvalRecord)
    by_domain = {r.domain: r.task_type for r in ev}
    assert by_domain["txn_classify"] == TaskType.EXACT
    assert by_domain["report_extract"] == TaskType.JSON_SCHEMA
    assert by_domain["ops_tools"] == TaskType.TOOL_CALL
    assert by_domain["benign_sensitive"] == TaskType.REFUSAL_BENIGN
    assert by_domain["should_refuse"] == TaskType.SHOULD_REFUSE
    # scoreability fields present
    for r in ev:
        if r.task_type == TaskType.EXACT:
            assert r.gold
        elif r.task_type == TaskType.JSON_SCHEMA:
            assert r.gold and r.json_schema
        elif r.task_type == TaskType.TOOL_CALL:
            assert r.tools and r.expected_tool
        elif r.task_type == TaskType.OPEN_ENDED:
            assert r.rubric


def test_refusal_suite_counts(make_spec, tmp_path):
    _, eval_path, _ = _run(make_spec(), tmp_path)
    ev = read_jsonl(eval_path, EvalRecord)
    counts = Counter(r.domain for r in ev)
    assert counts["benign_sensitive"] == 10
    assert counts["should_refuse"] == 15


def test_long_context_items_exist_and_are_long(make_spec, tmp_path):
    spec = make_spec(data=DataCfg(calibration_size=20, eval_size=10, long_context_share=0.5))
    cal_path, eval_path, _ = _run(spec, tmp_path)
    ev_long = [r for r in read_jsonl(eval_path, EvalRecord) if "long_context" in r.tags]
    cal_long = [r for r in read_jsonl(cal_path, CalibrationRecord) if "long_context" in r.tags]
    assert len(ev_long) == 1  # ceil(0.5 * 1) for long_review's single eval item
    assert len(cal_long) == 1  # ceil(0.5 * 2)
    for r in ev_long + cal_long:
        assert r.domain == "long_review"
        assert estimate_tokens(r.prompt) >= 16_000


def test_same_seed_produces_identical_files(make_spec, tmp_path):
    spec = make_spec()
    _, _, ws_a = _run(spec, tmp_path, "a")
    _, _, ws_b = _run(spec, tmp_path, "b")
    for f in OUTPUTS:
        assert (ws_a.data / f).read_bytes() == (ws_b.data / f).read_bytes(), f


def test_different_seed_produces_different_data(make_spec, tmp_path):
    _, _, ws_a = _run(make_spec(), tmp_path, "a")
    _, _, ws_b = _run(make_spec(seeds=[7]), tmp_path, "b")
    assert (ws_a.data / "eval_v1.jsonl").read_bytes() != (ws_b.data / "eval_v1.jsonl").read_bytes()


def test_dedup_report_written_and_consistent(make_spec, tmp_path):
    _, eval_path, ws = _run(make_spec(), tmp_path)
    report = json.loads((ws.data / DEDUP_REPORT_FILENAME).read_text(encoding="utf-8"))
    ev = read_jsonl(eval_path, EvalRecord)
    assert report["backend"] == "fuzzy"
    assert report["eval_kept"] == len(ev)
    assert report["eval_in"] == report["eval_kept"] + len(report["dropped"])


def test_embedding_backend_works_offline_via_mock_embeddings(make_spec, tmp_path):
    spec = make_spec(
        data=DataCfg(calibration_size=20, eval_size=10, dedup_backend="embedding")
    )
    _, eval_path, ws = _run(spec, tmp_path)
    report = json.loads((ws.data / DEDUP_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["backend"] == "embedding"
    assert len(read_jsonl(eval_path, EvalRecord)) == report["eval_kept"]


def test_audit_sample_written_with_min_ten_items(make_spec, tmp_path):
    _, eval_path, ws = _run(make_spec(), tmp_path)
    text = (ws.data / AUDIT_SAMPLE_FILENAME).read_text(encoding="utf-8")
    ev = read_jsonl(eval_path, EvalRecord)
    sampled = text.count("### ev-")
    assert sampled == max(10, round(0.05 * len(ev)))
    assert "audit" in text.lower()


def test_state_marked_done_and_resume_short_circuits(make_spec, tmp_path, monkeypatch):
    spec = make_spec()
    ws = Workspace(tmp_path / "ws").ensure()
    with StateDB(tmp_path / "state.db") as db:
        cal1, ev1 = generate_datasets(spec, ws, state=db)
        jobs = db.jobs("datagen")
        assert len(jobs) == 1 and jobs[0]["status"] == "done"
        assert jobs[0]["meta"]["eval_count"] == 35

        # if resume fails, this raises: planning must never run again
        def _boom(*a, **k):  # pragma: no cover - only on regression
            raise AssertionError("plan_counts called despite completed datagen stage")

        monkeypatch.setattr("reaplab.datagen.pipeline.plan_counts", _boom)
        cal2, ev2 = generate_datasets(spec, ws, state=db)
        assert (cal2, ev2) == (cal1, ev1)


def test_missing_files_force_regeneration_even_when_state_done(make_spec, tmp_path):
    spec = make_spec()
    ws = Workspace(tmp_path / "ws").ensure()
    with StateDB(tmp_path / "state.db") as db:
        cal, ev = generate_datasets(spec, ws, state=db)
        ev.unlink()  # simulate a deleted artifact
        cal2, ev2 = generate_datasets(spec, ws, state=db)
        assert ev2.exists() and cal2.exists()


def test_failure_marks_state_failed(make_spec, tmp_path):
    # embedding backend + a provider that cannot embed -> instructive ValueError
    spec = make_spec(
        data=DataCfg(
            calibration_size=20,
            eval_size=10,
            dedup_backend="embedding",
            embedding_provider=ProviderCfg(kind="claude-cli"),
        )
    )
    ws = Workspace(tmp_path / "ws").ensure()
    with StateDB(tmp_path / "state.db") as db:
        with pytest.raises(ValueError, match="cannot embed"):
            generate_datasets(spec, ws, state=db)
        jobs = db.jobs("datagen")
        assert jobs[0]["status"] == "failed"
        assert "cannot embed" in jobs[0]["error"]


def test_missing_domain_pack_is_instructive(make_spec, tmp_path):
    spec = make_spec()
    spec.domain_pack = str(tmp_path / "nope.yaml")
    with pytest.raises(FileNotFoundError, match="domain-packs"):
        generate_datasets(spec, Workspace(tmp_path / "ws").ensure())


def test_pre_existing_datasets_are_reused_not_regenerated(make_spec, tmp_path):
    pre_cal = [
        CalibrationRecord(id="cal-900001", domain="txn_classify", prompt="Existing calibration prompt one."),
        CalibrationRecord(id="cal-900002", domain="txn_classify", prompt="A different existing calibration prompt."),
    ]
    pre_ev = [
        EvalRecord(
            id="ev-900001", domain="txn_classify", task_type=TaskType.EXACT,
            prompt="Existing eval prompt about categorizing a hardware invoice.", gold="Equipment",
        )
    ]
    cal_src = tmp_path / "pre_cal.jsonl"
    ev_src = tmp_path / "pre_ev.jsonl"
    write_jsonl(cal_src, pre_cal)
    write_jsonl(ev_src, pre_ev)

    spec = make_spec(calibration=str(cal_src), eval=str(ev_src))
    cal_path, eval_path, ws = _run(spec, tmp_path)
    cal = read_jsonl(cal_path, CalibrationRecord)
    ev = read_jsonl(eval_path, EvalRecord)
    assert [r.id for r in cal] == ["cal-900001", "cal-900002"]
    assert [r.id for r in ev] == ["ev-900001"]
    assert cal_path.parent == ws.data  # re-emitted into the workspace


def test_pre_existing_path_missing_is_instructive(make_spec, tmp_path):
    spec = make_spec(calibration=str(tmp_path / "ghost.jsonl"))
    with pytest.raises(FileNotFoundError, match="calibration"):
        generate_datasets(spec, Workspace(tmp_path / "ws").ensure())
