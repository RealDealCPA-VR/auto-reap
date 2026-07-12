"""End-to-end sweep engine tests with injected fake stages: happy path,
failure isolation, resume, disk guards, manual steps, and the promote hand-off."""

from __future__ import annotations

from pathlib import Path

import pytest

from reaplab.core.config import PromoteCfg
from reaplab.core.state import StateDB
from reaplab.prune import NeedsManualStep


def _state(harness) -> StateDB:
    from reaplab.core.paths import Workspace

    workspace = Workspace(harness.spec.workspace)
    return StateDB(workspace.state_db(harness.spec.config_hash()))


def test_happy_path_report_lists_all_grid_artifacts(harness):
    report_path = harness.run()
    assert report_path.exists()
    md = report_path.read_text(encoding="utf-8")
    for artifact_id in ("baseline-q4_k_m", "r0.75-q4_k_m", "r0.5-q4_k_m"):
        assert artifact_id in md
    winner_line = next(line for line in md.splitlines() if "**Winner:**" in line)
    assert "r0.75-q4_k_m" in winner_line
    # stage call accounting: 1 datagen, 1 baseline build, 2 retention builds,
    # 3 evals (baseline + 2 candidates)
    assert harness.counts["datagen"] == 1
    assert harness.counts["build_baseline"] == 1
    assert harness.counts["build_artifacts"] == 2
    assert harness.counts["evaluate"] == 3


def test_candidates_receive_matching_baseline_responses(harness):
    harness.run()
    assert harness.baseline_responses_seen == [
        {"ev-1": "baseline response"},
        {"ev-1": "baseline response"},
    ]


def test_datasets_land_under_the_per_sweep_run_dir(harness):
    harness.run()
    workspace = Path(harness.spec.workspace)
    data_dir = workspace / "runs" / harness.spec.config_hash() / "data"
    assert (data_dir / "eval_v1.jsonl").exists()
    assert (data_dir / "calibration_v1.jsonl").exists()


def test_failure_isolation_one_bad_retention_does_not_kill_the_sweep(harness):
    harness.fail_retentions = {0.5}
    report_path = harness.run()
    md = report_path.read_text(encoding="utf-8")
    # the failed retention's artifacts never appear as candidates
    assert "r0.5-q4_k_m" not in md
    # ...the component's own failure row is reported verbatim
    assert "prune:r0.5" in md
    assert "synthetic prune failure" in md
    # ...and run_sweep records its coarse failure in its OWN namespace
    assert "sweep:r0.5" in md
    # the healthy retention is still ranked and wins
    winner_line = next(line for line in md.splitlines() if "**Winner:**" in line)
    assert "r0.75-q4_k_m" in winner_line


def test_run_sweep_never_writes_component_owned_rows(harness):
    """[6]/[m4]: prune/convert rows belong to the prune component. run_sweep must not
    pre-emptively mark them running (phantom 'running' rows that never resolve) nor
    overwrite a component's fine-grained failure with a coarse one."""
    harness.fail_retentions = {0.5}
    harness.run()
    with _state(harness) as state:
        jobs = {(j["stage"], j["key"]): j for j in state.jobs()}
        # the component's error text survived — run_sweep did not clobber the row
        assert jobs[("prune", "r0.5")]["status"] == "failed"
        assert "synthetic prune failure" in jobs[("prune", "r0.5")]["error"]
        # run_sweep's own coarse failure lives in the ("sweep", key) namespace
        assert jobs[("sweep", "r0.5")]["status"] == "failed"
        # no convert row is ever left 'running' by run_sweep
        assert not [
            key for key, job in jobs.items()
            if key[0] == "convert" and job["status"] == "running"
        ]
        # the successful retention's rows are the component's done rows
        assert jobs[("prune", "r0.75")]["status"] == "done"
        assert jobs[("convert", "r0.75-q4_k_m")]["status"] == "done"


def test_resume_skips_completed_stages(harness):
    harness.run()
    first_counts = dict(harness.counts)
    report_path = harness.run()  # same spec, same workspace, same state db
    assert report_path.exists()
    assert dict(harness.counts) == first_counts, "resume must not re-run finished stages"


def test_resume_false_forces_full_rerun_and_is_threaded_into_eval(harness):
    harness.run()
    harness.run(resume=False)
    assert harness.counts["datagen"] == 2
    assert harness.counts["build_baseline"] == 2
    assert harness.counts["build_artifacts"] == 4
    assert harness.counts["evaluate"] == 6
    # C5: the eval harness must be told NOT to reuse stored per-item rows
    assert harness.resume_seen[:3] == [True, True, True]
    assert harness.resume_seen[3:] == [False, False, False]


def test_initial_disk_guard_aborts_instructively(harness):
    harness.spec.min_free_disk_gb = 10**9  # absurd requirement no machine meets
    with pytest.raises(RuntimeError) as excinfo:
        harness.run()
    message = str(excinfo.value)
    assert "min_free_disk_gb" in message
    assert "GB" in message
    assert harness.counts["datagen"] == 0  # aborted before any work


def test_disk_guard_runs_before_each_build_not_before_the_evals(harness, monkeypatch):
    """[9]/[26]: the BUILDS consume 15-35 GB; evals consume ~nothing. The guard must
    stop the builds (and isolate the retention) rather than skip cheap evals."""
    harness.spec.include_baseline = False
    harness.spec.min_free_disk_gb = 10.0
    calls = {"n": 0}

    def fake_free(path):
        calls["n"] += 1
        return 1000.0 if calls["n"] == 1 else 0.0  # initial guard passes, re-checks fail

    monkeypatch.setattr("reaplab.orchestrate.sweep.free_disk_gb", fake_free)
    report_path = harness.run()
    md = report_path.read_text(encoding="utf-8")

    assert harness.counts["build_artifacts"] == 0, "the disk-hungry build must be skipped"
    assert harness.counts["evaluate"] == 0
    assert "sweep:r0.75" in md and "sweep:r0.5" in md
    assert "insufficient disk" in md
    assert "no candidate passed all blocking gates" in md


def test_disk_guard_before_baseline_is_fatal(harness, monkeypatch):
    harness.spec.min_free_disk_gb = 10.0
    calls = {"n": 0}

    def fake_free(path):
        calls["n"] += 1
        return 1000.0 if calls["n"] == 1 else 0.0

    monkeypatch.setattr("reaplab.orchestrate.sweep.free_disk_gb", fake_free)
    with pytest.raises(RuntimeError) as excinfo:
        harness.run()
    assert "insufficient disk" in str(excinfo.value)
    assert harness.counts["build_baseline"] == 0


def test_datagen_failure_is_fatal_and_instructive(harness):
    def broken_datagen(spec, workspace):
        raise ValueError("provider exploded")

    from reaplab.orchestrate import run_sweep

    with pytest.raises(RuntimeError) as excinfo:
        run_sweep(
            harness.spec,
            datagen_fn=broken_datagen,
            build_baseline_fn=harness.build_baseline,
            build_artifacts_fn=harness.build_artifacts,
            evaluate_fn=harness.evaluate,
        )
    assert "Dataset generation failed" in str(excinfo.value)


def test_quant_typo_fails_before_any_work(harness):
    harness.spec.quants = ["Q4_KM"]  # typo: no such llama.cpp quant
    with pytest.raises(RuntimeError) as excinfo:
        harness.run()
    message = str(excinfo.value)
    assert "Q4_K_M" in message  # the validator suggests the closest valid name
    assert harness.counts["datagen"] == 0, "a typo must not cost a $75 remote prune"


def test_manual_step_continues_the_grid_and_reports_instructions(harness):
    """[13]/[29]/C6: a retention needing a manual (remote) step is NOT a failure. The
    grid continues, the report carries the instructions, and the sweep returns."""
    harness.manual_retentions = {0.5}
    report_path = harness.run()
    md = report_path.read_text(encoding="utf-8")

    assert "## Manual steps pending" in md
    assert "prune_remote_r0.5.sh" in md
    # not a failure: the failed-configs section stays clean
    failed_part = md.split("## Failed configs")[1].split("##")[0]
    assert "None." in failed_part
    # the healthy retention still won
    winner_line = next(line for line in md.splitlines() if "**Winner:**" in line)
    assert "r0.75-q4_k_m" in winner_line
    with _state(harness) as state:
        assert state.status("prune", "r0.5") == "manual"


def test_manual_step_with_nothing_evaluated_reraises_after_writing_the_report(harness):
    harness.manual_retentions = {0.75, 0.5}
    harness.spec.include_baseline = False
    with pytest.raises(NeedsManualStep) as excinfo:
        harness.run()
    assert "prune_remote_r0.75.sh" in str(excinfo.value)
    assert "prune_remote_r0.5.sh" in str(excinfo.value)
    # the report was written BEFORE the re-raise, with both steps in it
    report = (
        Path(harness.spec.workspace) / "reports" / f"sweep-{harness.spec.config_hash()}.md"
    )
    assert report.exists()
    md = report.read_text(encoding="utf-8")
    assert "## Manual steps pending" in md
    assert "prune:r0.75" in md and "prune:r0.5" in md


def test_promote_places_winner_in_lmstudio_layout(harness, tmp_path):
    lms_dir = tmp_path / "lms-models"
    harness.spec.promote = PromoteCfg(lmstudio_dir=str(lms_dir), publisher="reap-lab")
    report_path = harness.run(promote=True)
    assert report_path.exists()
    dest = (
        lms_dir
        / "reap-lab"
        / "Qwen3-30B-A3B-r0.75-q4_k_m"
        / "Qwen3-30B-A3B-r0.75-q4_k_m.gguf"
    )
    assert dest.exists(), "winner must land in the two-level publisher/model layout"
    # losers were archived (archive_losers defaults to True), winner source kept
    workspace = Path(harness.spec.workspace)
    artifacts = workspace / "artifacts" / harness.spec.config_hash()
    assert (artifacts / "r0.75-q4_k_m.gguf").exists()
    assert not (artifacts / "r0.5-q4_k_m.gguf").exists()
    assert (workspace / "archive" / "r0.5-q4_k_m.gguf").exists()
    # a decision page was written alongside the report
    decisions = list((workspace / "reports").glob("decision-*.md"))
    assert decisions, "promotion must write a decision page"


def test_promotion_archives_only_evaluated_losers(harness, tmp_path):
    """[7]: never-evaluated builds, bf16 intermediates and baselines must survive —
    archiving them strands a partially-evaluated grid forever."""
    lms_dir = tmp_path / "lms-models"
    harness.spec.promote = PromoteCfg(lmstudio_dir=str(lms_dir), publisher="reap-lab")
    harness.fail_retentions = {0.5}  # r0.5 is never built, never evaluated

    workspace = Path(harness.spec.workspace)
    artifacts = workspace / "artifacts" / harness.spec.config_hash()
    artifacts.mkdir(parents=True, exist_ok=True)
    stranded = artifacts / "r0.5-q4_k_m.gguf"  # a build from an earlier partial run
    stranded.write_bytes(b"GGUF-not-evaluated")
    bf16 = artifacts / "Qwen3-30B-A3B-bf16.gguf"
    bf16.write_bytes(b"GGUF-bf16-intermediate")

    harness.run(promote=True)

    assert stranded.exists(), "an unevaluated candidate must not be archived"
    assert bf16.exists(), "the bf16 intermediate must not be archived"
    assert (artifacts / "baseline-q4_k_m.gguf").exists(), "baselines are not losers"
    assert not list((workspace / "archive").glob("*.gguf")), "nothing to archive here"


def test_promotion_failure_fails_the_sweep(harness, tmp_path):
    """[m21]: a failed smoke test must not exit 0."""
    harness.spec.promote = PromoteCfg(
        lmstudio_dir=str(tmp_path / "lms-models"),
        publisher="reap-lab",
        smoke_command='python -c "import sys; sys.exit(3)"',
    )
    with pytest.raises(RuntimeError) as excinfo:
        harness.run(promote=True)
    assert "Promotion failed" in str(excinfo.value)
    with _state(harness) as state:
        assert state.status("sweep", "promote") == "failed"


def test_user_baseline_gguf_expects_exactly_one_convert_row(harness, tmp_path):
    """[m4]/C4: a user-supplied baseline_gguf is ONE artifact, not one per quant. Using
    a naive baseline-<q>-per-quant expectation left phantom rows that never completed
    and defeated the baseline resume."""
    baseline = tmp_path / "moe-Q4_K_M.gguf"
    baseline.write_bytes(b"GGUF-user-baseline")
    harness.spec.baseline_gguf = str(baseline)
    harness.spec.quants = ["Q4_K_M"]
    harness.spec.retention = [0.75]

    harness.run()
    harness.run()  # resume

    assert harness.counts["build_baseline"] == 1, "the baseline must resume, not rebuild"
    with _state(harness) as state:
        rows = {(j["stage"], j["key"]): j["status"] for j in state.jobs()}
        assert rows[("convert", "baseline-q4_k_m")] == "done"
        assert "running" not in rows.values()


def test_no_baseline_sweep_still_reports(harness):
    harness.spec.include_baseline = False
    report_path = harness.run()
    md = report_path.read_text(encoding="utf-8")
    assert "baseline-q4_k_m" not in md
    assert "r0.75-q4_k_m" in md
    # without a baseline, relative gates are "not measured" and pass, so a
    # winner is still selected on weighted score
    winner_line = next(line for line in md.splitlines() if "**Winner:**" in line)
    assert "r0.75-q4_k_m" in winner_line
    assert harness.baseline_responses_seen == [None, None]
    assert "## Notes" not in md  # a deliberate no-baseline sweep needs no caveat


def test_missing_matching_quant_baseline_is_not_substituted(harness):
    """[5]: with a baseline at Q4 only (e.g. a user-supplied baseline_gguf), the Q5
    candidate must be scored WITHOUT a baseline — no cross-quant substitution — noted
    in the report, and judged without baseline responses."""
    harness.spec.quants = ["Q4_K_M", "Q5_K_M"]
    harness.spec.retention = [0.75]
    harness.baseline_quants = ["Q4_K_M"]

    report_path = harness.run()
    md = report_path.read_text(encoding="utf-8")

    assert "## Notes" in md
    notes = md.split("## Notes")[1].split("##")[0]
    assert "r0.75-q5_k_m" in notes
    assert "no baseline at quant Q5_K_M" in notes
    # the Q5 row reports retention as unmeasured rather than a bogus cross-quant ratio
    q5_line = next(line for line in md.splitlines() if "`r0.75-q5_k_m`" in line)
    assert "n/a" in q5_line
    # and the Q5 candidate was judged without baseline responses; Q4 got them
    assert {"ev-1": "baseline response"} in harness.baseline_responses_seen
    assert None in harness.baseline_responses_seen
