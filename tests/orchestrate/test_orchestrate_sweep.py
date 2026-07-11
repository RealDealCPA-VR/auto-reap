"""End-to-end sweep engine tests with injected fake stages: happy path,
failure isolation, resume, disk guards, and the promote hand-off."""

from __future__ import annotations

from pathlib import Path

import pytest

from reaplab.core.config import PromoteCfg


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


def test_failure_isolation_one_bad_retention_does_not_kill_the_sweep(harness):
    harness.fail_retentions = {0.5}
    report_path = harness.run()
    md = report_path.read_text(encoding="utf-8")
    # the failed retention's artifacts never appear as candidates
    assert "r0.5-q4_k_m" not in md
    # ...but the failure is reported with its stage key and error
    assert "prune:r0.5" in md
    assert "synthetic prune failure" in md
    # the healthy retention is still ranked and wins
    assert "r0.75-q4_k_m" in md
    winner_line = next(line for line in md.splitlines() if "**Winner:**" in line)
    assert "r0.75-q4_k_m" in winner_line


def test_resume_skips_completed_stages(harness):
    harness.run()
    first_counts = dict(harness.counts)
    report_path = harness.run()  # same spec, same workspace, same state db
    assert report_path.exists()
    assert dict(harness.counts) == first_counts, "resume must not re-run finished stages"


def test_resume_false_forces_full_rerun(harness):
    harness.run()
    harness.run(resume=False)
    assert harness.counts["datagen"] == 2
    assert harness.counts["build_baseline"] == 2
    assert harness.counts["build_artifacts"] == 4
    assert harness.counts["evaluate"] == 6


def test_initial_disk_guard_aborts_instructively(harness):
    harness.spec.min_free_disk_gb = 10**9  # absurd requirement no machine meets
    with pytest.raises(RuntimeError) as excinfo:
        harness.run()
    message = str(excinfo.value)
    assert "min_free_disk_gb" in message
    assert "GB" in message
    assert harness.counts["datagen"] == 0  # aborted before any work


def test_mid_sweep_disk_recheck_isolates_candidates(harness, monkeypatch):
    harness.spec.min_free_disk_gb = 10.0
    calls = {"n": 0}

    def fake_free(path):
        calls["n"] += 1
        return 1000.0 if calls["n"] == 1 else 0.0  # initial guard passes, re-checks fail

    monkeypatch.setattr("reaplab.orchestrate.sweep.free_disk_gb", fake_free)
    report_path = harness.run()
    md = report_path.read_text(encoding="utf-8")
    # both candidate evals were skipped and recorded as failed
    assert "eval:r0.75-q4_k_m" in md
    assert "eval:r0.5-q4_k_m" in md
    assert "insufficient disk" in md
    # the baseline (evaluated before the re-checks) is still in the report
    assert "baseline-q4_k_m" in md
    assert "no candidate passed all blocking gates" in md
    # only the baseline was evaluated
    assert harness.counts["evaluate"] == 1


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
    assert (workspace / "artifacts" / "r0.75-q4_k_m.gguf").exists()
    assert not (workspace / "artifacts" / "r0.5-q4_k_m.gguf").exists()
    assert (workspace / "archive" / "r0.5-q4_k_m.gguf").exists()
    # a decision page was written alongside the report
    decisions = list((workspace / "reports").glob("decision-*.md"))
    assert decisions, "promotion must write a decision page"


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
