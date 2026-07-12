"""sweep_status summary tests."""

from __future__ import annotations

from reaplab.orchestrate import sweep_status


def test_status_before_any_run_is_instructive(harness):
    text = sweep_status(harness.spec)
    assert "No sweep state" in text
    assert "reap-lab sweep" in text


def test_status_after_run_shows_stages_and_metrics(harness):
    harness.run()
    text = sweep_status(harness.spec)
    assert "done" in text
    assert "prune:r0.75" in text
    assert "eval:r0.75-q4_k_m" in text
    assert "weighted_score" in text
    assert "r0.75-q4_k_m" in text
    assert "failed" in text  # the counts line always names failed


def test_status_reports_failures_with_errors(harness):
    harness.fail_retentions = {0.5}
    harness.run()
    text = sweep_status(harness.spec)
    # the component's own prune:r0.5 row AND run_sweep's coarse sweep:r0.5 row
    assert "2 failed" in text
    assert "prune:r0.5" in text
    assert "sweep:r0.5" in text
    assert "synthetic prune failure" in text


def test_status_shows_manual_steps_distinctly_from_failures(harness):
    harness.manual_retentions = {0.5}
    harness.run()
    text = sweep_status(harness.spec)
    assert "1 awaiting a manual step" in text
    assert "0 failed" in text
    assert "[ manual] prune:r0.5" in text
