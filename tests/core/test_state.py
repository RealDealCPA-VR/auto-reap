from __future__ import annotations

from reaplab.core.state import StateDB


def test_job_lifecycle_and_resume(tmp_path):
    db_path = tmp_path / "state.db"
    with StateDB(db_path) as db:
        assert db.status("prune", "r0.50") is None
        db.mark_running("prune", "r0.50")
        assert db.status("prune", "r0.50") == "running"
        db.mark_done("prune", "r0.50", meta={"path": "artifacts/r0.50"})
        assert db.is_done("prune", "r0.50")
        assert db.meta("prune", "r0.50") == {"path": "artifacts/r0.50"}

    # resume: a fresh connection sees completed work
    with StateDB(db_path) as db:
        assert db.is_done("prune", "r0.50")
        assert not db.is_done("prune", "r0.625")


def test_failure_isolation(tmp_path):
    with StateDB(tmp_path / "s.db") as db:
        db.mark_running("eval", "r0.50-q4")
        db.mark_failed("eval", "r0.50-q4", "server crashed")
        assert db.status("eval", "r0.50-q4") == "failed"
        jobs = db.jobs("eval")
        assert jobs[0]["error"] == "server crashed"
        # retry transitions back to running then done
        db.mark_running("eval", "r0.50-q4")
        db.mark_done("eval", "r0.50-q4")
        assert db.is_done("eval", "r0.50-q4")


def test_metrics(tmp_path):
    with StateDB(tmp_path / "s.db") as db:
        db.record_metric("r0.50-q4_k_m", "weighted_score", 0.91)
        db.record_metric("r0.50-q4_k_m", "weighted_score", 0.93)  # upsert
        db.record_metric("r0.50-q4_k_m", "gate_status", "pass")
        assert db.metrics_for("r0.50-q4_k_m") == {"weighted_score": 0.93, "gate_status": "pass"}
        assert db.all_artifacts() == ["r0.50-q4_k_m"]
