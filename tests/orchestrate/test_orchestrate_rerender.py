"""State-only views: `reap-lab report` and `reap-lab promote` must rebuild their
rows from the StateDB alone — no datagen, no builds, no evals ([34]/[m3]/[m21]),
plus the eval-record cache key ([m5])."""

from __future__ import annotations

from pathlib import Path

import pytest

from reaplab.core.config import PromoteCfg
from reaplab.orchestrate import promote_from_state, render_report_from_state
from reaplab.orchestrate.sweep import _load_eval_records


def test_render_report_from_state_runs_no_stages(harness):
    harness.run()
    before = dict(harness.counts)

    report_path = render_report_from_state(harness.spec)

    assert dict(harness.counts) == before, "a re-render must not execute any stage"
    md = report_path.read_text(encoding="utf-8")
    # the rows are rebuilt from stored summaries + manifests, so the report is whole
    for artifact_id in ("baseline-q4_k_m", "r0.75-q4_k_m", "r0.5-q4_k_m"):
        assert artifact_id in md
    winner_line = next(line for line in md.splitlines() if "**Winner:**" in line)
    assert "r0.75-q4_k_m" in winner_line
    assert "## Ranked candidates" in md
    assert "Pareto front" in md


def test_render_report_from_state_matches_the_sweep_report(harness):
    sweep_md = harness.run().read_text(encoding="utf-8")
    rerender_md = render_report_from_state(harness.spec).read_text(encoding="utf-8")

    def stable(md: str) -> str:
        return "\n".join(line for line in md.splitlines() if not line.startswith("- **Date:**"))

    assert stable(sweep_md) == stable(rerender_md)


def test_render_report_from_state_without_a_sweep_is_instructive(harness):
    with pytest.raises(RuntimeError) as excinfo:
        render_report_from_state(harness.spec)
    message = str(excinfo.value)
    assert "Nothing has been evaluated yet" in message
    assert "reap-lab sweep" in message


def test_render_report_from_state_surfaces_manual_steps(harness):
    harness.manual_retentions = {0.5}
    harness.run()
    md = render_report_from_state(harness.spec).read_text(encoding="utf-8")
    # the 'manual' StateDB rows are re-read, so the instructions survive a re-render
    assert "## Manual steps pending" in md
    assert "prune_remote_r0.5.sh" in md


def test_promote_from_state_promotes_the_stored_winner(harness, tmp_path):
    lms_dir = tmp_path / "lms-models"
    harness.spec.promote = PromoteCfg(lmstudio_dir=str(lms_dir), publisher="reap-lab")
    harness.run()  # no promotion during the sweep
    before = dict(harness.counts)

    result = promote_from_state(harness.spec)

    assert result.ok, result.message
    assert dict(harness.counts) == before, "promotion must not re-run any stage"
    dest = lms_dir / "reap-lab" / "Qwen3-30B-A3B-r0.75-q4_k_m" / "Qwen3-30B-A3B-r0.75-q4_k_m.gguf"
    assert dest.exists()
    # exactly the evaluated loser was archived; the winner source stayed put
    workspace = Path(harness.spec.workspace)
    assert (workspace / "archive" / "r0.5-q4_k_m.gguf").exists()
    artifacts = workspace / "artifacts" / harness.spec.config_hash()
    assert (artifacts / "r0.75-q4_k_m.gguf").exists()
    assert (artifacts / "baseline-q4_k_m.gguf").exists()


def test_promote_from_state_artifact_override(harness, tmp_path):
    lms_dir = tmp_path / "lms-models"
    harness.spec.promote = PromoteCfg(
        lmstudio_dir=str(lms_dir), publisher="reap-lab", archive_losers=False
    )
    harness.run()

    result = promote_from_state(harness.spec, artifact_id="r0.5-q4_k_m")

    assert result.ok, result.message
    assert result.dest_path is not None
    assert "r0.5-q4_k_m" in str(result.dest_path)
    page = result.decision_page.read_text(encoding="utf-8")
    assert "Operator override" in page
    assert "r0.75-q4_k_m" in page  # names the gate-selected winner it displaced


def test_promote_from_state_unknown_artifact_is_instructive(harness, tmp_path):
    harness.spec.promote = PromoteCfg(lmstudio_dir=str(tmp_path / "lms"))
    harness.run()
    with pytest.raises(RuntimeError) as excinfo:
        promote_from_state(harness.spec, artifact_id="r0.9-q8_0")
    message = str(excinfo.value)
    assert "r0.9-q8_0" in message
    assert "r0.75-q4_k_m" in message  # lists what IS available


def test_promote_from_state_without_a_winner_is_instructive(harness, tmp_path):
    harness.spec.promote = PromoteCfg(lmstudio_dir=str(tmp_path / "lms"))
    harness.spec.gates.min_quality_retention = 1.5  # nothing can pass
    harness.run()
    with pytest.raises(RuntimeError) as excinfo:
        promote_from_state(harness.spec)
    message = str(excinfo.value)
    assert "no winner to promote" in message
    assert "--artifact" in message


def test_promote_from_state_smoke_failure_reports_not_ok(harness, tmp_path):
    harness.spec.promote = PromoteCfg(
        lmstudio_dir=str(tmp_path / "lms-models"),
        publisher="reap-lab",
        smoke_command='python -c "import sys; sys.exit(7)"',
    )
    harness.run()
    result = promote_from_state(harness.spec)
    assert not result.ok  # the CLI turns this into exit code 1
    assert result.stage == "smoke"


def test_eval_record_cache_keys_on_content_not_just_path(tmp_path):
    """[m5]: `sweep --no-resume` regenerates the eval set AT THE SAME PATH; a
    name-only cache would serve the previous run's records to every later artifact."""
    path = tmp_path / "eval_v1.jsonl"
    row = (
        '{{"id": "{id}", "domain": "alpha", "task_type": "open_ended", '
        '"prompt": "p", "gold": null}}'
    )
    path.write_text(row.format(id="ev-1") + "\n", encoding="utf-8")
    first = _load_eval_records(str(path))
    assert [r.id for r in first] == ["ev-1"]

    path.write_text(
        row.format(id="ev-1") + "\n" + row.format(id="ev-2") + "\n", encoding="utf-8"
    )
    second = _load_eval_records(str(path))
    assert [r.id for r in second] == ["ev-1", "ev-2"], "stale cache served old records"
