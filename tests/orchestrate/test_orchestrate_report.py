"""Report rendering tests: ranked order, winner callout, anomalies, failures."""

from __future__ import annotations

import pytest

from reaplab.core.config import SweepSpec
from reaplab.orchestrate.report import ArtifactRow, render_report, write_report
from reaplab.orchestrate.scoring import GateResult


@pytest.fixture
def spec(tmp_path):
    return SweepSpec(
        model_id="Qwen/Qwen3-30B-A3B",
        domain_pack=str(tmp_path / "pack.yaml"),  # never loaded by the renderer
        retention=[0.75, 0.5],
        quants=["Q4_K_M"],
        workspace=str(tmp_path / "ws"),
    )


def passing_gates() -> list[GateResult]:
    return [
        GateResult(name="quality_retention", value=0.98, limit=0.95, passed=True),
        GateResult(name="should_refuse", value=1.0, limit=1.0, passed=True),
    ]


def failing_gates() -> list[GateResult]:
    return [
        GateResult(name="quality_retention", value=0.80, limit=0.95, passed=False),
        GateResult(name="domain_regression", value=20.0, limit=5.0, passed=False),
    ]


@pytest.fixture
def rows() -> list[ArtifactRow]:
    return [
        ArtifactRow(
            artifact_id="baseline-q4_k_m",
            weighted=0.875,
            retention_vs_baseline=1.0,
            peak_vram_gb=32.0,
            decode_tps=55.0,
            domain_scores={"alpha": 0.90, "beta": 0.80},
            is_baseline=True,
        ),
        ArtifactRow(
            artifact_id="r0.75-q4_k_m",
            weighted=0.86,
            retention_vs_baseline=0.983,
            peak_vram_gb=28.0,
            decode_tps=70.0,
            gates=passing_gates(),
            domain_scores={"alpha": 0.88, "beta": 0.80},
            regressions={"alpha": 2.0, "beta": 0.0},
            false_refusal_rate=0.01,
            baseline_false_refusal_rate=0.02,
        ),
        ArtifactRow(
            artifact_id="r0.5-q4_k_m",
            weighted=0.70,
            retention_vs_baseline=0.80,
            peak_vram_gb=22.0,
            decode_tps=90.0,
            gates=failing_gates(),
            domain_scores={"alpha": 0.70, "beta": 0.70},
            regressions={"alpha": 20.0, "beta": 10.0},
            false_refusal_rate=0.05,
            baseline_false_refusal_rate=0.02,
        ),
    ]


@pytest.fixture
def md(spec, pack, rows) -> str:
    failed = [
        {
            "stage": "prune",
            "key": "r0.625",
            "status": "failed",
            "error": "RuntimeError: synthetic gpu meltdown",
        }
    ]
    return render_report(spec, "cafe0123beef", pack, rows, "r0.75-q4_k_m", failed)


def test_header_and_grid(md):
    assert "Qwen/Qwen3-30B-A3B" in md
    assert "cafe0123beef" in md
    assert "Q4_K_M" in md
    assert "test-pack" in md


def test_ranked_order_by_weighted_score(md):
    # 0.875 baseline > 0.86 candidate A > 0.70 candidate B; the ranked table
    # is the first section that mentions artifact ids.
    assert (
        md.index("baseline-q4_k_m")
        < md.index("r0.75-q4_k_m")
        < md.index("r0.5-q4_k_m")
    )


def test_gate_pass_fail_column(md):
    assert "PASS" in md
    assert "FAIL" in md


def test_winner_callout(md):
    assert "Winner" in md
    winner_line = next(line for line in md.splitlines() if "**Winner:**" in line)
    assert "r0.75-q4_k_m" in winner_line


def test_no_winner_callout(spec, pack, rows):
    text = render_report(spec, "cafe0123beef", pack, rows, None, [])
    assert "no candidate passed all blocking gates" in text


def test_per_domain_breakdown(md):
    assert "Per-domain breakdown" in md
    assert "`alpha`" in md
    assert "`beta`" in md


def test_pareto_section(md):
    assert "Pareto front" in md
    # r0.5 has the lowest VRAM and highest tps -> on the front despite low quality
    pareto_part = md.split("Pareto front")[1].split("##")[0]
    assert "r0.5-q4_k_m" in pareto_part


def test_regression_diff_section(md):
    assert "Regression vs baseline" in md
    assert "+20.0" in md


def test_anomaly_section_flags_domain_drop_and_refusal_regression(md):
    anomalies = md.split("## Anomalies")[1].split("##")[0]
    assert "r0.5-q4_k_m" in anomalies
    assert "alpha" in anomalies  # 20 pts > 5 pt limit
    assert "false-refusal" in anomalies  # 5% > 2% baseline
    # the healthy candidate must not be flagged
    assert "r0.75-q4_k_m" not in anomalies


def test_failed_config_section(md):
    failed_part = md.split("## Failed configs")[1]
    assert "prune:r0.625" in failed_part
    assert "synthetic gpu meltdown" in failed_part


def test_failed_section_empty_says_none(spec, pack, rows):
    text = render_report(spec, "cafe0123beef", pack, rows, "r0.75-q4_k_m", [])
    failed_part = text.split("## Failed configs")[1].split("##")[0]
    assert "None." in failed_part


def test_promotion_footer_matches_the_real_cli(md):
    """[33]/[m1]/[m23]: the footer must be a command the CLI actually accepts."""
    from typer.testing import CliRunner

    from reaplab.cli.main import app

    assert "uv run reap-lab promote <your-sweep.yaml>" in md
    assert "--artifact <artifact-id>" in md
    assert "--artifact r0.75-q4_k_m" not in md  # the old, non-existent form
    # ...and --artifact really exists on the promote command
    help_text = CliRunner().invoke(app, ["promote", "--help"]).output
    assert "--artifact" in help_text


def test_notes_section_surfaces_row_caveats(spec, pack, rows):
    rows[1].notes = ["no baseline at quant Q5_K_M — relative gates not measured"]
    text = render_report(spec, "cafe0123beef", pack, rows, "r0.75-q4_k_m", [])
    notes = text.split("## Notes")[1].split("##")[0]
    assert "r0.75-q4_k_m" in notes
    assert "no baseline at quant Q5_K_M" in notes


def test_manual_steps_section_is_not_a_failure_section(spec, pack, rows):
    manual = [
        {
            "stage": "prune",
            "key": "r0.5",
            "instructions": "1. scp prune_remote_r0.5.sh gpu:~\n2. bash prune_remote_r0.5.sh",
        }
    ]
    text = render_report(spec, "cafe0123beef", pack, rows, "r0.75-q4_k_m", [], manual)
    assert "## Manual steps pending" in text
    manual_part = text.split("## Manual steps pending")[1].split("## Promotion")[0]
    assert "prune:r0.5" in manual_part
    assert "bash prune_remote_r0.5.sh" in manual_part
    failed_part = text.split("## Failed configs")[1].split("##")[0]
    assert "None." in failed_part


def test_no_html_in_report(md):
    assert "<table" not in md
    assert "<div" not in md


def test_empty_rows_render_without_crash(spec, pack):
    text = render_report(spec, "cafe0123beef", pack, [], None, [])
    assert "No artifacts evaluated" in text


def test_write_report_path_and_content(ws, md):
    path = write_report(ws, "cafe0123beef", md)
    assert path == ws.reports / "sweep-cafe0123beef.md"
    assert path.read_text(encoding="utf-8") == md
