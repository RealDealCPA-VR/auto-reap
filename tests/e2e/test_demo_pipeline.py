"""End-to-end proof: `reap-lab demo` executes the entire pipeline offline and the
result is a coherent, gated, promoted sweep — the build's primary validation
vehicle (real GPU runs are the user's milestone M2+)."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from reaplab.cli.main import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(result) -> str:
    """CLI output with ANSI codes stripped and wrapping collapsed.

    Rich picks its width from the terminal and its color mode from the environment,
    so rendered output differs between a dev box, Docker, and a CI runner: it wraps
    mid-phrase and injects escape codes between tokens. Assertions on raw `.output`
    therefore test Rich's layout, not our text. Normalize first.
    """
    text = getattr(result, "output", result)
    return " ".join(_ANSI.sub("", text).split())



def test_demo_end_to_end_with_promotion(tmp_path):
    ws = tmp_path / "demo"
    result = runner.invoke(app, ["demo", "--workspace", str(ws), "--no-show-report"])
    assert result.exit_code == 0, result.output

    reports = list((ws / "workspace" / "reports").glob("sweep-*.md"))
    assert len(reports) == 1
    md = reports[0].read_text(encoding="utf-8")

    # every grid artifact ranked, baseline included
    for aid in (
        "baseline-q4_k_m", "baseline-q5_k_m",
        "r0.75-q4_k_m", "r0.75-q5_k_m",
        "r0.625-q4_k_m", "r0.5-q4_k_m",
    ):
        assert aid in md, f"{aid} missing from report"
    assert "## Ranked candidates" in md
    assert "Pareto front" in md
    assert "**Winner:**" in md

    # the demo curve is tuned so r0.75 passes all gates and gets promoted into
    # the sandboxed LM Studio dir with the REQUIRED two-level layout
    assert "`r0.75-q4_k_m`" in md.split("**Winner:**")[1].splitlines()[0]
    promoted = list((ws / "lmstudio-models").rglob("*.gguf"))
    assert len(promoted) == 1
    rel = promoted[0].relative_to(ws / "lmstudio-models")
    assert len(rel.parts) == 3, f"LM Studio layout must be publisher/model/file.gguf, got {rel}"

    # decision page written next to the report
    decisions = list((ws / "workspace" / "reports").glob("decision-*.md"))
    assert decisions, "promotion must write a decision page"

    # the demo also leaves an editable example sweep spec behind
    assert (ws / "demo-sweep.yaml").exists()
    assert (ws / "demo-pack.yaml").exists()


def test_demo_datasets_live_under_the_per_sweep_run_dir(tmp_path):
    """C1: datasets are keyed by config hash (runs/<hash>/data), never at a shared
    workspace/data path a second sweep could overwrite."""
    ws = tmp_path / "demo"
    result = runner.invoke(app, ["demo", "--workspace", str(ws), "--no-show-report"])
    assert result.exit_code == 0, result.output

    workspace = ws / "workspace"
    eval_sets = list(workspace.glob("runs/*/data/eval_v1.jsonl"))
    assert len(eval_sets) == 1, "exactly one per-sweep eval set"
    run_dir = eval_sets[0].parent.parent
    assert (run_dir / "data" / "calibration_v1.jsonl").exists()
    assert (run_dir / "state.db").exists()  # datasets sit next to the sweep's state
    assert not (workspace / "data" / "eval_v1.jsonl").exists(), "old shared location"


def test_report_command_is_a_true_rerender(tmp_path):
    """[34]: `reap-lab report` re-renders from the state DB — no datagen, no builds,
    no evals — and produces the same report the sweep wrote."""
    ws = tmp_path / "demo"
    assert runner.invoke(app, ["demo", "--workspace", str(ws), "--no-show-report"]).exit_code == 0
    report = next((ws / "workspace" / "reports").glob("sweep-*.md"))

    def stable(md: str) -> str:
        return "\n".join(line for line in md.splitlines() if not line.startswith("- **Date:**"))

    before = stable(report.read_text(encoding="utf-8"))
    result = runner.invoke(app, ["report", str(ws / "demo-sweep.yaml")])
    assert result.exit_code == 0, result.output
    assert "Report:" in plain(result)
    assert stable(report.read_text(encoding="utf-8")) == before
    # a re-render touches no dataset/artifact stage, so nothing new appears on disk
    assert len(list((ws / "workspace" / "reports").glob("sweep-*.md"))) == 1


def test_demo_is_deterministic_and_resumable(tmp_path):
    ws = tmp_path / "demo"
    first = runner.invoke(app, ["demo", "--workspace", str(ws), "--no-show-report"])
    assert first.exit_code == 0, first.output
    report = next((ws / "workspace" / "reports").glob("sweep-*.md"))
    first_md = report.read_text(encoding="utf-8")

    # second run over the same workspace resumes: same report content is
    # regenerated from completed stages (timestamps differ; strip the date line)
    second = runner.invoke(app, ["demo", "--workspace", str(ws), "--no-show-report"])
    assert second.exit_code == 0, second.output
    assert "reusing" in plain(second)
    second_md = report.read_text(encoding="utf-8")

    def stable(md: str) -> str:
        return "\n".join(line for line in md.splitlines() if not line.startswith("- **Date:**"))

    assert stable(first_md) == stable(second_md)
