"""reap-lab CLI entry point (typer). One command per pipeline stage plus
`sweep` (everything), `demo` (offline proof), `init` (wizard), `doctor` (checkup)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from reaplab import __version__

app = typer.Typer(
    name="reap-lab",
    no_args_is_help=True,
    help=(
        "Prune any Mixture-of-Experts model to fit your GPU: domain-tuned calibration, "
        "REAP expert pruning, GGUF evaluation, ranked reports - one command.\n\n"
        "New here? Run `reap-lab demo` (offline, ~1 min), then `reap-lab init`."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console(markup=True, highlight=False)

SPEC_ARG = typer.Argument(..., help="Path to a sweep spec YAML (see `reap-lab init`)", exists=True)


def _load_spec(spec_path: Path):
    from reaplab.core.config import SweepSpec  # noqa: PLC0415

    try:
        return SweepSpec.from_yaml(spec_path)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Invalid sweep spec {spec_path}:[/red] {e}")
        raise typer.Exit(1) from e


def _fail(message: str, code: int = 1) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code)


@app.callback()
def _version_callback() -> None:
    """reap-lab CLI."""


@app.command()
def version() -> None:
    """Print the reap-lab version."""
    console.print(f"reap-lab {__version__}")


@app.command()
def demo(
    workspace: Path = typer.Option(Path("reap-lab-demo"), help="Where demo artifacts land"),
    calibration_size: int = typer.Option(150, min=20, help="Calibration items to generate"),
    eval_size: int = typer.Option(150, min=20, help="Eval items to generate"),
    show_report: bool = typer.Option(True, help="Render the report to the terminal"),
) -> None:
    """Run the FULL pipeline offline with deterministic mocks (~1 minute, no GPU/keys)."""
    from reaplab.cli.demo import run_demo  # noqa: PLC0415

    try:
        run_demo(console, workspace, calibration_size, eval_size, show_report)
    except Exception as e:  # noqa: BLE001
        _fail(f"demo failed: {e}")


@app.command()
def init(
    out_dir: Path = typer.Option(Path("."), "--out", help="Directory for the generated YAMLs"),
    name: Optional[str] = typer.Option(None, help="Project name (kebab-case)"),
    model_id: Optional[str] = typer.Option(None, help="Base MoE model HF id"),
    describe: Optional[str] = typer.Option(None, help="Plain-English workload description"),
    provider: Optional[str] = typer.Option(
        None, help="Provider kind: claude-cli | openai-compat | anthropic-api | mock"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Non-interactive; use flags/defaults"),
) -> None:
    """Draft a domain pack + sweep spec for YOUR workload (wizard)."""
    from reaplab.cli.wizard import run_init  # noqa: PLC0415

    try:
        run_init(
            console, out_dir, name=name, model_id=model_id,
            describe=describe, provider=provider, yes=yes,
        )
    except typer.BadParameter:
        raise
    except Exception as e:  # noqa: BLE001
        _fail(f"init failed: {e}")


@app.command()
def doctor(
    spec: Optional[Path] = typer.Argument(None, help="Optional sweep YAML to validate against"),
    strict: bool = typer.Option(False, help="Exit 1 when any check FAILs"),
) -> None:
    """Check the environment: providers, llama.cpp, GPU, LM Studio, disk."""
    from reaplab.cli.doctor import run_doctor  # noqa: PLC0415

    raise typer.Exit(run_doctor(console, spec, strict))


@app.command()
def generate(spec_path: Path = SPEC_ARG) -> None:
    """Generate calibration + eval datasets only (then audit the printed sample)."""
    from reaplab.core.paths import Workspace  # noqa: PLC0415
    from reaplab.datagen import AUDIT_SAMPLE_FILENAME, generate_datasets  # noqa: PLC0415

    spec = _load_spec(spec_path)
    workspace = Workspace(spec.workspace).ensure()
    try:
        cal, ev = generate_datasets(spec, workspace)
    except Exception as e:  # noqa: BLE001
        _fail(f"dataset generation failed: {e}")
        return
    console.print(f"[green]calibration:[/green] {cal}")
    console.print(f"[green]eval:[/green] {ev}")
    console.print(f"[green]audit sample:[/green] {workspace.data / AUDIT_SAMPLE_FILENAME}")
    console.print("Review the audit sample (PRD M1: a 5% human check) before running the sweep.")


@app.command()
def audit(spec_path: Path = SPEC_ARG) -> None:
    """Show the human-audit sample of the generated eval set."""
    from reaplab.core.paths import Workspace  # noqa: PLC0415
    from reaplab.datagen import AUDIT_SAMPLE_FILENAME  # noqa: PLC0415

    spec = _load_spec(spec_path)
    sample = Workspace(spec.workspace).data / AUDIT_SAMPLE_FILENAME
    if not sample.exists():
        _fail(f"No audit sample at {sample}. Run `reap-lab generate {spec_path}` first.")
    from rich.markdown import Markdown  # noqa: PLC0415

    console.print(Markdown(sample.read_text(encoding="utf-8")))


@app.command()
def sweep(
    spec_path: Path = SPEC_ARG,
    resume: bool = typer.Option(True, help="Reuse completed stages from earlier runs"),
    promote: bool = typer.Option(False, help="Promote the winner to LM Studio when gates pass"),
) -> None:
    """Run the full pipeline: data -> prune -> GGUF -> eval -> report (G3's one command)."""
    _run_sweep(spec_path, resume=resume, promote=promote)


@app.command()
def report(spec_path: Path = SPEC_ARG) -> None:
    """Re-render the report from completed stages (no new work)."""
    _run_sweep(spec_path, resume=True, promote=False)


@app.command()
def promote(spec_path: Path = SPEC_ARG) -> None:
    """Promote the sweep winner: copy to LM Studio, decision page, smoke test, archive."""
    _run_sweep(spec_path, resume=True, promote=True)


def _run_sweep(spec_path: Path, *, resume: bool, promote: bool) -> None:
    from reaplab.orchestrate import run_sweep  # noqa: PLC0415
    from reaplab.prune import NeedsManualStep  # noqa: PLC0415

    spec = _load_spec(spec_path)
    try:
        report_path = run_sweep(spec, resume=resume, promote=promote)
    except NeedsManualStep as e:
        console.print("[yellow]Manual step required (remote prune without ssh_host):[/yellow]")
        console.print(str(e))
        raise typer.Exit(2) from e
    except Exception as e:  # noqa: BLE001
        _fail(f"sweep failed: {e}")
        return
    console.print(f"\n[green]Report:[/green] {report_path}")


@app.command()
def prune(
    spec_path: Path = SPEC_ARG,
    retention: float = typer.Option(..., min=0.01, max=1.0, help="Expert retention, e.g. 0.5"),
) -> None:
    """Prune + convert one retention point (datasets are generated if missing)."""
    from reaplab.core.paths import Workspace  # noqa: PLC0415
    from reaplab.core.state import StateDB  # noqa: PLC0415
    from reaplab.datagen import generate_datasets  # noqa: PLC0415
    from reaplab.prune import NeedsManualStep, build_artifacts  # noqa: PLC0415

    spec = _load_spec(spec_path)
    config_hash = spec.config_hash()
    workspace = Workspace(spec.workspace).ensure(config_hash)
    try:
        cal, _ = generate_datasets(spec, workspace)
        with StateDB(workspace.state_db(config_hash)) as state:
            manifests = build_artifacts(spec, retention, cal, workspace, state)
    except NeedsManualStep as e:
        console.print("[yellow]Manual step required:[/yellow]")
        console.print(str(e))
        raise typer.Exit(2) from e
    except Exception as e:  # noqa: BLE001
        _fail(f"prune failed: {e}")
        return
    for m in manifests:
        console.print(f"[green]{m.artifact_id}[/green] -> {m.path}")


@app.command()
def convert(spec_path: Path = SPEC_ARG) -> None:
    """Build/convert the unpruned baseline GGUF(s) for the quant grid."""
    from reaplab.core.paths import Workspace  # noqa: PLC0415
    from reaplab.core.state import StateDB  # noqa: PLC0415
    from reaplab.prune import build_baseline  # noqa: PLC0415

    spec = _load_spec(spec_path)
    config_hash = spec.config_hash()
    workspace = Workspace(spec.workspace).ensure(config_hash)
    try:
        with StateDB(workspace.state_db(config_hash)) as state:
            manifests = build_baseline(spec, workspace, state)
    except Exception as e:  # noqa: BLE001
        _fail(f"baseline conversion failed: {e}")
        return
    for m in manifests:
        console.print(f"[green]{m.artifact_id}[/green] -> {m.path}")


@app.command("eval")
def eval_cmd(
    spec_path: Path = SPEC_ARG,
    gguf: Path = typer.Option(..., exists=True, help="A GGUF to evaluate against your eval set"),
    artifact_id: Optional[str] = typer.Option(None, help="Label for the results (default: filename)"),
) -> None:
    """Evaluate ANY local GGUF against your domain pack's eval set.

    The buy-vs-build shortcut: score a pre-pruned community checkpoint first -
    if it clears your bar, you may not need a custom prune run at all.
    """
    from reaplab.core.jsonl import read_jsonl  # noqa: PLC0415
    from reaplab.core.paths import Workspace  # noqa: PLC0415
    from reaplab.core.records import ArtifactManifest, EvalRecord  # noqa: PLC0415
    from reaplab.core.state import StateDB  # noqa: PLC0415
    from reaplab.datagen import generate_datasets  # noqa: PLC0415
    from reaplab.evalharness import evaluate_artifact  # noqa: PLC0415
    from reaplab.prune import detect_quant_from_name  # noqa: PLC0415

    spec = _load_spec(spec_path)
    config_hash = spec.config_hash()
    workspace = Workspace(spec.workspace).ensure(config_hash)
    try:
        _, eval_path = generate_datasets(spec, workspace)
        records = read_jsonl(eval_path, EvalRecord)
        manifest = ArtifactManifest(
            artifact_id=artifact_id or gguf.stem.lower().replace(" ", "-"),
            kind="gguf",
            model_id=spec.model_id,
            quant=detect_quant_from_name(gguf.name),
            path=str(gguf),
            config_hash=config_hash,
        )
        with StateDB(workspace.state_db(config_hash)) as state:
            summary = evaluate_artifact(spec, manifest, records, workspace, state)
    except Exception as e:  # noqa: BLE001
        _fail(f"eval failed: {e}")
        return

    table = Table(title=f"eval: {manifest.artifact_id} ({summary['items_scored']} items)")
    table.add_column("domain")
    table.add_column("score", justify="right")
    table.add_column("items", justify="right")
    for domain, score in sorted(summary["domain_scores"].items()):
        table.add_row(domain, f"{score:.3f}", str(summary["counts"].get(domain, "")))
    console.print(table)
    for key in ("false_refusal_rate", "should_refuse_pass_rate", "tool_call_validity"):
        if summary.get(key) is not None:
            console.print(f"{key}: {summary[key]:.3f}")
    console.print(
        "\nNote: open-ended domains score via the judge only when a baseline exists in the same "
        "sweep; standalone evals use the non-refusal heuristic for those domains."
    )


@app.command()
def status(spec_path: Path = SPEC_ARG) -> None:
    """Show sweep progress: stages done/failed/running and per-artifact metrics."""
    from reaplab.orchestrate import sweep_status  # noqa: PLC0415

    spec = _load_spec(spec_path)
    console.print(sweep_status(spec))


if __name__ == "__main__":
    app()
