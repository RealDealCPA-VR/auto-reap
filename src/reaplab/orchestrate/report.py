"""Markdown sweep report (PRD FR-4.3).

Plain markdown only — the report must render cleanly both in a terminal
(`rich.markdown` / plain cat) and on GitHub, so no HTML, no images.

Sections: header, ranked table, winner callout, notes, per-domain breakdown,
Pareto front, regression diff vs baseline, anomalies, failed configs, manual
steps pending, promotion command footer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from reaplab.core.config import DomainPack, SweepSpec
from reaplab.core.paths import Workspace
from reaplab.orchestrate.scoring import SPECIAL_DOMAINS, GateResult, pareto_front


class ArtifactRow(BaseModel):
    """Everything the report needs about one evaluated artifact.

    The orchestrator builds one row per baseline and per candidate; the
    renderer never recomputes scores — it only formats what it is given.
    """

    artifact_id: str
    weighted: float
    retention_vs_baseline: float | None = None  # ratio (1.0 = parity); None = no baseline
    peak_vram_gb: float | None = None
    decode_tps: float | None = None
    gates: list[GateResult] = Field(default_factory=list)
    domain_scores: dict[str, float] = Field(default_factory=dict)
    regressions: dict[str, float] = Field(default_factory=dict)  # +pts = drop vs baseline
    false_refusal_rate: float | None = None
    baseline_false_refusal_rate: float | None = None
    is_baseline: bool = False
    #: Caveats that qualify this row's numbers (e.g. "no matching-quant baseline"),
    #: surfaced verbatim in the report's Notes section.
    notes: list[str] = Field(default_factory=list)

    @property
    def gates_pass(self) -> bool:
        """True when every blocking gate passed (advisory failures ignored)."""
        return all(g.passed for g in self.gates if g.blocking)


def _fmt(value: float | None, spec: str = ".3f", suffix: str = "") -> str:
    return "n/a" if value is None else format(value, spec) + suffix


def _ordered_domains(pack: DomainPack, rows: list[ArtifactRow]) -> list[str]:
    """Pack order first, then extra measured domains, special domains last."""
    seen: set[str] = set()
    ordered: list[str] = []
    measured = {d for r in rows for d in r.domain_scores}
    for d in pack.domains:
        if d.name in measured and d.name not in seen:
            ordered.append(d.name)
            seen.add(d.name)
    for name in sorted(measured - seen - SPECIAL_DOMAINS):
        ordered.append(name)
        seen.add(name)
    for name in sorted(measured & SPECIAL_DOMAINS):
        ordered.append(name)
    return ordered


def render_report(
    spec: SweepSpec,
    config_hash: str,
    pack: DomainPack,
    rows: list[ArtifactRow],
    winner_id: str | None,
    failed_jobs: list[dict[str, Any]] | None = None,
    manual_steps: list[dict[str, Any]] | None = None,
    build_seconds: float | None = None,
) -> str:
    """Render the full sweep report as a markdown string.

    Args:
        spec: the sweep spec (model, grid, gates — used for header + limits).
        config_hash: reproducibility key of this sweep.
        pack: domain pack (domain ordering + names).
        rows: one ArtifactRow per evaluated artifact (baselines included).
        winner_id: artifact_id chosen by select_winner, or None.
        failed_jobs: StateDB job dicts with status == "failed"
            (keys: stage, key, error) for the failed-configs section.
        manual_steps: stages waiting on a human action (keys: stage, key,
            instructions) — e.g. a remote prune whose script must be run by
            hand. Rendered as "Manual steps pending"; these are NOT failures.
        build_seconds: total artifact build wall-clock summed across manifests
            (PRD §5's advisory "full sweep fits in an overnight window" metric).
    """
    gates_cfg = spec.gates
    ranked = sorted(rows, key=lambda r: r.weighted, reverse=True)
    candidates = [r for r in ranked if not r.is_baseline]
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# reap-lab sweep report",
        "",
        f"- **Model:** `{spec.model_id}`",
        f"- **Config hash:** `{config_hash}`",
        f"- **Date:** {now}",
        f"- **Sweep grid:** retention {spec.retention} x quants {spec.quants}",
        f"- **Domain pack:** `{pack.name}`",
    ]
    if build_seconds is not None:
        hours = build_seconds / 3600.0
        window = "within the overnight window" if hours <= 12 else "OVER the 12 h overnight window"
        lines.append(f"- **Artifact build time:** {hours:.2f} h — {window} (advisory)")
    lines += [
        "",
        "## Ranked candidates",
        "",
        "| # | Artifact | Weighted score | Retention vs baseline | Peak VRAM (GB) | Decode tok/s | Gates |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, row in enumerate(ranked, 1):
        if row.is_baseline:
            gate_cell = "baseline"
            retention_cell = "100.0%"
        else:
            gate_cell = "PASS" if row.gates_pass else "FAIL"
            retention_cell = (
                f"{row.retention_vs_baseline * 100:.1f}%"
                if row.retention_vs_baseline is not None
                else "n/a"
            )
        lines.append(
            f"| {rank} | `{row.artifact_id}` | {row.weighted:.4f} | {retention_cell} "
            f"| {_fmt(row.peak_vram_gb, '.1f')} | {_fmt(row.decode_tps, '.1f')} | {gate_cell} |"
        )

    lines.append("")
    if winner_id:
        lines.append(
            f"**Winner:** `{winner_id}` — highest weighted score among candidates "
            "passing all blocking gates."
        )
    else:
        lines.append(
            "**Winner:** none — no candidate passed all blocking gates. "
            "Consider a higher retention ratio, a gentler quant, or reviewing the "
            "failed gates in the table above."
        )

    # Row notes (caveats that qualify the numbers above) -----------------------
    noted = [r for r in ranked if r.notes]
    if noted:
        lines += ["", "## Notes", ""]
        for row in noted:
            for note in row.notes:
                lines.append(f"- `{row.artifact_id}`: {note}")

    # Per-domain breakdown ---------------------------------------------------
    domains = _ordered_domains(pack, ranked)
    lines += ["", "## Per-domain breakdown", ""]
    if domains and ranked:
        lines.append("| Artifact | " + " | ".join(f"`{d}`" for d in domains) + " |")
        lines.append("|---|" + "---|" * len(domains))
        for row in ranked:
            cells = " | ".join(_fmt(row.domain_scores.get(d)) for d in domains)
            lines.append(f"| `{row.artifact_id}` | {cells} |")
    else:
        lines.append("No per-domain scores recorded.")

    # Pareto front -----------------------------------------------------------
    lines += ["", "## Pareto front (quality vs VRAM vs speed)", ""]
    if ranked:
        points = [
            {
                "artifact_id": r.artifact_id,
                "quality": r.weighted,
                "vram": r.peak_vram_gb,
                "tps": r.decode_tps,
            }
            for r in ranked
        ]
        front = pareto_front(points)
        for row in ranked:
            if row.artifact_id in front:
                lines.append(
                    f"- `{row.artifact_id}` — quality {row.weighted:.4f}, "
                    f"VRAM {_fmt(row.peak_vram_gb, '.1f')} GB, "
                    f"decode {_fmt(row.decode_tps, '.1f')} tok/s"
                )
    else:
        lines.append("No artifacts evaluated.")

    # Regression diff vs baseline ---------------------------------------------
    lines += ["", "## Regression vs baseline", ""]
    reg_rows = [r for r in candidates if r.regressions]
    if reg_rows:
        reg_domains = sorted({d for r in reg_rows for d in r.regressions})
        lines.append("Positive = points dropped vs the matching-quant baseline (0-100 scale).")
        lines.append("")
        lines.append("| Artifact | " + " | ".join(f"`{d}`" for d in reg_domains) + " |")
        lines.append("|---|" + "---|" * len(reg_domains))
        for row in reg_rows:
            cells = " | ".join(
                (f"{row.regressions[d]:+.1f}" if d in row.regressions else "n/a")
                for d in reg_domains
            )
            lines.append(f"| `{row.artifact_id}` | {cells} |")
    else:
        lines.append("No baseline comparison available.")

    # Anomalies ----------------------------------------------------------------
    lines += ["", "## Anomalies", ""]
    anomalies: list[str] = []
    for row in candidates:
        for domain, pts in sorted(row.regressions.items()):
            if domain not in SPECIAL_DOMAINS and pts > gates_cfg.max_domain_regression_pts:
                anomalies.append(
                    f"- `{row.artifact_id}`: domain `{domain}` dropped {pts:.1f} pts vs "
                    f"baseline (limit {gates_cfg.max_domain_regression_pts:g})"
                )
        if (
            row.false_refusal_rate is not None
            and row.baseline_false_refusal_rate is not None
            and row.false_refusal_rate > row.baseline_false_refusal_rate
        ):
            anomalies.append(
                f"- `{row.artifact_id}`: false-refusal rate regressed vs baseline "
                f"({row.false_refusal_rate:.1%} > {row.baseline_false_refusal_rate:.1%})"
            )
    lines += anomalies if anomalies else ["None detected."]

    # Failed configs -------------------------------------------------------------
    lines += ["", "## Failed configs", ""]
    failed_jobs = failed_jobs or []
    if failed_jobs:
        for job in failed_jobs:
            error = job.get("error") or "unknown error"
            lines.append(f"- `{job.get('stage')}:{job.get('key')}` — {error}")
    else:
        lines.append("None.")

    # Manual steps pending ---------------------------------------------------------
    manual_steps = manual_steps or []
    if manual_steps:
        lines += ["", "## Manual steps pending", ""]
        lines.append(
            "These stages are waiting on YOU, not on a failure. Do the step, then re-run "
            "`uv run reap-lab sweep <your-sweep.yaml>` — completed work resumes automatically."
        )
        for step in manual_steps:
            lines += ["", f"### `{step.get('stage')}:{step.get('key')}`", ""]
            instructions = str(step.get("instructions") or "").strip()
            lines += [f"    {line}" for line in instructions.splitlines()] or ["    (no details)"]

    # Promotion footer -------------------------------------------------------------
    lines += [
        "",
        "## Promotion",
        "",
        "To place the winner in LM Studio (with decision page, smoke test, and archival):",
        "",
        "    uv run reap-lab promote <your-sweep.yaml>",
        "",
    ]
    if winner_id:
        lines += [
            f"That promotes the stored winner, `{winner_id}`. To promote a different artifact "
            "from this sweep instead:",
            "",
            "    uv run reap-lab promote <your-sweep.yaml> --artifact <artifact-id>",
            "",
        ]
    else:
        lines += [
            "No candidate passed all blocking gates, so `promote` has no winner to place. "
            "Promote a specific artifact anyway (at your own risk) with:",
            "",
            "    uv run reap-lab promote <your-sweep.yaml> --artifact <artifact-id>",
            "",
        ]
    return "\n".join(lines)


def write_report(workspace: Workspace, config_hash: str, md: str) -> Path:
    """Write the rendered report to ``workspace.reports/sweep-<hash>.md``.

    Returns the written path. Overwrites any previous report for the same
    config hash (the content is deterministic given the same results).
    """
    workspace.reports.mkdir(parents=True, exist_ok=True)
    path = workspace.reports / f"sweep-{config_hash}.md"
    path.write_text(md, encoding="utf-8", newline="\n")
    return path
