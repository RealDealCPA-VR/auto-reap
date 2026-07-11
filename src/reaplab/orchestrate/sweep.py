"""Sweep engine (PRD FR-4.1/FR-4.2): retention x quant grid, resumable state,
failure isolation, disk guards, scoring, report, optional promotion.

Component boundaries: the heavy stages are injected callables so this module
never imports sibling components at import time (they may not exist yet, and
tests inject deterministic fakes). Expected callable shapes:

    datagen_fn(spec, workspace) -> (calibration_path: Path, eval_path: Path)
    build_baseline_fn(spec, workspace) -> list[ArtifactManifest]
        # one GGUF manifest per quant, artifact_id "baseline-<quant lowercase>"
    build_artifacts_fn(spec, workspace, retention: float, calibration_path: Path)
        -> list[ArtifactManifest]
        # one GGUF manifest per quant, artifact_id "r<retention:g>-<quant lowercase>"
    evaluate_fn(spec, workspace, manifest, eval_path, *, baseline_responses=None)
        -> summary dict (see reaplab.orchestrate.scoring module docstring);
        # baseline summaries additionally carry "responses": {item_id: text}
        # used for pairwise judging of candidates against the same quant.

StateDB stage/key naming (shared contract):
    ("datagen", "datasets"), ("prune", "r<retention:g>"),
    ("convert", "<artifact_id>"), ("eval", "<artifact_id>") — rendered as
    "stage:key" in reports, e.g. "prune:r0.5".
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

from reaplab.core.config import DomainPack, SweepSpec
from reaplab.core.paths import Workspace, free_disk_gb
from reaplab.core.records import ArtifactManifest
from reaplab.core.state import StateDB
from reaplab.orchestrate.promote import promote_winner
from reaplab.orchestrate.report import ArtifactRow, render_report, write_report
from reaplab.orchestrate.scoring import (
    domain_regressions,
    evaluate_gates,
    quality_retention,
    select_winner,
    weighted_score,
)

DatagenFn = Callable[..., tuple[Path, Path]]
BuildFn = Callable[..., list[ArtifactManifest]]
EvaluateFn = Callable[..., dict[str, Any]]


def _default_datagen() -> DatagenFn:
    try:
        from reaplab.datagen import generate_datasets  # noqa: PLC0415 - lazy by design
    except ImportError as e:
        raise RuntimeError(
            "The datagen component (reaplab.datagen.generate_datasets) is unavailable: "
            f"{e}. Reinstall reap-lab (`uv sync`) or pass datagen_fn explicitly."
        ) from e
    return generate_datasets


def _default_build_baseline() -> BuildFn:
    try:
        from reaplab.prune import build_baseline  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The prune component (reaplab.prune.build_baseline) is unavailable: "
            f"{e}. Reinstall reap-lab (`uv sync`) or pass build_baseline_fn explicitly."
        ) from e
    return build_baseline


def _default_build_artifacts() -> BuildFn:
    try:
        from reaplab.prune import build_artifacts  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The prune component (reaplab.prune.build_artifacts) is unavailable: "
            f"{e}. Reinstall reap-lab (`uv sync`) or pass build_artifacts_fn explicitly."
        ) from e
    return build_artifacts


def _default_evaluate() -> EvaluateFn:
    try:
        from reaplab.evalharness import evaluate_artifact  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The eval harness (reaplab.evalharness.evaluate_artifact) is unavailable: "
            f"{e}. Reinstall reap-lab (`uv sync`) or pass evaluate_fn explicitly."
        ) from e
    return evaluate_artifact


def _perf_field(summary: dict[str, Any], context: int, field: str) -> float | None:
    """Read one PerfMetrics field at the given context; falls back to the
    largest measured context so report rows stay informative on partial data."""
    raw = summary.get("perf") or {}
    perf: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        perf[str(key)] = dict(value)
    entry = perf.get(str(context))
    if entry is not None and entry.get(field) is not None:
        return float(entry[field])
    for key in sorted(perf, key=lambda s: int(s) if s.isdigit() else -1, reverse=True):
        value = perf[key].get(field)
        if value is not None:
            return float(value)
    return None


def _manifest_from_state(state: StateDB, artifact_id: str) -> ArtifactManifest:
    return ArtifactManifest.model_validate(state.meta("convert", artifact_id)["manifest"])


def run_sweep(
    spec: SweepSpec,
    resume: bool = True,
    datagen_fn: DatagenFn | None = None,
    build_baseline_fn: BuildFn | None = None,
    build_artifacts_fn: BuildFn | None = None,
    evaluate_fn: EvaluateFn | None = None,
    promote: bool = False,
) -> Path:
    """Run the full sweep end to end and return the written report path.

    Resumable: every completed stage is recorded in SQLite keyed by the spec's
    config hash; re-running the same spec skips finished work (``resume=False``
    forces a full re-run). Failure-isolated: one bad retention or artifact is
    marked failed and the sweep continues (PRD FR-4.2). Raises RuntimeError
    with instructive text only for fatal conditions (no disk, no datasets, no
    baseline when one was requested).
    """
    console = Console(markup=False, highlight=False)
    config_hash = spec.config_hash()
    workspace = Workspace(spec.workspace).ensure(config_hash)

    free = free_disk_gb(workspace.root)
    if free < spec.min_free_disk_gb:
        raise RuntimeError(
            f"Only {free:.1f} GB free on the volume holding {workspace.root}, but the sweep "
            f"requires min_free_disk_gb={spec.min_free_disk_gb:g}. Each candidate GGUF weighs "
            "roughly 15-35 GB. Free up space, point `workspace:` in the sweep YAML at a larger "
            "drive, or lower `min_free_disk_gb` if you know what you are doing."
        )

    pack = DomainPack.from_yaml(spec.domain_pack)
    datagen_fn = datagen_fn or _default_datagen()
    build_baseline_fn = build_baseline_fn or _default_build_baseline()
    build_artifacts_fn = build_artifacts_fn or _default_build_artifacts()
    evaluate_fn = evaluate_fn or _default_evaluate()

    console.print(f"reap-lab sweep {config_hash}: {spec.model_id} "
                  f"(retention {spec.retention} x quants {spec.quants})")

    state = StateDB(workspace.state_db(config_hash))
    try:
        # -- datasets ---------------------------------------------------------
        if resume and state.is_done("datagen", "datasets"):
            meta = state.meta("datagen", "datasets")
            cal_path, eval_path = Path(meta["calibration"]), Path(meta["eval"])
            console.print("datagen: reusing datasets from a previous run")
        else:
            state.mark_running("datagen", "datasets")
            try:
                cal_path, eval_path = datagen_fn(spec, workspace)
            except Exception as e:
                state.mark_failed("datagen", "datasets", f"{type(e).__name__}: {e}")
                raise RuntimeError(
                    f"Dataset generation failed: {e}. Nothing downstream can run without "
                    "datasets. Check the generator provider (`reap-lab doctor`) and re-run — "
                    "completed stages resume automatically."
                ) from e
            state.mark_done(
                "datagen", "datasets",
                meta={"calibration": str(cal_path), "eval": str(eval_path)},
            )
            console.print(f"datagen: calibration={cal_path} eval={eval_path}")

        # -- baseline ----------------------------------------------------------
        baseline_manifests: list[ArtifactManifest] = []
        baselines: dict[str, dict[str, Any]] = {}  # quant (lowercase) -> summary
        if spec.include_baseline:
            expected = [f"baseline-{q.lower()}" for q in spec.quants]
            if resume and all(state.is_done("convert", aid) for aid in expected):
                baseline_manifests = [_manifest_from_state(state, aid) for aid in expected]
                console.print("baseline: reusing converted baseline artifacts")
            else:
                for aid in expected:
                    state.mark_running("convert", aid)
                try:
                    baseline_manifests = build_baseline_fn(spec, workspace)
                except Exception as e:
                    for aid in expected:
                        state.mark_failed("convert", aid, f"{type(e).__name__}: {e}")
                    raise RuntimeError(
                        f"Baseline build failed: {e}. Quality-retention gates need the unpruned "
                        "baseline at each quant. Fix the conversion toolchain (`reap-lab doctor`), "
                        "provide `baseline_gguf`, or set `include_baseline: false` in the sweep YAML."
                    ) from e
                for m in baseline_manifests:
                    state.mark_done(
                        "convert", m.artifact_id, meta={"manifest": m.model_dump(mode="json")}
                    )
            for manifest in baseline_manifests:
                aid = manifest.artifact_id
                if resume and state.is_done("eval", aid):
                    summary = state.meta("eval", aid)["summary"]
                else:
                    state.mark_running("eval", aid)
                    try:
                        summary = evaluate_fn(
                            spec, workspace, manifest, eval_path, baseline_responses=None
                        )
                    except Exception as e:
                        state.mark_failed("eval", aid, f"{type(e).__name__}: {e}")
                        console.print(f"baseline eval failed for {aid}: {e} — continuing")
                        continue
                    state.mark_done("eval", aid, meta={"summary": summary})
                if manifest.quant:
                    baselines[manifest.quant.lower()] = summary
                console.print(f"baseline: evaluated {aid}")

        # -- retention grid ------------------------------------------------------
        candidates: list[tuple[ArtifactManifest, dict[str, Any]]] = []
        manifests_by_id: dict[str, ArtifactManifest] = {
            m.artifact_id: m for m in baseline_manifests
        }
        for retention in spec.retention:
            rkey = f"r{retention:g}"
            expected = [f"{rkey}-{q.lower()}" for q in spec.quants]
            try:
                if (
                    resume
                    and state.is_done("prune", rkey)
                    and all(state.is_done("convert", aid) for aid in expected)
                ):
                    manifests = [_manifest_from_state(state, aid) for aid in expected]
                    console.print(f"{rkey}: reusing pruned artifacts")
                else:
                    state.mark_running("prune", rkey)
                    manifests = build_artifacts_fn(spec, workspace, retention, cal_path)
                    state.mark_done("prune", rkey)
                    for m in manifests:
                        state.mark_done(
                            "convert", m.artifact_id,
                            meta={"manifest": m.model_dump(mode="json")},
                        )
                    console.print(f"{rkey}: built {len(manifests)} artifact(s)")
            except Exception as e:
                state.mark_failed("prune", rkey, f"{type(e).__name__}: {e}")
                console.print(f"{rkey}: FAILED ({e}) — continuing with the rest of the grid")
                continue

            for manifest in manifests:
                aid = manifest.artifact_id
                manifests_by_id[aid] = manifest
                if resume and state.is_done("eval", aid):
                    candidates.append((manifest, state.meta("eval", aid)["summary"]))
                    console.print(f"eval: reusing results for {aid}")
                    continue
                free_now = free_disk_gb(workspace.root)
                if free_now < spec.min_free_disk_gb:
                    state.mark_failed(
                        "eval", aid,
                        f"insufficient disk: {free_now:.1f} GB free < required "
                        f"{spec.min_free_disk_gb:g} GB — free up space and re-run to resume",
                    )
                    console.print(f"eval: skipping {aid} — insufficient disk")
                    continue
                base_summary = baselines.get((manifest.quant or "").lower()) or (
                    next(iter(baselines.values())) if baselines else None
                )
                state.mark_running("eval", aid)
                try:
                    summary = evaluate_fn(
                        spec, workspace, manifest, eval_path,
                        baseline_responses=(base_summary or {}).get("responses"),
                    )
                except Exception as e:
                    state.mark_failed("eval", aid, f"{type(e).__name__}: {e}")
                    console.print(f"eval failed for {aid}: {e} — continuing")
                    continue
                state.mark_done("eval", aid, meta={"summary": summary})
                candidates.append((manifest, summary))
                console.print(f"eval: scored {aid}")

        # -- scoring + gates -------------------------------------------------------
        rows: list[ArtifactRow] = []
        for quant, base_summary in baselines.items():
            base_ws = weighted_score(base_summary.get("domain_scores") or {}, pack)
            aid = base_summary.get("artifact_id", f"baseline-{quant}")
            rows.append(
                ArtifactRow(
                    artifact_id=aid,
                    weighted=base_ws,
                    retention_vs_baseline=1.0,
                    peak_vram_gb=_vram_gb(base_summary, spec.gates.min_context),
                    decode_tps=_perf_field(base_summary, spec.gates.min_context, "decode_tps"),
                    domain_scores=dict(base_summary.get("domain_scores") or {}),
                    false_refusal_rate=base_summary.get("false_refusal_rate"),
                    is_baseline=True,
                )
            )
            state.record_metric(aid, "weighted_score", base_ws)
            state.record_metric(aid, "gates", "baseline")

        winner_inputs: list[tuple[str, float, list[Any]]] = []
        for manifest, summary in candidates:
            aid = manifest.artifact_id
            base_summary = baselines.get((manifest.quant or "").lower()) or (
                next(iter(baselines.values())) if baselines else None
            )
            cand_ws = weighted_score(summary.get("domain_scores") or {}, pack)
            gate_results = evaluate_gates(
                spec.gates, summary, base_summary, summary.get("perf"), pack=pack
            )
            if base_summary is not None:
                base_ws = weighted_score(base_summary.get("domain_scores") or {}, pack)
                retention_ratio = quality_retention(cand_ws, base_ws)
                regressions = domain_regressions(
                    summary.get("domain_scores") or {},
                    base_summary.get("domain_scores") or {},
                )
            else:
                retention_ratio = None
                regressions = {}
            row = ArtifactRow(
                artifact_id=aid,
                weighted=cand_ws,
                retention_vs_baseline=retention_ratio,
                peak_vram_gb=_vram_gb(summary, spec.gates.min_context),
                decode_tps=_perf_field(summary, spec.gates.min_context, "decode_tps"),
                gates=gate_results,
                domain_scores=dict(summary.get("domain_scores") or {}),
                regressions=regressions,
                false_refusal_rate=summary.get("false_refusal_rate"),
                baseline_false_refusal_rate=(base_summary or {}).get("false_refusal_rate"),
            )
            rows.append(row)
            winner_inputs.append((aid, cand_ws, gate_results))

            state.record_metric(aid, "weighted_score", cand_ws)
            if retention_ratio is not None:
                state.record_metric(aid, "quality_retention", retention_ratio)
            if row.peak_vram_gb is not None:
                state.record_metric(aid, "peak_vram_gb", row.peak_vram_gb)
            if row.decode_tps is not None:
                state.record_metric(aid, "decode_tps", row.decode_tps)
            state.record_metric(aid, "gates", "PASS" if row.gates_pass else "FAIL")

        winner_id = select_winner(winner_inputs)

        # -- report ------------------------------------------------------------------
        failed_jobs = [j for j in state.jobs() if j["status"] == "failed"]
        md = render_report(spec, config_hash, pack, rows, winner_id, failed_jobs)
        report_path = write_report(workspace, config_hash, md)
        console.print(f"report: {report_path}")
        if winner_id:
            console.print(f"winner: {winner_id}")
        else:
            console.print("winner: none (no candidate passed all blocking gates)")

        # -- optional promotion ---------------------------------------------------------
        if promote:
            if winner_id is None:
                console.print("promotion skipped: no candidate passed all blocking gates")
            else:
                winner_row = next(r for r in rows if r.artifact_id == winner_id)
                result = promote_winner(
                    spec,
                    manifests_by_id[winner_id],
                    report_path,
                    workspace,
                    gates=winner_row.gates,
                    rationale=(
                        f"Highest weighted score ({winner_row.weighted:.4f}) among candidates "
                        "passing all blocking gates."
                    ),
                )
                console.print(result.message)

        return report_path
    finally:
        state.close()


def _vram_gb(summary: dict[str, Any], context: int) -> float | None:
    mb = _perf_field(summary, context, "peak_vram_mb")
    return mb / 1024.0 if mb is not None else None
