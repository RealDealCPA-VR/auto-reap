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
    evaluate_fn(spec, workspace, manifest, eval_path, *, baseline_responses=None,
                resume=True) -> summary dict (see reaplab.orchestrate.scoring);
        # baseline summaries additionally carry "responses": {item_id: text}
        # used for pairwise judging of candidates against the SAME quant.

StateDB stage ownership (shared contract — one writer per row):
    ("datagen", "datasets")    run_sweep
    ("prune",   "r<ret:g>")    the prune component (also marks 'manual' when a
                               remote step needs the user)
    ("convert", "<artifact>")  the prune component (done-meta: manifest PATH)
    ("eval",    "<artifact>")  run_sweep
    ("sweep",   "<key>")       run_sweep's OWN coarse failures (disk guard,
                               build errors) — never a component-owned row, so a
                               component's fine-grained progress is never
                               clobbered by a coarse failure above it.
Rendered as "stage:key" in reports, e.g. "prune:r0.5", "sweep:r0.5".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from reaplab.core.config import DomainPack, SweepSpec
from reaplab.core.paths import Workspace, free_disk_gb
from reaplab.core.records import ArtifactManifest
from reaplab.core.state import StateDB
from reaplab.orchestrate.promote import PromotionResult, promote_winner
from reaplab.orchestrate.report import ArtifactRow, render_report, write_report
from reaplab.orchestrate.scoring import (
    GateResult,
    domain_regressions,
    evaluate_gates,
    quality_retention,
    select_winner,
    weighted_score,
)

DatagenFn = Callable[..., tuple[Path, Path]]
BuildFn = Callable[..., list[ArtifactManifest]]
EvaluateFn = Callable[..., dict[str, Any]]


def _component_state(spec: SweepSpec, workspace: Workspace) -> StateDB:
    """Short-lived StateDB connection for component calls. Components record their own
    fine-grained stage progress; run_sweep's connection tracks the stages it owns. Both
    write small committed transactions, so the concurrent SQLite connections are safe."""
    return StateDB(workspace.state_db(spec.config_hash()))


def _default_datagen() -> DatagenFn:
    try:
        from reaplab.datagen import generate_datasets  # noqa: PLC0415 - lazy by design
    except ImportError as e:
        raise RuntimeError(
            "The datagen component (reaplab.datagen.generate_datasets) is unavailable: "
            f"{e}. Reinstall reap-lab (`uv sync`) or pass datagen_fn explicitly."
        ) from e

    def _adapter(spec: SweepSpec, workspace: Workspace) -> tuple[Path, Path]:
        # run_sweep owns the ("datagen", "datasets") stage record; the component
        # resolves its provider from spec.generator internally and writes the
        # datasets under workspace.data_dir(config_hash) (per-sweep, content-keyed).
        return generate_datasets(spec, workspace)

    return _adapter


def _default_build_baseline() -> BuildFn:
    try:
        from reaplab.prune import build_baseline  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The prune component (reaplab.prune.build_baseline) is unavailable: "
            f"{e}. Reinstall reap-lab (`uv sync`) or pass build_baseline_fn explicitly."
        ) from e

    def _adapter(spec: SweepSpec, workspace: Workspace) -> list[ArtifactManifest]:
        with _component_state(spec, workspace) as state:
            return build_baseline(spec, workspace, state)

    return _adapter


def _default_build_artifacts() -> BuildFn:
    try:
        from reaplab.prune import build_artifacts  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The prune component (reaplab.prune.build_artifacts) is unavailable: "
            f"{e}. Reinstall reap-lab (`uv sync`) or pass build_artifacts_fn explicitly."
        ) from e

    def _adapter(
        spec: SweepSpec, workspace: Workspace, retention: float, calibration_path: Path
    ) -> list[ArtifactManifest]:
        with _component_state(spec, workspace) as state:
            return build_artifacts(spec, retention, calibration_path, workspace, state)

    return _adapter


def _load_eval_records(eval_path: str) -> list:
    """Read+validate the eval set once per (path, mtime, size).

    The cache key includes the file's stamp, not just its name: a regenerated eval
    set (``sweep --no-resume``) lands at the same path with different content, and a
    name-only cache would serve the previous run's records to every later artifact.
    """
    from reaplab.core.jsonl import read_jsonl  # noqa: PLC0415
    from reaplab.core.records import EvalRecord  # noqa: PLC0415

    stat = Path(eval_path).stat()
    key = (eval_path, stat.st_mtime_ns, stat.st_size)
    cached = _load_eval_records._cache  # type: ignore[attr-defined]
    if key not in cached:
        cached.clear()  # one sweep = one eval set; don't grow across specs
        cached[key] = read_jsonl(eval_path, EvalRecord)
    return cached[key]


_load_eval_records._cache = {}  # type: ignore[attr-defined]


def _default_evaluate() -> EvaluateFn:
    try:
        from reaplab.evalharness import evaluate_artifact  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The eval harness (reaplab.evalharness.evaluate_artifact) is unavailable: "
            f"{e}. Reinstall reap-lab (`uv sync`) or pass evaluate_fn explicitly."
        ) from e

    def _adapter(
        spec: SweepSpec,
        workspace: Workspace,
        manifest: ArtifactManifest,
        eval_path: Path,
        *,
        baseline_responses: dict[str, str] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        records = _load_eval_records(str(eval_path))
        with _component_state(spec, workspace) as state:
            return evaluate_artifact(
                spec, manifest, records, workspace, state,
                baseline_responses=baseline_responses,
                resume=resume,
            )

    return _adapter


def _manual_step_error() -> type[BaseException]:
    """The prune component's NeedsManualStep, resolved lazily (this module must not
    import siblings at load time). Returns a private never-raised class when the
    component is unavailable, so the ``except`` clause is always well-formed."""
    try:
        from reaplab.prune import NeedsManualStep  # noqa: PLC0415
    except ImportError:

        class _NeverRaised(Exception):
            pass

        return _NeverRaised
    return NeedsManualStep


def _expected_baseline_ids(spec: SweepSpec) -> list[str]:
    """Artifact ids build_baseline will produce, from the prune component itself
    (it knows the single-quant ``baseline_gguf`` case, where a naive
    ``baseline-<q>`` per quant would leave phantom rows that never complete)."""
    try:
        from reaplab.prune import expected_baseline_ids  # noqa: PLC0415
    except ImportError:  # component absent (injected fakes in tests)
        return [f"baseline-{q.lower()}" for q in spec.quants]
    return list(expected_baseline_ids(spec))


def _validate_quants(spec: SweepSpec) -> None:
    """Fail on a quant typo BEFORE any expensive work (a remote prune can cost $75).

    Uses the prune component's validator so the CLI, the sweep, and the builders
    all agree on the confirmed llama.cpp quant set.
    """
    if not spec.quants:
        raise RuntimeError(
            "spec.quants is empty. Add at least one quantization to the sweep YAML "
            "(e.g. `quants: [Q4_K_M]`) — there is nothing to build or evaluate otherwise."
        )
    try:
        from reaplab.prune import validate_quant  # noqa: PLC0415
    except ImportError:  # component absent (injected fakes in tests)
        return
    for quant in spec.quants:
        validate_quant(quant)  # raises PruneError naming the closest valid quant


def _perf_field(summary: dict[str, Any], context: int, field_name: str) -> float | None:
    """Read one PerfMetrics field at the given context; falls back to the
    largest measured context so report rows stay informative on partial data."""
    raw = summary.get("perf") or {}
    perf: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        perf[str(key)] = dict(value)
    entry = perf.get(str(context))
    if entry is not None and entry.get(field_name) is not None:
        return float(entry[field_name])
    for key in sorted(perf, key=lambda s: int(s) if s.isdigit() else -1, reverse=True):
        value = perf[key].get(field_name)
        if value is not None:
            return float(value)
    return None


def _vram_gb(summary: dict[str, Any], context: int) -> float | None:
    mb = _perf_field(summary, context, "peak_vram_mb")
    return mb / 1024.0 if mb is not None else None


def _manifest_from_state(state: StateDB, artifact_id: str) -> ArtifactManifest:
    """Load an artifact manifest from the prune component's ("convert", id) done-meta.

    The component records the manifest as a PATH string; a full dict is tolerated
    for older state files.
    """
    manifest = state.meta("convert", artifact_id)["manifest"]
    if isinstance(manifest, str):
        import json  # noqa: PLC0415

        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    return ArtifactManifest.model_validate(manifest)


def _quant_of(artifact_id: str, manifest: ArtifactManifest | None) -> str:
    """Lowercase quant key for baseline pairing. Prefers the manifest; falls back to
    the shared id naming contract ``<prefix>-<quant lowercase>``."""
    if manifest is not None and manifest.quant:
        return manifest.quant.lower()
    return artifact_id.split("-", 1)[1].lower() if "-" in artifact_id else ""


def _baseline_note(quant: str, baselines: dict[str, dict[str, Any]]) -> str:
    """Explain a missing matching-quant baseline (the numbers it would have produced
    are reported as 'not measured', never substituted from another quant)."""
    available = ", ".join(sorted(q.upper() for q in baselines)) or "none"
    return (
        f"no baseline at quant {quant.upper() or '?'} — quality retention, domain regression "
        "and refusal-vs-baseline are NOT measured for this row, and its open-ended items were "
        f"scored without the pairwise judge. Baselines available: {available}. Build the "
        "matching baseline (`uv run reap-lab convert <your-sweep.yaml>`) for a gated comparison."
    )


def _score_rows(
    spec: SweepSpec,
    pack: DomainPack,
    baselines: dict[str, dict[str, Any]],
    candidates: list[tuple[ArtifactManifest | None, dict[str, Any]]],
) -> tuple[list[ArtifactRow], list[tuple[str, float, list[GateResult]]]]:
    """Build report rows + winner inputs from evaluated summaries (pure).

    A candidate is ONLY ever compared against the baseline of its own quant. When
    that baseline is missing, base_summary is None: the relative gates report "not
    measured" and pass, and the row carries a note saying so. Substituting an
    arbitrary other-quant baseline (the old fallback) silently mixed quantization
    error into quality retention and pairwise judging.
    """
    rows: list[ArtifactRow] = []
    for quant, base_summary in sorted(baselines.items()):
        base_ws = weighted_score(base_summary.get("domain_scores") or {}, pack)
        rows.append(
            ArtifactRow(
                artifact_id=base_summary.get("artifact_id", f"baseline-{quant}"),
                weighted=base_ws,
                retention_vs_baseline=1.0,
                peak_vram_gb=_vram_gb(base_summary, spec.gates.min_context),
                decode_tps=_perf_field(base_summary, spec.gates.min_context, "decode_tps"),
                domain_scores=dict(base_summary.get("domain_scores") or {}),
                false_refusal_rate=base_summary.get("false_refusal_rate"),
                is_baseline=True,
            )
        )

    winner_inputs: list[tuple[str, float, list[GateResult]]] = []
    for manifest, summary in candidates:
        aid = str(summary.get("artifact_id") or (manifest.artifact_id if manifest else "?"))
        quant = _quant_of(aid, manifest)
        base_summary = baselines.get(quant)
        cand_ws = weighted_score(summary.get("domain_scores") or {}, pack)
        gate_results = evaluate_gates(
            spec.gates, summary, base_summary, summary.get("perf"), pack=pack
        )
        notes: list[str] = []
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
            if baselines:  # baselines exist, just not at THIS quant
                notes.append(_baseline_note(quant, baselines))
        rows.append(
            ArtifactRow(
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
                notes=notes,
            )
        )
        winner_inputs.append((aid, cand_ws, gate_results))
    return rows, winner_inputs


@dataclass
class SweepSnapshot:
    """Everything a report/promotion needs, rebuilt from the StateDB alone."""

    rows: list[ArtifactRow] = field(default_factory=list)
    winner_id: str | None = None
    manifests: dict[str, ArtifactManifest] = field(default_factory=dict)
    candidate_ids: list[str] = field(default_factory=list)  # evaluated, non-baseline
    failed_jobs: list[dict[str, Any]] = field(default_factory=list)
    manual_steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def evaluated(self) -> bool:
        return bool(self.rows)


def load_sweep_snapshot(spec: SweepSpec, pack: DomainPack, state: StateDB) -> SweepSnapshot:
    """Rebuild rows/winner/manifests from completed StateDB stages — no new work.

    Reads the done ("eval", <artifact_id>) summaries and their ("convert", ...)
    manifests, re-scores with the same helpers run_sweep uses, and re-selects the
    winner. This is what makes `reap-lab report` and `reap-lab promote` true
    re-renders instead of "resume the whole sweep and hope nothing runs".
    """
    snapshot = SweepSnapshot()
    baselines: dict[str, dict[str, Any]] = {}
    candidates: list[tuple[ArtifactManifest | None, dict[str, Any]]] = []

    for job in state.jobs("eval"):
        if job["status"] != "done":
            continue
        aid = job["key"]
        summary = (job.get("meta") or {}).get("summary")
        if not isinstance(summary, dict):
            continue
        summary.setdefault("artifact_id", aid)
        manifest: ArtifactManifest | None = None
        if state.is_done("convert", aid):
            try:
                manifest = _manifest_from_state(state, aid)
            except (KeyError, OSError, ValueError):
                manifest = None  # manifest file moved/corrupt: score without it
        if manifest is not None:
            snapshot.manifests[aid] = manifest
        is_baseline = (
            manifest.kind == "baseline" if manifest is not None else aid.startswith("baseline-")
        )
        if is_baseline:
            baselines[_quant_of(aid, manifest)] = summary
        else:
            candidates.append((manifest, summary))
            snapshot.candidate_ids.append(aid)

    rows, winner_inputs = _score_rows(spec, pack, baselines, candidates)
    snapshot.rows = rows
    snapshot.winner_id = select_winner(winner_inputs)
    for job in state.jobs():
        if job["status"] == "failed":
            snapshot.failed_jobs.append(job)
        elif job["status"] == "manual":
            snapshot.manual_steps.append(
                {"stage": job["stage"], "key": job["key"], "instructions": job.get("error") or ""}
            )
    return snapshot


def _no_results_error(spec: SweepSpec, config_hash: str, workspace: Workspace) -> RuntimeError:
    return RuntimeError(
        f"Nothing has been evaluated yet for config {config_hash} (state: "
        f"{workspace.state_db(config_hash)}). Run `uv run reap-lab sweep <your-sweep.yaml>` "
        "first — it builds and scores the grid; `report` and `promote` only re-read what "
        "the sweep already finished."
    )


def render_report_from_state(spec: SweepSpec) -> Path:
    """Re-render the sweep report from completed stages ONLY; returns its path.

    Never generates datasets, never builds, never evaluates: the rows come from
    the StateDB's done eval summaries + manifests. Raises RuntimeError with
    guidance when the sweep has not evaluated anything yet.
    """
    config_hash = spec.config_hash()
    workspace = Workspace(spec.workspace)
    if not workspace.state_db(config_hash).exists():
        raise _no_results_error(spec, config_hash, workspace)
    pack = DomainPack.from_yaml(spec.domain_pack)
    with StateDB(workspace.state_db(config_hash)) as state:
        snapshot = load_sweep_snapshot(spec, pack, state)
        if not snapshot.evaluated:
            raise _no_results_error(spec, config_hash, workspace)
        md = render_report(
            spec, config_hash, pack, snapshot.rows, snapshot.winner_id,
            snapshot.failed_jobs, snapshot.manual_steps,
        )
    return write_report(workspace, config_hash, md)


def promote_from_state(spec: SweepSpec, artifact_id: str | None = None) -> PromotionResult:
    """Promote the stored winner (or ``artifact_id``) using ONLY completed stages.

    Re-renders the report first (so the decision page references a current one),
    then copies the artifact into LM Studio, writes the decision page, runs the
    smoke command, and archives exactly the EVALUATED non-winner candidates.
    Raises RuntimeError with guidance when there is nothing to promote.
    """
    config_hash = spec.config_hash()
    workspace = Workspace(spec.workspace)
    if not workspace.state_db(config_hash).exists():
        raise _no_results_error(spec, config_hash, workspace)
    pack = DomainPack.from_yaml(spec.domain_pack)
    with StateDB(workspace.state_db(config_hash)) as state:
        snapshot = load_sweep_snapshot(spec, pack, state)
        if not snapshot.evaluated:
            raise _no_results_error(spec, config_hash, workspace)

        target = artifact_id or snapshot.winner_id
        if target is None:
            raise RuntimeError(
                "No candidate passed all blocking gates, so there is no winner to promote. "
                "Review the gate columns in the report, then either loosen the gates, try a "
                "higher retention / gentler quant, or promote a specific artifact anyway with "
                "`uv run reap-lab promote <your-sweep.yaml> --artifact <artifact-id>`."
            )
        if target not in snapshot.manifests:
            known = ", ".join(sorted(snapshot.manifests)) or "none"
            raise RuntimeError(
                f"No evaluated artifact named '{target}' with a build manifest in this sweep "
                f"(config {config_hash}). Artifacts reap-lab built and can promote: {known}. "
                "(A GGUF scored with `reap-lab eval --gguf` has no build manifest — reap-lab "
                "will not move a file it did not build; copy it into LM Studio yourself.)"
            )

        md = render_report(
            spec, config_hash, pack, snapshot.rows, snapshot.winner_id,
            snapshot.failed_jobs, snapshot.manual_steps,
        )
        report_path = write_report(workspace, config_hash, md)
        row = next(r for r in snapshot.rows if r.artifact_id == target)
        if target == snapshot.winner_id:
            rationale = (
                f"Highest weighted score ({row.weighted:.4f}) among candidates passing all "
                "blocking gates."
            )
        else:
            verdict = "passes" if row.gates_pass else "FAILS"
            rationale = (
                f"Operator override: promoted with --artifact {target} (weighted "
                f"{row.weighted:.4f}; {verdict} the blocking gates). The gate-selected winner "
                f"was {snapshot.winner_id or 'none'}."
            )
        losers = [
            snapshot.manifests[aid]
            for aid in snapshot.candidate_ids
            if aid != target and aid in snapshot.manifests
        ]
        return promote_winner(
            spec,
            snapshot.manifests[target],
            report_path,
            workspace,
            gates=row.gates,
            rationale=rationale,
            losers=losers,
        )


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
    marked failed under the ("sweep", key) namespace and the sweep continues
    (PRD FR-4.2). Disk-guarded before every build (the 15-35 GB steps).

    Raises RuntimeError with instructive text for fatal conditions (no disk, no
    datasets, no baseline when one was requested, quant typo), and re-raises an
    aggregated NeedsManualStep when manual steps are pending AND nothing was
    evaluated — the report (with the instructions) is always written first.
    """
    console = Console(markup=False, highlight=False)
    config_hash = spec.config_hash()
    workspace = Workspace(spec.workspace).ensure(config_hash)
    manual_error = _manual_step_error()

    _validate_quants(spec)  # a typo must fail before a $75 remote prune, not after

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
            expected = _expected_baseline_ids(spec)
            if resume and expected and all(state.is_done("convert", aid) for aid in expected):
                baseline_manifests = [_manifest_from_state(state, aid) for aid in expected]
                console.print("baseline: reusing converted baseline artifacts")
            else:
                _guard_disk(spec, workspace, state, "baseline", fatal=True)
                try:
                    # the prune component owns (and writes) the ("convert", <id>) rows
                    baseline_manifests = build_baseline_fn(spec, workspace)
                except Exception as e:
                    state.mark_failed("sweep", "baseline", f"{type(e).__name__}: {e}")
                    raise RuntimeError(
                        f"Baseline build failed: {e}. Quality-retention gates need the unpruned "
                        "baseline at each quant. Fix the conversion toolchain (`reap-lab doctor`), "
                        "provide `baseline_gguf`, or set `include_baseline: false` in the sweep YAML."
                    ) from e
            for manifest in baseline_manifests:
                aid = manifest.artifact_id
                if resume and state.is_done("eval", aid):
                    summary = state.meta("eval", aid)["summary"]
                else:
                    state.mark_running("eval", aid)
                    try:
                        summary = evaluate_fn(
                            spec, workspace, manifest, eval_path,
                            baseline_responses=None, resume=resume,
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
        candidates: list[tuple[ArtifactManifest | None, dict[str, Any]]] = []
        manifests_by_id: dict[str, ArtifactManifest] = {
            m.artifact_id: m for m in baseline_manifests
        }
        manual_steps: list[dict[str, Any]] = []
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
                    if not _guard_disk(spec, workspace, state, rkey):
                        console.print(f"{rkey}: skipped — insufficient disk")
                        continue
                    # the prune component owns (and writes) ("prune", rkey) and
                    # ("convert", <id>) — run_sweep must not clobber those rows
                    manifests = build_artifacts_fn(spec, workspace, retention, cal_path)
                    console.print(f"{rkey}: built {len(manifests)} artifact(s)")
            except manual_error as e:
                # the prune component marked ("prune", rkey) as 'manual' before raising
                manual_steps.append({"stage": "prune", "key": rkey, "instructions": str(e)})
                console.print(
                    f"{rkey}: needs a MANUAL step (see the report) — continuing with the grid"
                )
                continue
            except Exception as e:
                state.mark_failed("sweep", rkey, f"{type(e).__name__}: {e}")
                console.print(f"{rkey}: FAILED ({e}) — continuing with the rest of the grid")
                continue

            for manifest in manifests:
                aid = manifest.artifact_id
                manifests_by_id[aid] = manifest
                if resume and state.is_done("eval", aid):
                    candidates.append((manifest, state.meta("eval", aid)["summary"]))
                    console.print(f"eval: reusing results for {aid}")
                    continue
                base_summary = baselines.get((manifest.quant or "").lower())
                state.mark_running("eval", aid)
                try:
                    summary = evaluate_fn(
                        spec, workspace, manifest, eval_path,
                        baseline_responses=(base_summary or {}).get("responses"),
                        resume=resume,
                    )
                except Exception as e:
                    state.mark_failed("eval", aid, f"{type(e).__name__}: {e}")
                    console.print(f"eval failed for {aid}: {e} — continuing")
                    continue
                state.mark_done("eval", aid, meta={"summary": summary})
                candidates.append((manifest, summary))
                console.print(f"eval: scored {aid}")

        # -- scoring + gates -------------------------------------------------------
        rows, winner_inputs = _score_rows(spec, pack, baselines, candidates)
        winner_id = select_winner(winner_inputs)
        _record_metrics(state, rows)

        # -- report ------------------------------------------------------------------
        failed_jobs = [j for j in state.jobs() if j["status"] == "failed"]
        md = render_report(spec, config_hash, pack, rows, winner_id, failed_jobs, manual_steps)
        report_path = write_report(workspace, config_hash, md)
        console.print(f"report: {report_path}")
        if winner_id:
            console.print(f"winner: {winner_id}")
        else:
            console.print("winner: none (no candidate passed all blocking gates)")

        # -- manual steps -------------------------------------------------------------
        if manual_steps and not candidates:
            # nothing was evaluated: the sweep genuinely cannot continue until the
            # user runs the prepared step(s). The CLI turns this into exit code 2.
            raise manual_error(_manual_summary(manual_steps, report_path))
        if manual_steps:
            console.print(
                f"{len(manual_steps)} stage(s) need a manual step — see 'Manual steps pending' "
                f"in {report_path}"
            )

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
                    losers=[
                        m
                        for m, _ in candidates
                        if m is not None and m.artifact_id != winner_id
                    ],
                )
                console.print(result.message)
                if not result.ok:
                    state.mark_failed("sweep", "promote", result.message)
                    raise RuntimeError(
                        f"Promotion failed at the {result.stage} step: {result.message} "
                        f"The sweep report is at {report_path}."
                    )

        return report_path
    finally:
        state.close()


def _guard_disk(
    spec: SweepSpec, workspace: Workspace, state: StateDB, key: str, fatal: bool = False
) -> bool:
    """Free-space check immediately before a build (PRD FR-4.2).

    The builds are the disk-hungry steps (a bf16 intermediate plus 15-35 GB per
    GGUF); evals consume ~nothing. Returns True when there is room. Otherwise
    records ("sweep", key) as failed and either raises (baseline: the sweep cannot
    proceed without it) or returns False so the grid continues with the retentions
    that still fit.
    """
    free = free_disk_gb(workspace.root)
    if free >= spec.min_free_disk_gb:
        return True
    message = (
        f"insufficient disk: {free:.1f} GB free on the volume holding {workspace.root} < "
        f"required min_free_disk_gb={spec.min_free_disk_gb:g}. A build writes a bf16 "
        "intermediate plus 15-35 GB per GGUF. Free up space (or archive earlier artifacts) "
        "and re-run — completed stages resume automatically."
    )
    state.mark_failed("sweep", key, message)
    if fatal:
        raise RuntimeError(f"Cannot build the baseline: {message}")
    return False


def _manual_summary(manual_steps: list[dict[str, Any]], report_path: Path) -> str:
    blocks = "\n\n".join(
        f"[{step['stage']}:{step['key']}]\n{step['instructions']}" for step in manual_steps
    )
    return (
        f"{len(manual_steps)} stage(s) need a manual step and nothing could be evaluated yet:\n\n"
        f"{blocks}\n\n"
        f"The same instructions are in the report: {report_path}\n"
        "Do the step(s), then re-run `uv run reap-lab sweep <your-sweep.yaml>` — everything "
        "already finished resumes automatically."
    )


def _record_metrics(state: StateDB, rows: list[ArtifactRow]) -> None:
    """Persist the scored rows for `reap-lab status` (report queries)."""
    for row in rows:
        state.record_metric(row.artifact_id, "weighted_score", row.weighted)
        if row.is_baseline:
            state.record_metric(row.artifact_id, "gates", "baseline")
            continue
        if row.retention_vs_baseline is not None:
            state.record_metric(row.artifact_id, "quality_retention", row.retention_vs_baseline)
        if row.peak_vram_gb is not None:
            state.record_metric(row.artifact_id, "peak_vram_gb", row.peak_vram_gb)
        if row.decode_tps is not None:
            state.record_metric(row.artifact_id, "decode_tps", row.decode_tps)
        state.record_metric(row.artifact_id, "gates", "PASS" if row.gates_pass else "FAIL")
