"""Top-level dataset generation pipeline (C1): plan -> generate -> filter -> emit.

``generate_datasets`` is the single entry point the orchestrator calls. It writes:

- ``workspace.data / "calibration_v1.jsonl"``      (CalibrationRecord lines, prompts only)
- ``workspace.data / "eval_v1.jsonl"``             (EvalRecord lines, held out + refusal suites)
- ``workspace.data / "dedup_report_v1.json"``      (what the near-dup/leakage filter dropped)
- ``workspace.data / "eval_v1_audit_sample.md"``   (stratified ~5% human-audit sample, PRD M1)

Resumability: with a StateDB, a completed "datagen" stage with both files present
short-circuits and returns the existing paths (PRD FR-4.1/G3).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reaplab.core.config import DomainPack, SweepSpec
from reaplab.core.jsonl import read_jsonl, write_jsonl
from reaplab.core.paths import Workspace
from reaplab.core.providers import LLMProvider, get_provider
from reaplab.core.records import CalibrationRecord, Difficulty, EvalRecord
from reaplab.core.state import StateDB
from reaplab.datagen.audit import write_audit_sample
from reaplab.datagen.dedup import filter_near_duplicates
from reaplab.datagen.planning import DomainAllocation, GenerationPlan, plan_counts
from reaplab.datagen.procedural import (
    generate_procedural_items,
    generate_refusal_suite,
    rng_for,
)
from reaplab.datagen.provider_gen import generate_domain_via_provider
from reaplab.datagen.synthdoc import with_long_document

log = logging.getLogger("reaplab.datagen")

CALIBRATION_FILENAME = "calibration_v1.jsonl"
EVAL_FILENAME = "eval_v1.jsonl"
DEDUP_REPORT_FILENAME = "dedup_report_v1.json"
AUDIT_SAMPLE_FILENAME = "eval_v1_audit_sample.md"

# StateDB coordinates for this stage (shared contract: stage key "datagen").
STAGE = "datagen"
STAGE_KEY = "datagen"


def _stage_done(state: StateDB) -> bool:
    """True when any 'datagen' job is done — tolerant of the key another
    component may have used when marking the stage."""
    return any(j["status"] == "done" for j in state.jobs(STAGE))


def _difficulty(value: Any) -> Difficulty:
    try:
        return Difficulty(value)
    except ValueError:
        return Difficulty.MEDIUM


def _apply_long_context(
    items: list[dict[str, Any]], alloc: DomainAllocation, seed: int, split: str
) -> None:
    """Wrap a planned share of a domain's items with a >=16k-token synthetic
    document (PRD FR-1.4). In-place; deterministic index choice per (domain, split)."""
    k = alloc.long_context_cal if split == "calibration" else alloc.long_context_eval
    k = min(k, len(items))
    if k <= 0:
        return
    pick = rng_for(seed, "lcpick", alloc.spec.name, split)
    for i in sorted(pick.sample(range(len(items)), k)):
        doc_rng = rng_for(seed, "doc", alloc.spec.name, split, i)
        items[i]["prompt"] = with_long_document(
            items[i]["prompt"], doc_rng, topic=alloc.spec.description
        )
        items[i]["tags"] = [*items[i].get("tags", []), "long_context"]


def _generate_split(
    plan: GenerationPlan,
    pack: DomainPack,
    provider: LLMProvider,
    seed: int,
    split: str,
    procedural: bool,
) -> list[tuple[DomainAllocation, dict[str, Any]]]:
    """Generate one split ("calibration" | "eval") across all planned allocations,
    in stable plan order. Returns (allocation, item-fields) pairs, long-context
    already applied."""
    out: list[tuple[DomainAllocation, dict[str, Any]]] = []
    for alloc in plan.allocations:
        n = alloc.cal_count if split == "calibration" else alloc.eval_count
        if n <= 0:
            continue
        if procedural:
            items = generate_procedural_items(alloc.spec, pack, seed, split, n)
        else:
            items = generate_domain_via_provider(alloc.spec, pack, provider, split, n)
            if alloc.suite and len(items) < n:
                # The G5 gates depend on these suites: top up from the canned lists
                # rather than shipping an undersized refusal suite.
                missing = n - len(items)
                log.warning(
                    "datagen: provider produced %d/%d %s items; topping up %d from "
                    "built-in templates",
                    len(items), n, alloc.spec.name, missing,
                )
                fill = generate_refusal_suite(alloc.spec.task_type, pack, seed, n)
                items.extend(fill[len(items) : n])
        _apply_long_context(items, alloc, seed, split)
        out.extend((alloc, item) for item in items)
    return out


def _to_calibration_records(
    pairs: list[tuple[DomainAllocation, dict[str, Any]]], source: str
) -> list[CalibrationRecord]:
    return [
        CalibrationRecord(
            id=f"cal-{i:06d}",
            domain=alloc.spec.name,
            prompt=item["prompt"],
            tags=item.get("tags", []),
            difficulty=_difficulty(item.get("difficulty", "medium")),
            source=source,
        )
        for i, (alloc, item) in enumerate(pairs, 1)
    ]


def _to_eval_records(
    pairs: list[tuple[DomainAllocation, dict[str, Any]]], source: str
) -> list[EvalRecord]:
    return [
        EvalRecord(
            id=f"ev-{i:06d}",
            domain=alloc.spec.name,
            prompt=item["prompt"],
            task_type=alloc.spec.task_type,
            gold=item.get("gold"),
            rubric=item.get("rubric"),
            json_schema=item.get("json_schema"),
            tools=item.get("tools"),
            expected_tool=item.get("expected_tool"),
            tags=item.get("tags", []),
            difficulty=_difficulty(item.get("difficulty", "medium")),
            source=source,
        )
        for i, (alloc, item) in enumerate(pairs, 1)
    ]


def _load_existing(path_str: str, model: type, what: str) -> list:
    src = Path(path_str)
    if not src.exists():
        raise FileNotFoundError(
            f"spec.{what} points to {src}, which does not exist. Fix the path in your "
            f"sweep YAML, or remove the `{what}:` key to generate the {what} set."
        )
    return read_jsonl(src, model)


def generate_datasets(
    spec: SweepSpec,
    workspace: Workspace,
    provider: LLMProvider | None = None,
    state: StateDB | None = None,
) -> tuple[Path, Path]:
    """Generate (or resume) the calibration and eval datasets for one sweep.

    Behavior:
    - provider defaults to ``get_provider(spec.generator)``; when its cfg.kind is
      "mock" the datasets are synthesized procedurally (deterministic, offline)
      instead of round-tripping through provider text.
    - ``spec.calibration`` / ``spec.eval`` may point at pre-existing JSONL files;
      those are validated, still passed through the leakage filter, and re-emitted
      into the workspace so downstream stages always read one canonical location.
    - eval always receives near-dup + leakage filtering (PRD FR-1.3), a dedup
      report JSON, and a stratified human-audit markdown sample (PRD M1).
    - with a StateDB: stage "datagen" is marked running/done/failed; a done stage
      with both output files present returns immediately (resume).

    Returns (calibration_path, eval_path) under ``workspace.data``.
    """
    workspace.ensure()
    cal_path = workspace.data / CALIBRATION_FILENAME
    eval_path = workspace.data / EVAL_FILENAME

    if state is not None and cal_path.exists() and eval_path.exists() and _stage_done(state):
        log.info("datagen already complete; reusing %s and %s", cal_path, eval_path)
        return cal_path, eval_path

    if provider is None:
        provider = get_provider(spec.generator)

    pack_path = Path(spec.domain_pack)
    if not pack_path.exists():
        raise FileNotFoundError(
            f"domain pack not found at {pack_path}. Set `domain_pack:` in your sweep YAML "
            "to a pack file (examples live in configs/domain-packs/)."
        )
    pack = DomainPack.from_yaml(pack_path)
    seed = spec.seeds[0] if spec.seeds else 42
    plan = plan_counts(pack, spec.data, seed=seed)
    procedural = provider.cfg.kind == "mock"
    source = f"synthetic-{provider.cfg.kind}"

    if state is not None:
        state.mark_running(STAGE, STAGE_KEY)
    try:
        # --- calibration -----------------------------------------------------
        if spec.calibration:
            cal_records: list[CalibrationRecord] = _load_existing(
                spec.calibration, CalibrationRecord, "calibration"
            )
            log.info("using pre-existing calibration set: %s (%d records)",
                     spec.calibration, len(cal_records))
        else:
            cal_pairs = _generate_split(plan, pack, provider, seed, "calibration", procedural)
            cal_records = _to_calibration_records(cal_pairs, source)

        # --- eval -------------------------------------------------------------
        if spec.eval:
            eval_records: list[EvalRecord] = _load_existing(spec.eval, EvalRecord, "eval")
            log.info("using pre-existing eval set: %s (%d records)", spec.eval, len(eval_records))
        else:
            eval_pairs = _generate_split(plan, pack, provider, seed, "eval", procedural)
            eval_records = _to_eval_records(eval_pairs, source)

        # --- near-dup + leakage filter (PRD FR-1.3) ----------------------------
        if spec.data.dedup_backend == "embedding":
            embed_provider = (
                get_provider(spec.data.embedding_provider)
                if spec.data.embedding_provider
                else provider
            )
        else:
            embed_provider = None
        kept_eval, report = filter_near_duplicates(
            eval_records,
            cal_records,
            threshold=spec.data.near_dup_threshold,
            backend=spec.data.dedup_backend,
            provider=embed_provider,
        )
        if report.dropped:
            log.info(
                "dedup filter dropped %d of %d eval items (%s backend, threshold %.2f)",
                len(report.dropped), report.eval_in, report.backend, report.threshold,
            )
        report_path = workspace.data / DEDUP_REPORT_FILENAME
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8", newline="\n")

        # --- emit ---------------------------------------------------------------
        n_cal = write_jsonl(cal_path, cal_records)
        n_eval = write_jsonl(eval_path, kept_eval)
        audit_path = write_audit_sample(
            kept_eval, workspace.data / AUDIT_SAMPLE_FILENAME, seed=seed
        )
        log.info(
            "datasets written: %s (%d), %s (%d); audit sample: %s",
            cal_path, n_cal, eval_path, n_eval, audit_path,
        )

        if state is not None:
            state.mark_done(
                STAGE,
                STAGE_KEY,
                meta={
                    "calibration": str(cal_path),
                    "eval": str(eval_path),
                    "calibration_count": n_cal,
                    "eval_count": n_eval,
                    "dedup_dropped": len(report.dropped),
                    "seed": seed,
                    "pack": pack.name,
                    "source": source,
                },
            )
    except Exception as e:
        if state is not None:
            state.mark_failed(STAGE, STAGE_KEY, f"{type(e).__name__}: {e}")
        raise
    return cal_path, eval_path
