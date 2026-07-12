"""Shared fixtures for orchestrate tests. Fully offline and deterministic:
no network, no GPU, no external tools — the only subprocess spawned anywhere
in this directory is `python -c ...` for the promotion smoke-test hook."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from reaplab.core.config import DomainPack, DomainSpec, Gates, SweepSpec
from reaplab.core.paths import Workspace
from reaplab.core.records import ArtifactManifest, PerfMetrics, TaskType
from reaplab.core.state import StateDB

PACK_YAML = """\
name: test-pack
description: two-domain pack for orchestrator tests
domains:
  - name: alpha
    description: alpha domain
    weight: 3.0
  - name: beta
    description: beta domain
    task_type: tool_call
    weight: 1.0
"""


def make_perf(
    vram_mb: float | None = 30000.0,
    tps: float | None = 60.0,
    contexts: tuple[int, ...] = (4096, 32768),
) -> dict[str, dict[str, Any]]:
    """Perf mapping in the evaluate_artifact summary shape."""
    return {
        str(ctx): PerfMetrics(
            context=ctx,
            load_time_s=10.0,
            prefill_tps=500.0,
            decode_tps=tps,
            peak_vram_mb=vram_mb,
        ).model_dump()
        for ctx in contexts
    }


def make_summary(
    artifact_id: str = "r0.5-q4_k_m",
    alpha: float = 0.88,
    beta: float = 0.80,
    *,
    false_refusal: float | None = 0.01,
    should_refuse: float | None = 1.0,
    tool_validity: float | None = 0.99,
    vram_mb: float | None = 30000.0,
    tps: float | None = 60.0,
    perf: dict[str, dict[str, Any]] | None = None,
    responses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Deterministic evaluate_artifact summary matching the shared contract."""
    summary: dict[str, Any] = {
        "artifact_id": artifact_id,
        "domain_scores": {
            "alpha": alpha,
            "beta": beta,
            "benign_sensitive": 0.98,
            "should_refuse": 1.0,
        },
        "counts": {"alpha": 10, "beta": 10, "benign_sensitive": 5, "should_refuse": 5},
        "false_refusal_rate": false_refusal,
        "should_refuse_pass_rate": should_refuse,
        "tool_call_validity": tool_validity,
        "perf": perf if perf is not None else make_perf(vram_mb=vram_mb, tps=tps),
        "items_scored": 30,
    }
    if responses is not None:
        summary["responses"] = responses
    return summary


@pytest.fixture
def pack() -> DomainPack:
    return DomainPack(
        name="test-pack",
        domains=[
            DomainSpec(name="alpha", description="alpha domain", weight=3.0),
            DomainSpec(
                name="beta", description="beta domain", weight=1.0, task_type=TaskType.TOOL_CALL
            ),
        ],
    )


@pytest.fixture
def gates() -> Gates:
    return Gates()


@pytest.fixture
def summary_factory():
    return make_summary


@pytest.fixture
def perf_factory():
    return make_perf


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws").ensure()


class SweepHarness:
    """Injectable fake pipeline that honors the REAL component contracts.

    Like the live components it writes its own StateDB rows — ("prune", r<x>) and
    ("convert", <artifact_id>) with a manifest PATH in the done-meta, ("prune",
    r<x>) marked 'manual' before raising NeedsManualStep — because run_sweep no
    longer writes those rows itself. Counts calls, records the baseline_responses
    and resume flags it was handed, and can be told to fail or stall specific
    retentions.
    """

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.counts: Counter[str] = Counter()
        self.baseline_responses_seen: list[dict[str, str] | None] = []
        self.resume_seen: list[bool] = []
        self.fail_retentions: set[float] = set()
        self.manual_retentions: set[float] = set()
        #: quants the baseline builder produces; None = every quant in the spec
        #: (a user-supplied baseline_gguf covers ONE quant, so this can be narrower)
        self.baseline_quants: list[str] | None = None

        pack_path = tmp_path / "pack.yaml"
        pack_path.write_text(PACK_YAML, encoding="utf-8")
        self.spec = SweepSpec(
            model_id="Qwen/Qwen3-30B-A3B",
            domain_pack=str(pack_path),
            retention=[0.75, 0.5],
            quants=["Q4_K_M"],
            workspace=str(tmp_path / "ws"),
            min_free_disk_gb=0.0,
        )

    # -- component-side state bookkeeping ----------------------------------

    def _state(self, workspace: Workspace) -> StateDB:
        return StateDB(workspace.state_db(self.spec.config_hash()))

    def _record_convert(
        self, workspace: Workspace, manifest: ArtifactManifest, state: StateDB
    ) -> None:
        man_dir = workspace.run_dir(manifest.config_hash) / "manifests"
        man_dir.mkdir(parents=True, exist_ok=True)
        man_path = man_dir / f"{manifest.artifact_id}.json"
        man_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        state.mark_done(
            "convert",
            manifest.artifact_id,
            meta={"path": manifest.path, "manifest": str(man_path)},
        )

    # -- injected stage callables ------------------------------------------

    def datagen(self, spec: SweepSpec, workspace: Workspace) -> tuple[Path, Path]:
        self.counts["datagen"] += 1
        data_dir = workspace.data_dir(spec.config_hash())  # per-sweep datasets (C1)
        data_dir.mkdir(parents=True, exist_ok=True)
        cal = data_dir / "calibration_v1.jsonl"
        cal.write_text('{"id": "cal-1"}\n', encoding="utf-8")
        ev = data_dir / "eval_v1.jsonl"
        ev.write_text('{"id": "ev-1"}\n', encoding="utf-8")
        return cal, ev

    def _manifest(
        self, spec: SweepSpec, workspace: Workspace, artifact_id: str,
        retention: float | None, quant: str,
    ) -> ArtifactManifest:
        art_dir = workspace.artifacts / spec.config_hash()
        art_dir.mkdir(parents=True, exist_ok=True)
        path = art_dir / f"{artifact_id}.gguf"
        path.write_bytes(b"GGUF" + artifact_id.encode("utf-8") * 8)
        return ArtifactManifest(
            artifact_id=artifact_id,
            kind="baseline" if retention is None else "gguf",
            model_id=spec.model_id,
            retention=retention,
            quant=quant,
            path=str(path),
            config_hash=spec.config_hash(),
            artifact_hash=f"hash-{artifact_id}",
        )

    def build_baseline(self, spec: SweepSpec, workspace: Workspace) -> list[ArtifactManifest]:
        self.counts["build_baseline"] += 1
        manifests = [
            self._manifest(spec, workspace, f"baseline-{q.lower()}", None, q)
            for q in (self.baseline_quants or spec.quants)
        ]
        with self._state(workspace) as state:
            for m in manifests:
                self._record_convert(workspace, m, state)
        return manifests

    def build_artifacts(
        self, spec: SweepSpec, workspace: Workspace, retention: float, calibration_path: Path
    ) -> list[ArtifactManifest]:
        self.counts["build_artifacts"] += 1
        rkey = f"r{retention:g}"
        with self._state(workspace) as state:
            if retention in self.manual_retentions:
                from reaplab.prune import NeedsManualStep

                instructions = (
                    f"Run the generated prune script for {rkey} on your GPU box:\n"
                    f"  bash prune_remote_{rkey}.sh"
                )
                state.mark_manual("prune", rkey, instructions)
                raise NeedsManualStep(instructions)
            if retention in self.fail_retentions:
                state.mark_failed("prune", rkey, f"synthetic prune failure at {rkey}")
                raise RuntimeError(f"synthetic prune failure at {rkey}")
            manifests = [
                self._manifest(spec, workspace, f"{rkey}-{q.lower()}", retention, q)
                for q in spec.quants
            ]
            state.mark_done("prune", rkey)
            for m in manifests:
                self._record_convert(workspace, m, state)
        return manifests

    def evaluate(
        self,
        spec: SweepSpec,
        workspace: Workspace,
        manifest: ArtifactManifest,
        eval_path: Path,
        *,
        baseline_responses: dict[str, str] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        self.counts["evaluate"] += 1
        self.resume_seen.append(resume)
        if manifest.retention is None:  # baseline
            return make_summary(
                manifest.artifact_id,
                alpha=0.90,
                beta=0.80,
                false_refusal=0.02,
                responses={"ev-1": "baseline response"},
            )
        self.baseline_responses_seen.append(baseline_responses)
        drop = 0.02 if manifest.retention >= 0.75 else 0.04
        return make_summary(
            manifest.artifact_id,
            alpha=0.90 - drop,
            beta=0.80 - drop,
            false_refusal=0.01,
        )

    def run(self, **kwargs: Any) -> Path:
        from reaplab.orchestrate import run_sweep

        stages: dict[str, Any] = {
            "datagen_fn": self.datagen,
            "build_baseline_fn": self.build_baseline,
            "build_artifacts_fn": self.build_artifacts,
            "evaluate_fn": self.evaluate,
        }
        stages.update(kwargs)  # callers may override any stage (or pass resume/promote)
        return run_sweep(self.spec, **stages)


@pytest.fixture
def harness(tmp_path: Path) -> SweepHarness:
    return SweepHarness(tmp_path)
