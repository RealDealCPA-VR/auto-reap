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
    """Injectable fake pipeline: counts calls, records baseline_responses
    forwarding, and can be told to fail specific retentions."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.counts: Counter[str] = Counter()
        self.baseline_responses_seen: list[dict[str, str] | None] = []
        self.fail_retentions: set[float] = set()

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

    # -- injected stage callables ------------------------------------------

    def datagen(self, spec: SweepSpec, workspace: Workspace) -> tuple[Path, Path]:
        self.counts["datagen"] += 1
        cal = workspace.data / "calibration_v1.jsonl"
        cal.write_text('{"id": "cal-1"}\n', encoding="utf-8")
        ev = workspace.data / "eval_v1.jsonl"
        ev.write_text('{"id": "ev-1"}\n', encoding="utf-8")
        return cal, ev

    def _manifest(
        self, spec: SweepSpec, workspace: Workspace, artifact_id: str,
        retention: float | None, quant: str,
    ) -> ArtifactManifest:
        path = workspace.artifacts / f"{artifact_id}.gguf"
        path.write_bytes(b"GGUF" + artifact_id.encode("utf-8") * 8)
        return ArtifactManifest(
            artifact_id=artifact_id,
            kind="gguf",
            model_id=spec.model_id,
            retention=retention,
            quant=quant,
            path=str(path),
            config_hash=spec.config_hash(),
            artifact_hash=f"hash-{artifact_id}",
        )

    def build_baseline(self, spec: SweepSpec, workspace: Workspace) -> list[ArtifactManifest]:
        self.counts["build_baseline"] += 1
        return [
            self._manifest(spec, workspace, f"baseline-{q.lower()}", None, q)
            for q in spec.quants
        ]

    def build_artifacts(
        self, spec: SweepSpec, workspace: Workspace, retention: float, calibration_path: Path
    ) -> list[ArtifactManifest]:
        self.counts["build_artifacts"] += 1
        if retention in self.fail_retentions:
            raise RuntimeError(f"synthetic prune failure at r{retention:g}")
        return [
            self._manifest(
                spec, workspace, f"r{retention:g}-{q.lower()}", retention, q
            )
            for q in spec.quants
        ]

    def evaluate(
        self,
        spec: SweepSpec,
        workspace: Workspace,
        manifest: ArtifactManifest,
        eval_path: Path,
        *,
        baseline_responses: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.counts["evaluate"] += 1
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

        return run_sweep(
            self.spec,
            datagen_fn=self.datagen,
            build_baseline_fn=self.build_baseline,
            build_artifacts_fn=self.build_artifacts,
            evaluate_fn=self.evaluate,
            **kwargs,
        )


@pytest.fixture
def harness(tmp_path: Path) -> SweepHarness:
    return SweepHarness(tmp_path)
