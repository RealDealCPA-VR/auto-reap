"""Test helpers for the C2 (prune) suite: spec construction, shared constants."""

from __future__ import annotations

from pathlib import Path

from reaplab.core.config import PruneCfg, RemoteCfg, SweepSpec

MODEL_ID = "Qwen/Qwen3-30B-A3B"


def make_spec(tmp_path: Path, *, profile: str = "mock", **overrides) -> SweepSpec:
    """A minimal, valid SweepSpec rooted in tmp_path. Never touches the network."""
    prune = overrides.pop(
        "prune",
        PruneCfg(
            execution_profile=profile,
            remote=RemoteCfg(budget_usd=75.0, usd_per_hour=2.5, ssh_host=None),
        ),
    )
    defaults: dict = dict(
        model_id=MODEL_ID,
        domain_pack="pack.yaml",  # path string only; nothing in C2 reads the pack
        retention=[0.75, 0.625, 0.5],
        quants=["Q4_K_M", "Q5_K_M"],
        seeds=[42],
        prune=prune,
        workspace=str(tmp_path / "ws"),
    )
    defaults.update(overrides)
    return SweepSpec(**defaults)
