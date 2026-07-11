"""Shared fixtures for C3 evalharness tests. Fully offline: mock provider/runner,
respx for HTTP, fake Popen for subprocess — no network, no GPU, no real llama-server."""

from __future__ import annotations

from typing import Any

import pytest

from reaplab.core.config import JudgeCfg, ProviderCfg, RuntimeCfg, SweepSpec
from reaplab.core.records import ArtifactManifest, EvalRecord, TaskType
from reaplab.evalharness.runners import RunnerResponse


@pytest.fixture
def demo_tools() -> list[dict[str, Any]]:
    """Two simple OpenAI-format tool defs used across runner/evaluate tests."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "post_journal_entry",
                "parameters": {
                    "type": "object",
                    "properties": {"account": {"type": "string"}, "amount": {"type": "number"}},
                    "required": ["account", "amount"],
                },
            },
        },
    ]


@pytest.fixture
def make_record():
    def _make(
        id: str = "ev-001",
        domain: str = "general",
        task_type: TaskType = TaskType.EXACT,
        prompt: str = "What is the answer?",
        **kw: Any,
    ) -> EvalRecord:
        return EvalRecord(id=id, domain=domain, prompt=prompt, task_type=task_type, **kw)

    return _make


@pytest.fixture
def make_manifest():
    def _make(
        artifact_id: str = "baseline-q4_k_m",
        kind: str = "baseline",
        retention: float | None = None,
        quant: str | None = "Q4_K_M",
    ) -> ArtifactManifest:
        return ArtifactManifest(
            artifact_id=artifact_id,
            kind=kind,
            model_id="test/moe-model",
            retention=retention,
            quant=quant,
            path=f"artifacts/{artifact_id}.gguf",
            config_hash="cfg000000000",
        )

    return _make


@pytest.fixture
def make_spec(tmp_path):
    def _make(
        judge_responses: dict[str, str] | None = None,
        contexts: list[int] | None = None,
        votes: int = 3,
    ) -> SweepSpec:
        judge_provider = ProviderCfg(kind="mock", extra={"responses": judge_responses or {}})
        return SweepSpec(
            model_id="test/moe-model",
            domain_pack="domain-packs/test.yaml",
            generator=ProviderCfg(kind="mock"),
            judge=JudgeCfg(provider=judge_provider, votes=votes, version="j-test"),
            runtime=RuntimeCfg(kind="mock", contexts=contexts or [2048, 4096]),
            workspace=str(tmp_path / "ws"),
        )

    return _make


@pytest.fixture
def rresp():
    def _make(text: str = "", tool_calls: list[dict[str, Any]] | None = None, **kw: Any) -> RunnerResponse:
        return RunnerResponse(text=text, tool_calls=tool_calls, **kw)

    return _make
