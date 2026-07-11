"""Shared fixtures for datagen tests. Fully offline and deterministic:
mock providers only, tmp_path workspaces, no network, no subprocesses."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from reaplab.core.config import DataCfg, DomainPack, ProviderCfg, SweepSpec
from reaplab.core.paths import Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS_DIR = REPO_ROOT / "configs" / "domain-packs"

# A compact pack covering every task type, sized so proportional math is exact:
# weights 3/2/2/2/1 (sum 10) with calibration_size=20 and eval_size=10.
MINI_PACK: dict[str, Any] = {
    "name": "mini-firm",
    "description": "Compact synthetic workload for offline datagen tests: "
    "classification, extraction, advisory, tooling, and long reviews.",
    "include_refusal_suites": True,
    "domains": [
        {
            "name": "txn_classify",
            "description": "Categorize transactions into a chart of accounts.",
            "task_type": "exact",
            "weight": 3.0,
            "prompt_guidance": "Realistic bank-feed lines with ambiguous vendors.",
            "tags": ["bookkeeping"],
        },
        {
            "name": "report_extract",
            "description": "Extract structured JSON from financial reports.",
            "task_type": "json_schema",
            "weight": 2.0,
            "json_schema": {
                "type": "object",
                "required": ["entity", "period", "lines"],
                "properties": {
                    "entity": {"type": "string"},
                    "period": {"type": "string"},
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["account", "amount"],
                            "properties": {
                                "account": {"type": "string"},
                                "amount": {"type": "number"},
                            },
                        },
                    },
                },
            },
        },
        {
            "name": "advisory",
            "description": "Answer advisory questions with clear recommendations.",
            "task_type": "open_ended",
            "weight": 2.0,
            "prompt_guidance": "Reward correct authority and flagged uncertainty.",
        },
        {
            "name": "ops_tools",
            "description": "Pick the right operations tool call.",
            "task_type": "tool_call",
            "weight": 2.0,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_statement",
                        "description": "Fetch a bank statement for a period.",
                        "parameters": {
                            "type": "object",
                            "required": ["account_id", "period"],
                            "properties": {
                                "account_id": {"type": "string"},
                                "period": {"type": "string"},
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "reconcile_account",
                        "description": "Run reconciliation for an account and period.",
                        "parameters": {
                            "type": "object",
                            "required": ["account_id", "period"],
                            "properties": {
                                "account_id": {"type": "string"},
                                "period": {"type": "string"},
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "post_journal_entry",
                        "description": "Post an adjusting journal entry.",
                        "parameters": {
                            "type": "object",
                            "required": ["memo"],
                            "properties": {"memo": {"type": "string"}},
                        },
                    },
                },
            ],
        },
        {
            "name": "long_review",
            "description": "Review long engagement files and summarize risks.",
            "task_type": "open_ended",
            "weight": 1.0,
            "long_context": True,
        },
    ],
}


@pytest.fixture
def packs_dir() -> Path:
    return PACKS_DIR


@pytest.fixture
def mini_pack_dict() -> dict[str, Any]:
    return copy.deepcopy(MINI_PACK)


@pytest.fixture
def make_pack_file(tmp_path: Path):
    """Write a pack dict to YAML under tmp_path and return the path."""

    def _make(pack: dict[str, Any] | None = None, filename: str = "pack.yaml") -> Path:
        data = copy.deepcopy(MINI_PACK) if pack is None else pack
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    return _make


@pytest.fixture
def mini_pack(make_pack_file) -> DomainPack:
    return DomainPack.from_yaml(make_pack_file())


@pytest.fixture
def make_spec(make_pack_file):
    """SweepSpec factory: mock generator, small sizes, mini pack by default."""

    def _make(
        pack: dict[str, Any] | None = None,
        *,
        generator: ProviderCfg | None = None,
        calibration_size: int = 20,
        eval_size: int = 10,
        **overrides: Any,
    ) -> SweepSpec:
        pack_path = make_pack_file(pack)
        data = overrides.pop(
            "data",
            DataCfg(calibration_size=calibration_size, eval_size=eval_size),
        )
        return SweepSpec(
            model_id="test/mini-moe",
            domain_pack=str(pack_path),
            generator=generator or ProviderCfg(kind="mock"),
            data=data,
            **overrides,
        )

    return _make


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws").ensure()
