"""Shared fixtures for the C2 (prune) test suite. Fully offline and deterministic."""

from __future__ import annotations

from pathlib import Path

import pytest

from reaplab.core.jsonl import write_jsonl
from reaplab.core.paths import Workspace
from reaplab.core.records import CalibrationRecord
from reaplab.core.state import StateDB

from .helpers import make_spec


@pytest.fixture
def mock_spec(tmp_path: Path):
    return make_spec(tmp_path, profile="mock")


@pytest.fixture
def remote_spec(tmp_path: Path):
    return make_spec(tmp_path, profile="remote")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws").ensure()


@pytest.fixture
def state(tmp_path: Path):
    db = StateDB(tmp_path / "state.db")
    yield db
    db.close()


@pytest.fixture
def calibration_jsonl(tmp_path: Path) -> Path:
    """A small calibration.jsonl of CalibrationRecord lines."""
    records = [
        CalibrationRecord(
            id=f"cal-{i:06d}",
            domain="qbo_categorization",
            prompt=f"Categorize transaction #{i}: 'ACME SUPPLIES {i * 7}.50 USD'",
        )
        for i in range(5)
    ]
    path = tmp_path / "calibration.jsonl"
    write_jsonl(path, records)
    return path


@pytest.fixture
def no_llama_tools(monkeypatch: pytest.MonkeyPatch):
    """Guarantee llama.cpp tool discovery fails: no env vars, no PATH hits, no common dirs."""
    from reaplab.prune import gguf

    for var in ("CONVERT_HF_TO_GGUF", "LLAMA_QUANTIZE", "LLAMA_CPP_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gguf.shutil, "which", lambda name: None)
    monkeypatch.setattr(gguf, "_common_dirs", lambda: [])


@pytest.fixture
def allow_local_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    """local-offload refuses to run on Windows (reap pins vllm). Tests that exercise the
    profile's command sequence opt out of that guard explicitly."""
    from reaplab.prune.profiles import ALLOW_LOCAL_OFFLOAD_ENV

    monkeypatch.setenv(ALLOW_LOCAL_OFFLOAD_ENV, "1")


@pytest.fixture
def empty_hf_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the HF cache at an empty tmp dir so no local model resolves."""
    cache = tmp_path / "hf-cache" / "hub"
    cache.mkdir(parents=True)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    monkeypatch.delenv("HF_HOME", raising=False)
    return cache
