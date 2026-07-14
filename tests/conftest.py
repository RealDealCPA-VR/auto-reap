from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def example_sweep_path() -> Path:
    return CONFIGS / "example-sweep.yaml"


@pytest.fixture
def cpa_pack_path() -> Path:
    return CONFIGS / "domain-packs" / "cpa-firm.yaml"


@pytest.fixture
def workspace(tmp_path: Path):
    from reaplab.core.paths import Workspace

    return Workspace(tmp_path / "ws").ensure()


@pytest.fixture
def mock_provider():
    from reaplab.core.config import ProviderCfg
    from reaplab.core.providers import get_provider

    return get_provider(ProviderCfg(kind="mock"))


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch):
    """Rich decides its width from COLUMNS / the terminal, so CLI output wraps
    differently on a dev box, in Docker, and on a CI runner — which silently breaks
    substring assertions on rendered output. Pin a wide terminal for every test so
    what we assert on is our text, not Rich's line-breaking."""
    monkeypatch.setenv("COLUMNS", "200")
