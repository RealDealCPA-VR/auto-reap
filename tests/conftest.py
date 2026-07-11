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
