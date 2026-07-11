"""MockProfile: fabricated checkpoint shape, expert scaling, determinism."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reaplab.prune.profiles import MockProfile, get_profile

from .helpers import make_spec


class TestMockProfile:
    @pytest.mark.parametrize(
        ("retention", "experts"), [(0.5, 64), (0.625, 80), (0.75, 96), (1.0, 128)]
    )
    def test_num_experts_scaled_by_retention(self, tmp_path: Path, retention: float, experts: int):
        spec = make_spec(tmp_path)
        out = MockProfile().run_prune(spec, retention, tmp_path / "ds", tmp_path / f"ckpt-{retention}")
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert config["num_experts"] == experts
        assert "model_type" in config

    def test_checkpoint_has_all_stub_files(self, tmp_path: Path):
        spec = make_spec(tmp_path)
        out = MockProfile().run_prune(spec, 0.5, tmp_path / "ds", tmp_path / "ckpt")
        assert (out / "config.json").exists()
        assert (out / "model.safetensors").exists()
        assert (out / "tokenizer_config.json").exists()
        assert (out / "model.safetensors").stat().st_size >= 1024

    def test_deterministic_weights(self, tmp_path: Path):
        spec = make_spec(tmp_path)
        a = MockProfile().run_prune(spec, 0.5, tmp_path / "ds", tmp_path / "a")
        b = MockProfile().run_prune(spec, 0.5, tmp_path / "ds", tmp_path / "b")
        assert (a / "model.safetensors").read_bytes() == (b / "model.safetensors").read_bytes()

    def test_different_retention_different_weights(self, tmp_path: Path):
        spec = make_spec(tmp_path)
        a = MockProfile().run_prune(spec, 0.5, tmp_path / "ds", tmp_path / "a")
        b = MockProfile().run_prune(spec, 0.75, tmp_path / "ds", tmp_path / "b")
        assert (a / "model.safetensors").read_bytes() != (b / "model.safetensors").read_bytes()

    def test_config_loads_as_json_with_retention_metadata(self, tmp_path: Path):
        spec = make_spec(tmp_path)
        out = MockProfile().run_prune(spec, 0.625, tmp_path / "ds", tmp_path / "ckpt")
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert config["retention"] == 0.625
        assert config["reaplab_mock"] is True


class TestGetProfile:
    def test_factory_selects_by_spec(self, tmp_path: Path):
        from reaplab.prune.profiles import LocalOffloadProfile, RemoteProfile

        assert isinstance(
            get_profile(make_spec(tmp_path, profile="mock"), tmp_path), MockProfile
        )
        assert isinstance(
            get_profile(make_spec(tmp_path, profile="local-offload"), tmp_path),
            LocalOffloadProfile,
        )
        assert isinstance(
            get_profile(make_spec(tmp_path, profile="remote"), tmp_path), RemoteProfile
        )
