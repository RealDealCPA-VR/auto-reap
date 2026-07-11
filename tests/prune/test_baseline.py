"""build_baseline: mock fabrication, user-provided GGUF registration, prerequisites."""

from __future__ import annotations

from pathlib import Path

import pytest

from reaplab.core.hashing import artifact_hash
from reaplab.core.paths import Workspace
from reaplab.core.state import StateDB
from reaplab.prune.baseline import build_baseline
from reaplab.prune.errors import PrerequisiteError, PruneError
from reaplab.prune.gguf import write_fake_gguf

from .helpers import MODEL_ID, make_spec


class TestMockBaseline:
    def test_fabricates_per_quant(self, tmp_path, ws: Workspace, state: StateDB):
        spec = make_spec(tmp_path, profile="mock")
        manifests = build_baseline(spec, ws, state)
        assert [m.artifact_id for m in manifests] == ["baseline-q4_k_m", "baseline-q5_k_m"]
        for m in manifests:
            assert m.kind == "baseline"
            assert m.retention is None
            assert m.quant in ("Q4_K_M", "Q5_K_M")
            assert Path(m.path).read_bytes()[:4] == b"GGUF"
            assert m.artifact_hash == artifact_hash(Path(m.path))
            assert m.config_hash == spec.config_hash()

    def test_state_keys_follow_contract(self, tmp_path, ws, state):
        spec = make_spec(tmp_path, profile="mock")
        build_baseline(spec, ws, state)
        assert state.is_done("convert", "baseline-q4_k_m")
        assert state.is_done("convert", "baseline-q5_k_m")

    def test_include_baseline_false_returns_empty(self, tmp_path, ws, state):
        spec = make_spec(tmp_path, profile="mock", include_baseline=False)
        assert build_baseline(spec, ws, state) == []

    def test_resume_reloads_manifests(self, tmp_path, ws, state, monkeypatch):
        spec = make_spec(tmp_path, profile="mock")
        first = build_baseline(spec, ws, state)
        from reaplab.prune import gguf

        monkeypatch.setattr(
            gguf,
            "write_fake_gguf",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-fabricated")),
        )
        second = build_baseline(spec, ws, state)
        assert [m.artifact_hash for m in second] == [m.artifact_hash for m in first]


class TestUserProvidedBaseline:
    def test_registers_existing_gguf_and_detects_quant(self, tmp_path, ws, state):
        gguf_file = tmp_path / "Qwen3-30B-A3B-Q5_K_M.gguf"
        write_fake_gguf(gguf_file, seed="user-baseline")
        spec = make_spec(tmp_path, profile="mock", baseline_gguf=str(gguf_file))
        manifests = build_baseline(spec, ws, state)
        assert len(manifests) == 1
        m = manifests[0]
        assert m.artifact_id == "baseline-q5_k_m"
        assert m.kind == "baseline"
        assert m.quant == "Q5_K_M"
        assert m.path == str(gguf_file)
        assert m.artifact_hash == artifact_hash(gguf_file)
        assert m.versions["quant_detection"] == "filename"

    def test_undetectable_quant_falls_back_to_first_spec_quant(self, tmp_path, ws, state):
        gguf_file = tmp_path / "mystery.gguf"
        write_fake_gguf(gguf_file, seed="x")
        spec = make_spec(tmp_path, profile="mock", baseline_gguf=str(gguf_file))
        (m,) = build_baseline(spec, ws, state)
        assert m.artifact_id == "baseline-q4_k_m"  # spec.quants[0]
        assert "assumed" in m.versions["quant_detection"]

    def test_missing_file_is_instructive(self, tmp_path, ws, state):
        spec = make_spec(tmp_path, profile="mock", baseline_gguf=str(tmp_path / "ghost.gguf"))
        with pytest.raises(PruneError, match="baseline_gguf"):
            build_baseline(spec, ws, state)


class TestRealBaselinePrerequisites:
    def test_model_not_in_cache_says_hf_download(
        self, tmp_path, ws, state, empty_hf_cache, monkeypatch
    ):
        """Non-mock, no baseline_gguf: needs the HF model locally, with llama tools present."""
        from reaplab.prune import gguf as gguf_mod

        conv = tmp_path / "convert_hf_to_gguf.py"
        quant = tmp_path / "llama-quantize.exe"
        conv.touch()
        quant.touch()
        monkeypatch.setenv("CONVERT_HF_TO_GGUF", str(conv))
        monkeypatch.setenv("LLAMA_QUANTIZE", str(quant))
        spec = make_spec(tmp_path, profile="remote")
        with pytest.raises(PrerequisiteError, match=rf"hf download {MODEL_ID}"):
            build_baseline(spec, ws, state)
        # the failure is recorded for the first artifact stage
        assert state.status("convert", "baseline-q4_k_m") == "failed"
        assert gguf_mod  # imported for parity; discovery must have succeeded before the raise

    def test_missing_llama_tools_is_instructive(self, tmp_path, ws, state, no_llama_tools):
        spec = make_spec(tmp_path, profile="remote")
        with pytest.raises(PruneError, match="github.com/ggml-org/llama.cpp/releases"):
            build_baseline(spec, ws, state)
