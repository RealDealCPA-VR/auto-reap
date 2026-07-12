"""build_baseline: mock fabrication, user-provided GGUF registration (and its
quant-coverage rule), expected_baseline_ids, prerequisites."""

from __future__ import annotations

from pathlib import Path

import pytest

from reaplab.core.hashing import artifact_hash
from reaplab.core.paths import Workspace
from reaplab.core.state import StateDB
from reaplab.prune.baseline import build_baseline, expected_baseline_ids
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

    def test_baseline_files_are_namespaced_by_config_hash(self, tmp_path, ws, state):
        """Contract C3: another spec (mock vs real, different model) cannot clobber them."""
        spec = make_spec(tmp_path, profile="mock")
        manifests = build_baseline(spec, ws, state)
        expected_dir = ws.artifacts / spec.config_hash() / "baseline"
        for m in manifests:
            assert Path(m.path).parent == expected_dir

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

    def test_bad_quant_fails_before_any_conversion(self, tmp_path, ws, state, monkeypatch):
        from reaplab.prune import gguf

        monkeypatch.setattr(
            gguf,
            "write_fake_gguf",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("converted despite a bad quant")),
        )
        spec = make_spec(tmp_path, profile="mock", quants=["Q4_K_M", "q4km"])
        with pytest.raises(PruneError, match="Q4_K_M"):
            build_baseline(spec, ws, state)


class TestExpectedBaselineIds:
    """Contract C4: the orchestrator must know which convert rows to expect. Guessing
    one-per-quant left phantom 'running' rows forever when baseline_gguf was set."""

    def test_one_id_per_quant_by_default(self, tmp_path):
        spec = make_spec(tmp_path, profile="mock")
        assert expected_baseline_ids(spec) == ["baseline-q4_k_m", "baseline-q5_k_m"]

    def test_empty_when_baseline_disabled(self, tmp_path):
        assert expected_baseline_ids(make_spec(tmp_path, include_baseline=False)) == []

    def test_user_baseline_yields_exactly_one_id(self, tmp_path):
        gguf_file = tmp_path / "Qwen3-30B-A3B-Q5_K_M.gguf"
        write_fake_gguf(gguf_file, seed="user-baseline")
        spec = make_spec(
            tmp_path, profile="mock", quants=["Q5_K_M"], baseline_gguf=str(gguf_file)
        )
        assert expected_baseline_ids(spec) == ["baseline-q5_k_m"]

    def test_matches_what_build_baseline_produces(self, tmp_path, ws, state):
        spec = make_spec(tmp_path, profile="mock")
        produced = [m.artifact_id for m in build_baseline(spec, ws, state)]
        assert produced == expected_baseline_ids(spec)

    def test_normalizes_lowercase_quants(self, tmp_path):
        spec = make_spec(tmp_path, profile="mock", quants=["q6_k"])
        assert expected_baseline_ids(spec) == ["baseline-q6_k"]


class TestUserProvidedBaseline:
    def test_registers_existing_gguf_and_detects_quant(self, tmp_path, ws, state):
        gguf_file = tmp_path / "Qwen3-30B-A3B-Q5_K_M.gguf"
        write_fake_gguf(gguf_file, seed="user-baseline")
        spec = make_spec(
            tmp_path, profile="mock", quants=["Q5_K_M"], baseline_gguf=str(gguf_file)
        )
        manifests = build_baseline(spec, ws, state)
        assert len(manifests) == 1
        m = manifests[0]
        assert m.artifact_id == "baseline-q5_k_m"
        assert m.kind == "baseline"
        assert m.quant == "Q5_K_M"
        assert m.path == str(gguf_file)
        assert m.artifact_hash == artifact_hash(gguf_file)
        assert m.versions["quant_detection"] == "filename"

    def test_undetectable_quant_falls_back_to_the_single_spec_quant(self, tmp_path, ws, state):
        gguf_file = tmp_path / "mystery.gguf"
        write_fake_gguf(gguf_file, seed="x")
        spec = make_spec(
            tmp_path, profile="mock", quants=["Q4_K_M"], baseline_gguf=str(gguf_file)
        )
        (m,) = build_baseline(spec, ws, state)
        assert m.artifact_id == "baseline-q4_k_m"  # spec.quants[0]
        assert "assumed" in m.versions["quant_detection"]

    def test_quant_not_covering_the_grid_fails_fast_with_three_options(self, tmp_path, ws, state):
        """One Q5_K_M file cannot be the baseline for a Q4_K_M candidate: the retention
        gate compares like with like, so a cross-quant baseline misreports quality."""
        gguf_file = tmp_path / "Qwen3-30B-A3B-Q5_K_M.gguf"
        write_fake_gguf(gguf_file, seed="user-baseline")
        spec = make_spec(
            tmp_path, profile="mock", quants=["Q4_K_M", "Q5_K_M"], baseline_gguf=str(gguf_file)
        )
        with pytest.raises(PruneError) as exc:
            build_baseline(spec, ws, state)
        msg = str(exc.value)
        assert "Q5_K_M" in msg and "Q4_K_M" in msg
        assert "quants: [Q5_K_M]" in msg  # option 1
        assert "remove baseline_gguf" in msg  # option 2
        assert "include_baseline: false" in msg  # option 3
        # nothing was half-registered
        assert state.status("convert", "baseline-q5_k_m") is None

    def test_assumed_quant_with_a_multi_quant_grid_also_fails(self, tmp_path, ws, state):
        gguf_file = tmp_path / "mystery.gguf"
        write_fake_gguf(gguf_file, seed="x")
        spec = make_spec(
            tmp_path, profile="mock", quants=["Q4_K_M", "Q5_K_M"], baseline_gguf=str(gguf_file)
        )
        with pytest.raises(PruneError, match="assumed"):
            build_baseline(spec, ws, state)

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
