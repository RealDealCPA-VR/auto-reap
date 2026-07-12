"""build_artifacts: full mock flow, naming contract, manifests, resume semantics,
per-config isolation, manual-step state, provenance enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reaplab.core.hashing import artifact_hash
from reaplab.core.paths import Workspace
from reaplab.core.records import ArtifactManifest
from reaplab.core.state import StateDB
from reaplab.prune import profiles
from reaplab.prune.errors import NeedsManualStep, PruneError
from reaplab.prune.runner import artifacts_dir, build_artifacts, read_expert_stats

from .helpers import MODEL_ID, make_spec


@pytest.fixture
def built(tmp_path: Path, ws: Workspace, state: StateDB, calibration_jsonl: Path):
    spec = make_spec(tmp_path, profile="mock")
    manifests = build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
    return spec, manifests


class TestMockFlow:
    def test_one_manifest_per_quant(self, built):
        spec, manifests = built
        assert [m.artifact_id for m in manifests] == ["r0.5-q4_k_m", "r0.5-q5_k_m"]

    def test_manifest_fields(self, built):
        spec, manifests = built
        for m in manifests:
            assert m.kind == "gguf"
            assert m.model_id == MODEL_ID
            assert m.retention == 0.5
            assert m.quant in ("Q4_K_M", "Q5_K_M")
            assert m.config_hash == spec.config_hash()
            assert m.reap_commit == spec.prune.reap_commit
            assert m.wall_clock_s is not None
            assert m.versions["execution_profile"] == "mock"

    def test_artifact_hash_matches_file(self, built):
        _, manifests = built
        for m in manifests:
            path = Path(m.path)
            assert path.exists()
            assert m.artifact_hash == artifact_hash(path)

    def test_gguf_magic_bytes(self, built):
        _, manifests = built
        for m in manifests:
            assert Path(m.path).read_bytes()[:4] == b"GGUF"

    def test_state_stage_keys_follow_contract(self, built, state: StateDB):
        assert state.is_done("prune", "r0.5")
        assert state.is_done("convert", "r0.5-q4_k_m")
        assert state.is_done("convert", "r0.5-q5_k_m")

    def test_retention_0625_naming(self, tmp_path, ws, state, calibration_jsonl):
        spec = make_spec(tmp_path, profile="mock")
        manifests = build_artifacts(spec, 0.625, calibration_jsonl, ws, state)
        assert [m.artifact_id for m in manifests] == ["r0.625-q4_k_m", "r0.625-q5_k_m"]
        assert state.is_done("prune", "r0.625")

    def test_dataset_folder_lives_under_the_run_dir(self, built, ws: Workspace):
        """Contract C2: the converted REAP dataset is per-config, so a second spec's
        calibration set can never be the one REAP calibrates against."""
        spec, _ = built
        data = ws.run_dir(spec.config_hash()) / "data" / "calibration_dataset" / "data.jsonl"
        assert data.exists()
        assert not (ws.data / "calibration_dataset").exists()  # never the shared location

    def test_dataset_rebuilt_when_data_jsonl_is_missing(
        self, tmp_path, ws, state, calibration_jsonl
    ):
        spec = make_spec(tmp_path, profile="mock")
        build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        data = ws.data_dir(spec.config_hash()) / "calibration_dataset" / "data.jsonl"
        data.unlink()
        build_artifacts(spec, 0.75, calibration_jsonl, ws, state)
        assert data.exists()

    def test_artifacts_are_namespaced_by_config_hash(self, built, ws: Workspace):
        """Contract C3: the path itself proves provenance."""
        spec, manifests = built
        art = ws.artifacts / spec.config_hash()
        for m in manifests:
            assert Path(m.path).parent == art
        assert (art / "Qwen3-30B-A3B-r0.5-hf" / "config.json").exists()
        assert (art / "Qwen3-30B-A3B-r0.5-bf16.gguf").exists()

    def test_manifest_json_persisted(self, built, ws: Workspace):
        spec, manifests = built
        man_dir = ws.run_dir(spec.config_hash()) / "manifests"
        for m in manifests:
            path = man_dir / f"{m.artifact_id}.json"
            assert path.exists()
            loaded = ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
            assert loaded == m


class TestQuantValidation:
    def test_quant_name_validated(self, tmp_path, ws, state, calibration_jsonl):
        spec = make_spec(tmp_path, profile="mock", quants=["q4km"])
        with pytest.raises(PruneError, match="Q4_K_M"):
            build_artifacts(spec, 0.5, calibration_jsonl, ws, state)

    def test_bad_quant_fails_before_the_prune_runs(
        self, tmp_path, ws, state, calibration_jsonl, monkeypatch
    ):
        """A typo must cost a second, not a completed (paid) prune."""
        spec = make_spec(tmp_path, profile="mock", quants=["Q4_K_M", "q4km"])

        def boom(self, *a, **k):  # pragma: no cover - must never run
            raise AssertionError("prune executed despite an invalid quant name")

        monkeypatch.setattr(profiles.MockProfile, "run_prune", boom)
        with pytest.raises(PruneError, match="Q4_K_M"):
            build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        assert state.status("prune", "r0.5") is None  # nothing was even started

    def test_empty_quants_is_instructive_and_early(
        self, tmp_path, ws, state, calibration_jsonl, monkeypatch
    ):
        spec = make_spec(tmp_path, profile="mock", quants=[])

        def boom(self, *a, **k):  # pragma: no cover
            raise AssertionError("prune executed with an empty quant grid")

        monkeypatch.setattr(profiles.MockProfile, "run_prune", boom)
        with pytest.raises(PruneError, match="quants"):
            build_artifacts(spec, 0.5, calibration_jsonl, ws, state)


class TestResume:
    def test_second_run_skips_prune_and_convert(
        self, tmp_path, ws, state, calibration_jsonl, monkeypatch
    ):
        spec = make_spec(tmp_path, profile="mock")
        first = build_artifacts(spec, 0.5, calibration_jsonl, ws, state)

        def boom(self, *a, **k):  # pragma: no cover - must never run
            raise AssertionError("prune re-executed despite done state")

        monkeypatch.setattr(profiles.MockProfile, "run_prune", boom)
        from reaplab.prune import gguf

        monkeypatch.setattr(
            gguf,
            "write_fake_gguf",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("gguf re-written")),
        )
        second = build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        assert [m.artifact_id for m in second] == [m.artifact_id for m in first]
        assert [m.artifact_hash for m in second] == [m.artifact_hash for m in first]

    def test_done_stage_with_missing_file_reruns(
        self, tmp_path, ws, state, calibration_jsonl
    ):
        spec = make_spec(tmp_path, profile="mock")
        first = build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        Path(first[0].path).unlink()  # artifact vanished; state still says done
        second = build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        assert Path(second[0].path).exists()

    def test_manifest_from_another_config_is_never_reused(
        self, tmp_path, ws, state, calibration_jsonl
    ):
        """Stale (stage, key) rows from a previous config must not resurrect its manifest:
        the manifest's config_hash has to match, and the file has to sit in OUR namespace."""
        spec = make_spec(tmp_path, profile="mock")
        manifests = build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        man_path = ws.run_dir(spec.config_hash()) / "manifests" / "r0.5-q4_k_m.json"
        stale = json.loads(man_path.read_text(encoding="utf-8"))
        stale["config_hash"] = "deadbeefcafe"
        man_path.write_text(json.dumps(stale), encoding="utf-8")

        rebuilt = build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        assert rebuilt[0].config_hash == spec.config_hash()
        assert rebuilt[0].artifact_hash == manifests[0].artifact_hash  # same content, rebuilt

    def test_two_specs_never_share_artifact_files(self, tmp_path, ws, calibration_jsonl):
        spec_a = make_spec(tmp_path, profile="mock")
        spec_b = make_spec(tmp_path, profile="mock", seeds=[7])
        assert spec_a.config_hash() != spec_b.config_hash()
        with StateDB(tmp_path / "a.db") as sa, StateDB(tmp_path / "b.db") as sb:
            a = build_artifacts(spec_a, 0.5, calibration_jsonl, ws, sa)
            b = build_artifacts(spec_b, 0.5, calibration_jsonl, ws, sb)
        assert {m.path for m in a}.isdisjoint({m.path for m in b})
        assert artifacts_dir(ws, spec_a.config_hash()) != artifacts_dir(ws, spec_b.config_hash())

    def test_failed_prune_marks_state_and_reraises(
        self, tmp_path, ws, state, calibration_jsonl, monkeypatch
    ):
        spec = make_spec(tmp_path, profile="mock")

        def boom(self, *a, **k):
            raise PruneError("synthetic failure")

        monkeypatch.setattr(profiles.MockProfile, "run_prune", boom)
        with pytest.raises(PruneError, match="synthetic failure"):
            build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        assert state.status("prune", "r0.5") == "failed"
        jobs = state.jobs("prune")
        assert jobs[0]["error"] == "synthetic failure"

    def test_needs_manual_step_marks_manual_not_failed(
        self, tmp_path, ws, state, calibration_jsonl, monkeypatch
    ):
        """Contract C6: a remote prune awaiting the user is 'manual'. Marking it 'failed'
        made the report claim the sweep broke when nothing had."""
        spec = make_spec(tmp_path, profile="mock")

        def needs_step(self, *a, **k):
            raise NeedsManualStep("run prune_remote_r0.5.sh on a rented box, then re-run")

        monkeypatch.setattr(profiles.MockProfile, "run_prune", needs_step)
        with pytest.raises(NeedsManualStep):
            build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        assert state.status("prune", "r0.5") == "manual"
        job = state.jobs("prune")[0]
        assert "prune_remote_r0.5.sh" in job["error"]  # the instructions are kept

    def test_missing_calibration_is_instructive(self, tmp_path, ws, state):
        spec = make_spec(tmp_path, profile="mock")
        with pytest.raises(PruneError, match="Calibration file not found"):
            build_artifacts(spec, 0.5, tmp_path / "ghost.jsonl", ws, state)


class TestProvenance:
    """FR-2.3: a manifest with a null field must be null for a stated reason."""

    def test_expert_counts_recorded_after_a_real_prune(
        self, tmp_path, ws, state, calibration_jsonl, monkeypatch
    ):
        spec = make_spec(tmp_path, profile="remote")  # non-mock => real-prune bookkeeping

        class StubProfile(profiles.ExecutionProfile):
            name = "stub"
            peak_mem_gb = 27.25

            def run_prune(self, spec, retention, dataset_dir, out_dir):
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": "qwen3_moe",
                            "num_experts": 64,
                            "num_experts_per_tok": 8,
                        }
                    ),
                    encoding="utf-8",
                )
                (out_dir / "model.safetensors").write_bytes(b"\x00" * 32)
                return out_dir

        monkeypatch.setattr(profiles, "get_profile", lambda *a, **k: StubProfile())
        # the quant grid still runs through llama.cpp; stub that out
        from reaplab.prune import gguf as gguf_mod

        monkeypatch.setattr(
            gguf_mod.LlamaCppTools,
            "discover",
            classmethod(
                lambda cls, **k: gguf_mod.LlamaCppTools(
                    convert_script=tmp_path / "convert_hf_to_gguf.py",
                    quantize_bin=tmp_path / "llama-quantize.exe",
                )
            ),
        )
        monkeypatch.setattr(
            gguf_mod, "convert_to_gguf", lambda hf, out, tools, outtype="bf16": gguf_mod.write_fake_gguf(out, "bf16")
        )
        monkeypatch.setattr(
            gguf_mod, "quantize", lambda src, out, q, tools: gguf_mod.write_fake_gguf(out, q)
        )

        manifests = build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        for m in manifests:
            assert m.saliency_stats == {
                "num_experts_after": 64,
                "num_experts_source_field": "num_experts",
                "num_experts_per_tok": 8,
                "model_type": "qwen3_moe",
            }
            assert m.peak_mem_gb == 27.25

        # and the enrichment survives a resume (it is kept in the prune stage meta)
        again = build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
        assert again[0].saliency_stats == manifests[0].saliency_stats

    def test_unmeasurable_peak_memory_is_explained(self, built):
        _, manifests = built  # mock profile: nothing to sample
        for m in manifests:
            assert m.peak_mem_gb is None
            assert "not measured" in m.versions["peak_mem_gb"]
            assert m.saliency_stats is None  # mock fabricates; no real counts to claim


class TestReadExpertStats:
    def test_reads_alternate_field_names(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"n_routed_experts": 96, "model_type": "deepseek_v3"}), encoding="utf-8"
        )
        assert read_expert_stats(d) == {
            "num_experts_after": 96,
            "num_experts_source_field": "n_routed_experts",
            "model_type": "deepseek_v3",
        }

    def test_unknown_shape_returns_none_rather_than_guessing(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"hidden_size": 4096}), encoding="utf-8")
        assert read_expert_stats(d) is None

    def test_missing_or_corrupt_config_returns_none(self, tmp_path):
        assert read_expert_stats(tmp_path / "nope") is None
        d = tmp_path / "bad"
        d.mkdir()
        (d / "config.json").write_text("{not json", encoding="utf-8")
        assert read_expert_stats(d) is None
