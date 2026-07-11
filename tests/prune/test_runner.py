"""build_artifacts: full mock flow, naming contract, manifests, resume semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from reaplab.core.hashing import artifact_hash
from reaplab.core.paths import Workspace
from reaplab.core.records import ArtifactManifest
from reaplab.core.state import StateDB
from reaplab.prune import profiles
from reaplab.prune.errors import PruneError
from reaplab.prune.runner import build_artifacts

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

    def test_dataset_folder_created_once(self, built, ws: Workspace):
        data = ws.data / "calibration_dataset" / "data.jsonl"
        assert data.exists()

    def test_manifest_json_persisted(self, built, ws: Workspace):
        spec, manifests = built
        man_dir = ws.run_dir(spec.config_hash()) / "manifests"
        for m in manifests:
            path = man_dir / f"{m.artifact_id}.json"
            assert path.exists()
            loaded = ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
            assert loaded == m

    def test_quant_name_validated(self, tmp_path, ws, state, calibration_jsonl):
        spec = make_spec(tmp_path, profile="mock", quants=["q4km"])
        with pytest.raises(PruneError, match="Q4_K_M"):
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

    def test_missing_calibration_is_instructive(self, tmp_path, ws, state):
        spec = make_spec(tmp_path, profile="mock")
        with pytest.raises(PruneError, match="Calibration file not found"):
            build_artifacts(spec, 0.5, tmp_path / "ghost.jsonl", ws, state)

    def test_empty_quants_is_instructive(self, tmp_path, ws, state, calibration_jsonl):
        spec = make_spec(tmp_path, profile="mock", quants=[])
        with pytest.raises(PruneError, match="quants"):
            build_artifacts(spec, 0.5, calibration_jsonl, ws, state)
