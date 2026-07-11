"""LocalOffloadProfile: prerequisite errors and the exact command sequence (no real subprocesses)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reaplab.prune import profiles
from reaplab.prune.errors import PrerequisiteError
from reaplab.prune.profiles import LocalOffloadProfile, resolve_hf_model_dir
from reaplab.prune.reap_cmd import calib_to_dataset_dir

from .helpers import MODEL_ID, make_spec


@pytest.fixture
def hf_cache_with_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache = tmp_path / "hf-cache" / "hub"
    snap = cache / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    monkeypatch.delenv("HF_HOME", raising=False)
    return snap


class TestResolveHfModelDir:
    def test_finds_snapshot(self, hf_cache_with_model: Path):
        assert resolve_hf_model_dir(MODEL_ID) == hf_cache_with_model

    def test_missing_model_says_hf_download(self, empty_hf_cache: Path):
        with pytest.raises(PrerequisiteError, match=rf"hf download {MODEL_ID}"):
            resolve_hf_model_dir(MODEL_ID)


class TestPrerequisites:
    def test_missing_git_is_instructive(self, tmp_path: Path, monkeypatch):
        spec = make_spec(tmp_path, profile="local-offload")
        monkeypatch.setattr(profiles.shutil, "which", lambda name: None)
        with pytest.raises(PrerequisiteError, match="winget install Git.Git"):
            LocalOffloadProfile(tmp_path / "w").run_prune(spec, 0.5, tmp_path / "ds", tmp_path / "out")

    def test_missing_uv_is_instructive(self, tmp_path: Path, monkeypatch):
        spec = make_spec(tmp_path, profile="local-offload")
        monkeypatch.setattr(
            profiles.shutil, "which", lambda name: "C:/git/git.exe" if name == "git" else None
        )
        with pytest.raises(PrerequisiteError, match="astral-sh.uv"):
            LocalOffloadProfile(tmp_path / "w").run_prune(spec, 0.5, tmp_path / "ds", tmp_path / "out")

    def test_missing_model_is_instructive(self, tmp_path: Path, monkeypatch, empty_hf_cache):
        spec = make_spec(tmp_path, profile="local-offload")
        monkeypatch.setattr(profiles.shutil, "which", lambda name: f"C:/tools/{name}.exe")
        with pytest.raises(PrerequisiteError, match="hf download"):
            LocalOffloadProfile(tmp_path / "w").run_prune(spec, 0.5, tmp_path / "ds", tmp_path / "out")


class TestCommandSequence:
    def test_clone_checkout_sync_prune_sequence(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        hf_cache_with_model: Path,
        calibration_jsonl: Path,
    ):
        spec = make_spec(tmp_path, profile="local-offload")
        dataset = calib_to_dataset_dir(calibration_jsonl, tmp_path / "dataset")
        work = tmp_path / "prune-work"
        repo = work / "reap"
        monkeypatch.setattr(
            profiles.shutil,
            "which",
            lambda name: {"git": "git", "uv": "uv"}.get(name),
        )
        calls: list[list[str]] = []

        def fake_run_logged(argv, *, cwd, log_path):
            calls.append(list(argv))
            if "prune.py" in " ".join(argv):
                ckpt = repo / "results" / "pruned_models" / "reap-42-0.25"
                ckpt.mkdir(parents=True)
                (ckpt / "config.json").write_text(
                    json.dumps({"num_experts": 96}), encoding="utf-8"
                )
                (ckpt / "model.safetensors").write_bytes(b"\x00" * 64)

        monkeypatch.setattr(profiles, "_run_logged", fake_run_logged)
        out = LocalOffloadProfile(work_dir=work).run_prune(spec, 0.75, dataset, tmp_path / "out")

        assert calls[0] == ["git", "clone", "--recursive", spec.prune.reap_repo, str(repo)]
        assert calls[1] == ["git", "checkout", spec.prune.reap_commit]
        assert calls[2] == ["git", "submodule", "update", "--init", "--recursive"]
        assert calls[3] == ["uv", "sync"]
        assert calls[4][:4] == ["uv", "run", "python", "src/reap/prune.py"]
        assert "--compression-ratio" in calls[4]
        assert calls[4][calls[4].index("--compression-ratio") + 1] == "0.25"
        assert calls[4][calls[4].index("--dataset-name") + 1] == str(dataset)
        # checkpoint was moved into the workspace out_dir
        assert out == tmp_path / "out"
        assert json.loads((out / "config.json").read_text(encoding="utf-8"))["num_experts"] == 96

    def test_existing_clone_skips_git_clone(
        self, tmp_path, monkeypatch, hf_cache_with_model, calibration_jsonl
    ):
        spec = make_spec(tmp_path, profile="local-offload")
        dataset = calib_to_dataset_dir(calibration_jsonl, tmp_path / "dataset")
        work = tmp_path / "prune-work"
        repo = work / "reap"
        (repo / ".git").mkdir(parents=True)
        monkeypatch.setattr(profiles.shutil, "which", lambda name: name)
        calls: list[list[str]] = []

        def fake_run_logged(argv, *, cwd, log_path):
            calls.append(list(argv))
            if "prune.py" in " ".join(argv):
                ckpt = repo / "pruned_models" / "reap-42-0.5"
                ckpt.mkdir(parents=True)
                (ckpt / "config.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(profiles, "_run_logged", fake_run_logged)
        LocalOffloadProfile(work_dir=work).run_prune(spec, 0.5, dataset, tmp_path / "out")
        assert "clone" not in [c[1] for c in calls if len(c) > 1]

    def test_resume_short_circuits(self, tmp_path, monkeypatch):
        """out_dir already holds a checkpoint -> no prereq checks, no subprocesses."""
        spec = make_spec(tmp_path, profile="local-offload")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            profiles.shutil,
            "which",
            lambda name: (_ for _ in ()).throw(AssertionError("should not check tools on resume")),
        )
        result = LocalOffloadProfile(tmp_path / "w").run_prune(spec, 0.5, tmp_path / "ds", out_dir)
        assert result == out_dir
