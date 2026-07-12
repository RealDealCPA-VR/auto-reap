"""LocalOffloadProfile: platform guard, prerequisite errors, and the exact command
sequence (no real subprocesses)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reaplab.prune import profiles
from reaplab.prune.errors import PrerequisiteError, PruneError
from reaplab.prune.profiles import (
    ALLOW_LOCAL_OFFLOAD_ENV,
    LocalOffloadProfile,
    resolve_hf_model_dir,
)
from reaplab.prune.reap_cmd import calib_to_dataset_dir

from .helpers import MODEL_ID, make_spec

pytestmark = pytest.mark.usefixtures("allow_local_offload")


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


class TestWindowsGuard:
    def test_windows_refuses_with_alternatives(self, tmp_path: Path, monkeypatch):
        """reap's env pins vllm (no Windows wheels): say so before `uv sync` burns minutes."""
        monkeypatch.delenv(ALLOW_LOCAL_OFFLOAD_ENV, raising=False)
        monkeypatch.setattr(profiles.os, "name", "nt")
        monkeypatch.setattr(
            profiles.shutil,
            "which",
            lambda name: (_ for _ in ()).throw(AssertionError("no tool checks before the guard")),
        )
        spec = make_spec(tmp_path, profile="local-offload")
        with pytest.raises(PrerequisiteError) as exc:
            LocalOffloadProfile(tmp_path / "w").run_prune(spec, 0.5, tmp_path / "ds", tmp_path / "out")
        msg = str(exc.value)
        assert "vllm" in msg
        assert "remote" in msg and "mock" in msg  # both escape hatches named
        assert ALLOW_LOCAL_OFFLOAD_ENV in msg  # and the override

    def test_env_override_allows_it(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(ALLOW_LOCAL_OFFLOAD_ENV, "1")
        monkeypatch.setattr(profiles.os, "name", "nt")
        monkeypatch.setattr(profiles.shutil, "which", lambda name: None)
        spec = make_spec(tmp_path, profile="local-offload")
        # gets past the platform guard and fails on the NEXT check (git), not the OS
        with pytest.raises(PrerequisiteError, match="winget install Git.Git"):
            LocalOffloadProfile(tmp_path / "w").run_prune(spec, 0.5, tmp_path / "ds", tmp_path / "out")

    def test_resume_short_circuits_before_the_guard(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv(ALLOW_LOCAL_OFFLOAD_ENV, raising=False)
        monkeypatch.setattr(profiles.os, "name", "nt")
        spec = make_spec(tmp_path, profile="local-offload")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "config.json").write_text("{}", encoding="utf-8")
        assert LocalOffloadProfile(tmp_path / "w").run_prune(
            spec, 0.5, tmp_path / "ds", out_dir
        ) == out_dir


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

        def fake_run_logged(argv, *, cwd, log_path, track_peak_mem=False):
            calls.append(list(argv))
            if "prune.py" in " ".join(argv):
                ckpt = repo / "results" / "pruned_models" / "reap-42-0.25"
                ckpt.mkdir(parents=True)
                (ckpt / "config.json").write_text(
                    json.dumps({"num_experts": 96}), encoding="utf-8"
                )
                (ckpt / "model.safetensors").write_bytes(b"\x00" * 64)
            return None

        monkeypatch.setattr(profiles, "_run_logged", fake_run_logged)
        out = LocalOffloadProfile(work_dir=work).run_prune(spec, 0.75, dataset, tmp_path / "out")

        assert calls[0] == ["git", "clone", "--recursive", spec.prune.reap_repo, str(repo)]
        assert calls[1] == ["git", "checkout", spec.prune.reap_commit]
        assert calls[2] == ["git", "submodule", "update", "--init", "--recursive"]
        assert calls[3] == ["uv", "sync"]
        assert calls[4][:4] == ["uv", "run", "python", "src/reap/prune.py"]
        assert "--compression-ratio" in calls[4]
        assert calls[4][calls[4].index("--compression-ratio") + 1] == "0.25"
        # HF load_dataset() gets a POSIX path even on Windows (backslashes are mangled
        # by its glob/URI handling)
        dataset_arg = calls[4][calls[4].index("--dataset-name") + 1]
        assert dataset_arg == dataset.as_posix()
        assert "\\" not in dataset_arg
        # checkpoint was moved into the workspace out_dir
        assert out == tmp_path / "out"
        assert json.loads((out / "config.json").read_text(encoding="utf-8"))["num_experts"] == 96

    def test_peak_memory_is_sampled_during_the_prune(
        self, tmp_path: Path, monkeypatch, hf_cache_with_model, calibration_jsonl
    ):
        """FR-2.3 peak memory: measured for the profile that actually hosts the prune."""
        spec = make_spec(tmp_path, profile="local-offload")
        dataset = calib_to_dataset_dir(calibration_jsonl, tmp_path / "dataset")
        work = tmp_path / "prune-work"
        repo = work / "reap"
        monkeypatch.setattr(profiles.shutil, "which", lambda name: name)

        def fake_run_logged(argv, *, cwd, log_path, track_peak_mem=False):
            if "prune.py" in " ".join(argv):
                assert track_peak_mem, "the prune is the step whose memory matters"
                ckpt = repo / "pruned_models" / "reap-42-0.5"
                ckpt.mkdir(parents=True)
                (ckpt / "config.json").write_text("{}", encoding="utf-8")
                return 31.5
            return None

        monkeypatch.setattr(profiles, "_run_logged", fake_run_logged)
        profile = LocalOffloadProfile(work_dir=work)
        profile.run_prune(spec, 0.5, dataset, tmp_path / "out")
        assert profile.peak_mem_gb == 31.5

    def test_peak_memory_absent_is_explained_not_silent(self, tmp_path: Path):
        assert LocalOffloadProfile(tmp_path / "w").peak_mem_gb is None
        assert "not measured" in LocalOffloadProfile(tmp_path / "w").peak_mem_note
        assert "remote" in profiles.RemoteProfile(tmp_path / "w").peak_mem_note
        assert "mock" in profiles.MockProfile().peak_mem_note

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

        def fake_run_logged(argv, *, cwd, log_path, track_peak_mem=False):
            calls.append(list(argv))
            if "prune.py" in " ".join(argv):
                ckpt = repo / "pruned_models" / "reap-42-0.5"
                ckpt.mkdir(parents=True)
                (ckpt / "config.json").write_text("{}", encoding="utf-8")
            return None

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


class TestFindPrunedCheckpoint:
    """The fallback must not ship an arbitrary directory that merely holds a config.json."""

    def test_prefers_pruned_models_dir(self, tmp_path: Path):
        ckpt = tmp_path / "results" / "pruned_models" / "reap-42-0.5"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text("{}", encoding="utf-8")
        other = tmp_path / "configs"
        other.mkdir()
        (other / "config.json").write_text("{}", encoding="utf-8")
        assert profiles._find_pruned_checkpoint(tmp_path) == ckpt

    def test_fallback_accepts_a_real_checkpoint(self, tmp_path: Path):
        ckpt = tmp_path / "outputs" / "run1"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text("{}", encoding="utf-8")
        (ckpt / "model-00001-of-00002.safetensors").write_bytes(b"\x00")
        assert profiles._find_pruned_checkpoint(tmp_path) == ckpt

    def test_fallback_rejects_config_only_dirs(self, tmp_path: Path):
        """A hydra/tokenizer folder with a config.json is not a model: raise, don't ship it."""
        stray = tmp_path / "reap" / "conf"
        stray.mkdir(parents=True)
        (stray / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(PruneError) as exc:
            profiles._find_pruned_checkpoint(tmp_path)
        msg = str(exc.value)
        assert "safetensors" in msg
        assert str(stray) in msg  # names what it rejected, so the user can go look

    def test_nothing_found_is_instructive(self, tmp_path: Path):
        with pytest.raises(PruneError, match="pruned_models"):
            profiles._find_pruned_checkpoint(tmp_path)
