"""RemoteProfile: script content, manual-step flow, ssh orchestration (all offline)."""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from reaplab.core.config import PruneCfg, RemoteCfg
from reaplab.prune import profiles
from reaplab.prune.errors import NeedsManualStep, PruneError
from reaplab.prune.profiles import RemoteProfile, budget_timeout_seconds, build_remote_script
from reaplab.prune.reap_cmd import calib_to_dataset_dir

from .helpers import MODEL_ID, make_spec

HOST = "user@gpu-box"


def _dataset_dir(calibration_jsonl: Path, tmp_path: Path) -> Path:
    return calib_to_dataset_dir(calibration_jsonl, tmp_path / "dataset")


def _make_pruned_tar(tar_path: Path, num_experts: int = 64) -> None:
    """Fabricate what the remote script would tar: pruned_models/<run>/config.json + weights."""
    buf = io.BytesIO()
    config = json.dumps({"model_type": "qwen3_moe", "num_experts": num_experts}).encode()

    def add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        add_bytes(tf, "pruned_models/reap-42-0.5/config.json", config)
        add_bytes(tf, "pruned_models/reap-42-0.5/model.safetensors", b"\x00" * 256)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tar_path.write_bytes(buf.getvalue())


class TestRemoteScript:
    def test_script_contains_all_required_steps(self, tmp_path: Path):
        spec = make_spec(tmp_path, profile="remote")
        script = build_remote_script(spec, 0.5)
        assert script.startswith("#!/usr/bin/env bash")
        assert spec.prune.reap_commit in script  # pinned commit checked out
        assert "git clone --recursive" in script
        assert "bash scripts/build.sh" in script  # reap's own uv build
        assert f"hf download {MODEL_ID}" in script  # model pre-download
        assert "HF_TOKEN" in script  # token passthrough (from the environment)
        assert "tar czf" in script  # output packaging
        assert "--compression-ratio 0.5" in script
        assert "$DATASET" in script  # dataset folder wired into the prune command

    def test_budget_timeout_math(self, tmp_path: Path):
        spec = make_spec(
            tmp_path,
            profile="remote",
            prune=PruneCfg(
                execution_profile="remote",
                remote=RemoteCfg(budget_usd=10.0, usd_per_hour=4.0),
            ),
        )
        assert budget_timeout_seconds(spec) == 9000
        assert "9000s" in build_remote_script(spec, 0.5)


class TestBudgetKillSwitch:
    """The box bills by wall clock: a hung clone / uv sync / 60 GB model download costs
    exactly as much as a hung prune, so the timeout must wrap the WHOLE script."""

    def test_watchdog_wraps_the_entire_script_not_just_the_prune(self, tmp_path: Path):
        spec = make_spec(tmp_path, profile="remote")  # 75 USD / 2.5 per h -> 108000s
        script = build_remote_script(spec, 0.5)
        timed = script.index("timeout -s TERM -k 60s 108000s bash")
        # every billable step comes AFTER the self re-exec under timeout
        for step in ("git clone --recursive", "bash scripts/build.sh", "hf download", "tar czf"):
            assert script.index(step) > timed, f"{step} runs outside the kill switch"
        assert script.count("timeout ") == 1  # the prune is no longer separately timed
        prune_line = next(ln for ln in script.splitlines() if "src/reap/prune.py" in ln)
        assert not prune_line.startswith("timeout")

    def test_kill_switch_message_is_loud(self, tmp_path: Path):
        script = build_remote_script(make_spec(tmp_path, profile="remote"), 0.5)
        assert "BUDGET KILL SWITCH" in script
        assert "$code -eq 124" in script  # GNU timeout's "killed" exit code is handled
        assert "DESTROY THE INSTANCE" in script

    def test_does_not_power_the_box_down(self, tmp_path: Path):
        """Shutting down from inside the guest keeps most rentals billing (and kills the
        shell you would fetch the tarball with). Teardown stays the user's call."""
        script = build_remote_script(make_spec(tmp_path, profile="remote"), 0.5)
        for forbidden in ("shutdown", "poweroff", "halt -p", "init 0"):
            assert forbidden not in script

    def test_no_accidental_shell_expansions(self, tmp_path: Path):
        """`$75` in a double-quoted echo expands as the positional param $7 — and under
        `set -u` an unbound `$2` aborts the whole run. Prices never sit next to a '$'."""
        script = build_remote_script(make_spec(tmp_path, profile="remote"), 0.5)
        for line in script.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#") or not stripped.startswith("echo"):
                continue  # comments do not expand; "$0"/"$@" in the re-exec are deliberate
            assert not re.search(r"\$\d", line), f"positional-param expansion in: {line}"

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
    def test_script_is_valid_bash(self, tmp_path: Path):
        script_path = tmp_path / "prune_remote_r0.5.sh"
        script_path.write_text(
            build_remote_script(make_spec(tmp_path, profile="remote"), 0.625),
            encoding="utf-8",
            newline="\n",
        )
        result = subprocess.run(
            [shutil.which("bash"), "-n", str(script_path)],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert result.returncode == 0, result.stderr

    def test_zero_rate_is_instructive(self, tmp_path: Path):
        spec = make_spec(
            tmp_path,
            profile="remote",
            prune=PruneCfg(
                execution_profile="remote",
                remote=RemoteCfg(budget_usd=10.0, usd_per_hour=0.0),
            ),
        )
        with pytest.raises(PruneError, match="usd_per_hour"):
            budget_timeout_seconds(spec)

    def test_ratio_is_clean_at_0625(self, tmp_path: Path):
        spec = make_spec(tmp_path, profile="remote")
        script = build_remote_script(spec, 0.625)
        assert "--compression-ratio 0.375" in script
        assert "0.37500000" not in script


class TestManualMode:
    def test_needs_manual_step_without_ssh_host(self, tmp_path: Path, calibration_jsonl: Path):
        spec = make_spec(tmp_path, profile="remote")
        assert spec.prune.remote.ssh_host is None
        dataset = _dataset_dir(calibration_jsonl, tmp_path)
        profile = RemoteProfile(work_dir=tmp_path / "prune-work")
        with pytest.raises(NeedsManualStep) as exc:
            profile.run_prune(spec, 0.5, dataset, tmp_path / "out")
        msg = str(exc.value)
        assert "prune_remote_r0.5.sh" in msg
        assert "ssh_host" in msg
        # script + numbered instructions were written
        script = tmp_path / "prune-work" / "prune_remote_r0.5.sh"
        steps = tmp_path / "prune-work" / "REMOTE_STEPS_r0.5.md"
        assert script.exists()
        assert steps.exists()
        text = steps.read_text(encoding="utf-8")
        for marker in ("1.", "2.", "3.", "4.", "5."):
            assert marker in text
        assert "scp" in text
        assert "pruned_r0.5.tar.gz" in text

    def test_pre_placed_tarball_resumes_without_ssh(self, tmp_path: Path, calibration_jsonl: Path):
        """User ran the manual steps and dropped the tarball -> extraction proceeds."""
        spec = make_spec(tmp_path, profile="remote")
        dataset = _dataset_dir(calibration_jsonl, tmp_path)
        work = tmp_path / "prune-work"
        _make_pruned_tar(work / "pruned_r0.5.tar.gz")
        out = RemoteProfile(work_dir=work).run_prune(spec, 0.5, dataset, tmp_path / "out")
        assert (out / "config.json").exists()
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert config["num_experts"] == 64

    def test_missing_dataset_is_instructive(self, tmp_path: Path):
        spec = make_spec(tmp_path, profile="remote")
        profile = RemoteProfile(work_dir=tmp_path / "w")
        with pytest.raises(PruneError, match="data.jsonl"):
            profile.run_prune(spec, 0.5, tmp_path / "no-dataset", tmp_path / "out")


class TestSshMode:
    def _spec(self, tmp_path: Path):
        return make_spec(
            tmp_path,
            profile="remote",
            prune=PruneCfg(
                execution_profile="remote",
                remote=RemoteCfg(budget_usd=75.0, usd_per_hour=2.5, ssh_host=HOST),
            ),
        )

    def test_ssh_orchestration_argv_sequence(
        self, tmp_path: Path, calibration_jsonl: Path, monkeypatch: pytest.MonkeyPatch
    ):
        spec = self._spec(tmp_path)
        dataset = _dataset_dir(calibration_jsonl, tmp_path)
        work = tmp_path / "prune-work"
        monkeypatch.delenv("HF_TOKEN", raising=False)
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            if argv[0] == "scp" and str(argv[1]).startswith(f"{HOST}:"):
                _make_pruned_tar(Path(argv[2]))  # the download step
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(profiles.subprocess, "run", fake_run)
        out = RemoteProfile(work_dir=work).run_prune(spec, 0.5, dataset, tmp_path / "out")

        assert [c[0] for c in calls] == ["ssh", "scp", "scp", "ssh", "scp"]
        # 1. prepare remote dir (and clear stale dataset)
        assert calls[0][1] == HOST
        assert "mkdir -p reap-work" in calls[0][2]
        assert "rm -rf reap-work/dataset" in calls[0][2]
        # 2. upload script
        assert calls[1] == [
            "scp",
            str(work / "prune_remote_r0.5.sh"),
            f"{HOST}:reap-work/prune_remote_r0.5.sh",
        ]
        # 3. upload dataset folder
        assert calls[2] == ["scp", "-r", str(dataset), f"{HOST}:reap-work/dataset"]
        # 4. execute the script remotely
        assert calls[3][1] == HOST
        assert calls[3][2] == "bash reap-work/prune_remote_r0.5.sh"
        # 5. download the tarball
        assert calls[4] == [
            "scp",
            f"{HOST}:reap-work/pruned_r0.5.tar.gz",
            str(work / "pruned_r0.5.tar.gz"),
        ]
        # extraction produced the checkpoint
        assert (out / "config.json").exists()

    def test_hf_token_is_piped_on_stdin_never_argv(self, tmp_path, calibration_jsonl, monkeypatch):
        """The token must not reach argv: it would land in this machine's process list,
        the on-disk log, the state DB error column, and the shareable report."""
        spec = self._spec(tmp_path)
        dataset = _dataset_dir(calibration_jsonl, tmp_path)
        monkeypatch.setenv("HF_TOKEN", "hf_secret123")
        calls: list[tuple[list[str], str | None]] = []

        def fake_run(argv, **kwargs):
            calls.append((list(argv), kwargs.get("input")))
            if argv[0] == "scp" and str(argv[1]).startswith(f"{HOST}:"):
                _make_pruned_tar(Path(argv[2]))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(profiles.subprocess, "run", fake_run)
        work = tmp_path / "w"
        RemoteProfile(work_dir=work).run_prune(spec, 0.5, dataset, tmp_path / "out")

        run_argv, run_input = [c for c in calls if c[0][0] == "ssh"][1]
        assert "hf_secret123" not in " ".join(run_argv)
        assert run_argv[2] == (
            "IFS= read -r HF_TOKEN; export HF_TOKEN; bash reap-work/prune_remote_r0.5.sh"
        )
        assert run_input == "hf_secret123\n"  # fed through the pipe instead
        # ...and no other step got the secret either
        assert not any("hf_secret123" in " ".join(argv) for argv, _ in calls)
        # the on-disk log never contains it
        log_text = (work / "remote-r0.5.log").read_text(encoding="utf-8")
        assert "hf_secret123" not in log_text

    def test_no_token_keeps_the_plain_run_command(self, tmp_path, calibration_jsonl, monkeypatch):
        spec = self._spec(tmp_path)
        dataset = _dataset_dir(calibration_jsonl, tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        calls: list[tuple[list[str], str | None]] = []

        def fake_run(argv, **kwargs):
            calls.append((list(argv), kwargs.get("input")))
            if argv[0] == "scp" and str(argv[1]).startswith(f"{HOST}:"):
                _make_pruned_tar(Path(argv[2]))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(profiles.subprocess, "run", fake_run)
        RemoteProfile(work_dir=tmp_path / "w").run_prune(spec, 0.5, dataset, tmp_path / "out")
        run_argv, run_input = [c for c in calls if c[0][0] == "ssh"][1]
        assert run_argv[2] == "bash reap-work/prune_remote_r0.5.sh"
        assert run_input is None

    def test_token_is_redacted_from_logs_and_errors(self, tmp_path, calibration_jsonl, monkeypatch):
        """Belt and braces: even if a token reaches a rendered command (an old script, a
        user-set SendEnv), nothing we log or raise may carry it."""
        spec = self._spec(tmp_path)
        dataset = _dataset_dir(calibration_jsonl, tmp_path)
        work = tmp_path / "w"

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="tried HF_TOKEN=hf_leaked999 and failed"
            )

        monkeypatch.setattr(profiles.subprocess, "run", fake_run)
        profile = RemoteProfile(work_dir=work)
        monkeypatch.setattr(
            profile,
            "_run_over_ssh",
            lambda spec, host, script_path, dataset_dir, tar_local, rtag: profile._ssh_step(
                ["ssh", HOST, "HF_TOKEN=hf_leaked999 bash x.sh"],
                what="run remote prune",
                host=HOST,
                timeout=None,
                log_path=work / "remote-r0.5.log",
            ),
        )
        with pytest.raises(PruneError) as exc:
            profile.run_prune(spec, 0.5, dataset, tmp_path / "out")
        msg = str(exc.value)
        assert "hf_leaked999" not in msg
        assert "HF_TOKEN=***" in msg
        assert "hf_leaked999" not in (work / "remote-r0.5.log").read_text(encoding="utf-8")

    def test_remote_failure_is_instructive(self, tmp_path, calibration_jsonl, monkeypatch):
        spec = self._spec(tmp_path)
        dataset = _dataset_dir(calibration_jsonl, tmp_path)

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 255, stdout="", stderr="Connection refused")

        monkeypatch.setattr(profiles.subprocess, "run", fake_run)
        with pytest.raises(PruneError) as exc:
            RemoteProfile(work_dir=tmp_path / "w").run_prune(spec, 0.5, dataset, tmp_path / "out")
        msg = str(exc.value)
        assert f"ssh {HOST}" in msg  # tells the user how to debug
        assert "Connection refused" in msg

    def test_missing_ssh_client_is_instructive(self, tmp_path, calibration_jsonl, monkeypatch):
        spec = self._spec(tmp_path)
        dataset = _dataset_dir(calibration_jsonl, tmp_path)

        def fake_run(argv, **kwargs):
            raise FileNotFoundError(argv[0])

        monkeypatch.setattr(profiles.subprocess, "run", fake_run)
        with pytest.raises(PruneError, match="OpenSSH"):
            RemoteProfile(work_dir=tmp_path / "w").run_prune(spec, 0.5, dataset, tmp_path / "out")

    def test_existing_checkpoint_short_circuits(self, tmp_path, calibration_jsonl, monkeypatch):
        """Resume: out_dir already extracted -> no subprocess at all."""
        spec = self._spec(tmp_path)
        dataset = _dataset_dir(calibration_jsonl, tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "config.json").write_text("{}", encoding="utf-8")

        def boom(argv, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("subprocess should not be invoked on resume")

        monkeypatch.setattr(profiles.subprocess, "run", boom)
        result = RemoteProfile(work_dir=tmp_path / "w").run_prune(spec, 0.5, dataset, out_dir)
        assert result == out_dir
