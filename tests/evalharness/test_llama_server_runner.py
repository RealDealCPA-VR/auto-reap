from __future__ import annotations

import subprocess
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from reaplab.core.config import RuntimeCfg
from reaplab.evalharness.runners import LlamaServerRunner, RunnerError, _VramPoller


class FakePopen:
    """Never launches anything; records lifecycle calls."""

    def __init__(self, cmd, **kw):
        self.cmd = cmd
        self.kw = kw
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout or 0)
        return self.returncode


def _cfg(**kw) -> RuntimeCfg:
    defaults = dict(kind="llama-server", llama_server_path="C:/tools/llama/llama-server.exe", port=18080)
    defaults.update(kw)
    return RuntimeCfg(**defaults)


def _healthy_get(url, timeout=None):
    """Answers 200 immediately — i.e. SOMETHING is already on the port."""
    return SimpleNamespace(status_code=200)


def _get_healthy_after(n: int):
    """Dead until the nth probe: the port is free at start(), then the server comes up.

    start() probes the port before spawning (a live port means a foreign server), so a
    test that wants a successful launch must not answer 200 on the first probe.
    """
    calls = {"n": 0}

    def _get(url, timeout=None):
        calls["n"] += 1
        if calls["n"] <= n:
            raise httpx.ConnectError("nothing listening")
        return SimpleNamespace(status_code=200)

    return _get


def _no_nvidia_smi(*a, **kw):
    raise FileNotFoundError("nvidia-smi")


def test_build_command(make_manifest):
    runner = LlamaServerRunner(_cfg())
    cmd = runner.build_command(make_manifest(), 32768)
    assert cmd[0] == "C:/tools/llama/llama-server.exe"
    assert cmd[cmd.index("-m") + 1] == "artifacts/baseline-q4_k_m.gguf"
    assert cmd[cmd.index("-c") + 1] == "32768"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "18080"
    assert cmd[cmd.index("-ngl") + 1] == "999"  # gpu_layers=-1 -> all layers


def test_build_command_explicit_gpu_layers(make_manifest):
    runner = LlamaServerRunner(_cfg(gpu_layers=20))
    cmd = runner.build_command(make_manifest(), 4096)
    assert cmd[cmd.index("-ngl") + 1] == "20"


def test_missing_binary_is_instructive(make_manifest, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    runner = LlamaServerRunner(RuntimeCfg(kind="llama-server", llama_server_path=None))
    with pytest.raises(RunnerError) as exc:
        runner.build_command(make_manifest(), 4096)
    msg = str(exc.value)
    assert "llama-server not found" in msg and "llama_server_path" in msg


def test_start_polls_health_until_ready_then_stop_terminates(make_manifest):
    procs: list[FakePopen] = []

    def popen_factory(cmd, **kw):
        p = FakePopen(cmd, **kw)
        procs.append(p)
        return p

    attempts = {"n": 0}

    def flaky_get(url, timeout=None):
        attempts["n"] += 1
        if attempts["n"] <= 3:
            raise httpx.ConnectError("not up yet")
        return SimpleNamespace(status_code=200)

    runner = LlamaServerRunner(
        _cfg(), popen_factory=popen_factory, http_get=flaky_get,
        sleep=lambda s: None, vram_poll_cmd=_no_nvidia_smi,
    )
    runner.start(make_manifest(), 4096)
    assert len(procs) == 1
    assert procs[0].kw["stdout"] == subprocess.DEVNULL
    assert attempts["n"] >= 4  # needed several polls before healthy
    assert runner.load_time_s is not None and runner.load_time_s >= 0

    runner.stop()
    assert procs[0].terminated
    assert runner.peak_vram_mb is None  # no nvidia-smi -> None, never a crash


def test_start_raises_when_server_dies(make_manifest):
    def popen_factory(cmd, **kw):
        p = FakePopen(cmd, **kw)
        p.returncode = 1  # dead on arrival
        return p

    runner = LlamaServerRunner(
        _cfg(), popen_factory=popen_factory,
        http_get=lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("down")),
        sleep=lambda s: None, vram_poll_cmd=_no_nvidia_smi,
    )
    with pytest.raises(RunnerError) as exc:
        runner.start(make_manifest(), 4096)
    msg = str(exc.value)
    assert "exited with code 1" in msg and "VRAM" in msg  # tells the user what to check


def test_start_times_out_with_guidance(make_manifest, monkeypatch):
    monkeypatch.setattr(LlamaServerRunner, "HEALTH_TIMEOUT_S", 0.05)

    def never_ready(url, timeout=None):
        raise httpx.ConnectError("still loading")

    runner = LlamaServerRunner(
        _cfg(), popen_factory=FakePopen, http_get=never_ready,
        sleep=lambda s: None, vram_poll_cmd=_no_nvidia_smi,
    )
    with pytest.raises(RunnerError, match="did not become healthy"):
        runner.start(make_manifest(), 4096)


def test_stop_kills_if_terminate_hangs(make_manifest):
    class StubbornPopen(FakePopen):
        def terminate(self):
            self.terminated = True  # refuses to die: returncode stays None

    procs: list[StubbornPopen] = []

    def popen_factory(cmd, **kw):
        p = StubbornPopen(cmd, **kw)
        procs.append(p)
        return p

    runner = LlamaServerRunner(
        _cfg(), popen_factory=popen_factory, http_get=_get_healthy_after(2),
        sleep=lambda s: None, vram_poll_cmd=_no_nvidia_smi,
    )
    runner.start(make_manifest(), 4096)
    runner.stop()
    assert procs[0].terminated and procs[0].killed  # Windows-safe: terminate then kill


def test_start_refuses_to_adopt_a_foreign_server_on_the_port(make_manifest):
    """A server already answering on the port is NOT our artifact: adopting it would
    report someone else's model as this artifact's scores."""
    procs: list[FakePopen] = []

    def popen_factory(cmd, **kw):
        p = FakePopen(cmd, **kw)
        procs.append(p)
        return p

    runner = LlamaServerRunner(
        _cfg(), popen_factory=popen_factory, http_get=_healthy_get,
        sleep=lambda s: None, vram_poll_cmd=_no_nvidia_smi,
    )
    with pytest.raises(RunnerError) as exc:
        runner.start(make_manifest(), 4096)
    msg = str(exc.value)
    assert "port 18080 is already serving" in msg
    assert "runtime.port" in msg and "openai-compat" in msg  # both escape hatches named
    assert not procs  # nothing was launched


class FakePoller:
    """Stands in for the background nvidia-smi poller with a controllable peak."""

    def __init__(self, peak: float | None = None):
        self.peak = peak
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout=None) -> None:
        return None


def test_peak_vram_is_visible_while_the_server_runs(make_manifest):
    """FR-3.4 / the §5 VRAM blocker gate: perf.py snapshots peak_vram_mb BEFORE stop(),
    so the live poller peak must be readable mid-run — not only after teardown."""
    runner = LlamaServerRunner(
        _cfg(), popen_factory=FakePopen, http_get=_get_healthy_after(2),
        sleep=lambda s: None, vram_poll_cmd=_no_nvidia_smi,
    )
    runner.start(make_manifest(), 32768)
    assert runner.peak_vram_mb is None  # no samples yet

    poller = FakePoller(41000.0)
    runner._poller = poller  # the real poller found no nvidia-smi; drive a fake one
    # perf.py reads the runner with getattr while the server is still up:
    assert getattr(runner, "peak_vram_mb", None) == 41000.0
    poller.peak = 43250.0  # VRAM keeps climbing as the KV cache fills
    assert runner.peak_vram_mb == 43250.0

    runner.stop()
    assert poller.stopped
    assert runner.peak_vram_mb == 43250.0  # folded into the stored value at stop()


class PortSim:
    """Simulates the port honestly: it answers only while a process WE launched runs."""

    def __init__(self) -> None:
        self.up = False

    def get(self, url, timeout=None):
        if not self.up:
            raise httpx.ConnectError("connection refused")
        return SimpleNamespace(status_code=200)

    def popen(self, cmd, **kw):
        sim = self

        class SimPopen(FakePopen):
            def terminate(self):
                sim.up = False
                super().terminate()

            def kill(self):
                sim.up = False
                super().kill()

        self.up = True
        return SimPopen(cmd, **kw)


def test_peak_vram_resets_per_context(make_manifest):
    sim = PortSim()
    runner = LlamaServerRunner(
        _cfg(), popen_factory=sim.popen, http_get=sim.get,
        sleep=lambda s: None, vram_poll_cmd=_no_nvidia_smi,
    )
    runner.start(make_manifest(), 4096)
    runner._poller = FakePoller(30000.0)
    runner.stop()
    assert runner.peak_vram_mb == 30000.0

    runner.start(make_manifest(), 32768)  # new server, new context: peak starts over
    assert runner.peak_vram_mb is None
    runner._poller = FakePoller(38000.0)
    runner.stop()
    assert runner.peak_vram_mb == 38000.0


def test_complete_requires_start():
    runner = LlamaServerRunner(_cfg())
    with pytest.raises(RunnerError, match="not running"):
        runner.complete("hi", max_tokens=8)


def test_vram_poller_tracks_peak_across_gpus():
    sampled = threading.Event()
    values = iter(["8100\n12000\n", "9000\n15250\n", "100\n200\n"])

    def run_cmd(cmd, **kw):
        assert cmd[0] == "nvidia-smi"
        out = next(values, "100\n200\n")
        sampled.set()
        return SimpleNamespace(stdout=out)

    poller = _VramPoller(interval_s=0.005, run_cmd=run_cmd)
    poller.start()
    assert sampled.wait(timeout=5)
    deadline = time.monotonic() + 5
    while (poller.peak is None or poller.peak < 15250) and time.monotonic() < deadline:
        time.sleep(0.005)
    poller.stop()
    poller.join(timeout=5)
    assert poller.peak == 15250  # max across GPUs, max over time


def test_vram_poller_absent_nvidia_smi_is_none():
    poller = _VramPoller(interval_s=0.001, run_cmd=_no_nvidia_smi)
    poller.start()
    poller.join(timeout=5)  # thread exits on its own
    assert not poller.is_alive()
    assert poller.peak is None
