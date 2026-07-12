"""PRD FR-3.5: the toolchain that produced an artifact must be pinned in its manifest.

Recording a tool PATH is not recording a version — these cover the real capture."""

from __future__ import annotations

import subprocess

import pytest

from reaplab.prune import runner


class FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 1):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture(autouse=True)
def _clear_cache():
    runner._llama_cpp_build.cache_clear()
    yield
    runner._llama_cpp_build.cache_clear()


def test_parses_the_llama_cpp_build_banner(monkeypatch):
    """llama-quantize prints its banner on stderr and exits nonzero when given no
    args — that nonzero exit is the normal path, not a failure."""
    banner = "version: 9966 (a1b2c3d)\nbuilt with MSVC 19.40 for x64\n"
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: FakeProc(stderr=banner, returncode=1))
    assert runner._llama_cpp_build(r"C:\llama\llama-quantize.exe") == "b9966 (a1b2c3d)"


def test_falls_back_to_a_generic_build_string(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: FakeProc(stdout="build: 1234-abc"))
    assert runner._llama_cpp_build("llama-quantize") == "1234-abc"


@pytest.mark.parametrize("boom", [OSError("not found"), subprocess.TimeoutExpired("x", 20)])
def test_unreadable_banner_never_fails_the_prune(monkeypatch, boom):
    def raise_it(cmd, **kw):
        raise boom

    monkeypatch.setattr(subprocess, "run", raise_it)
    assert runner._llama_cpp_build("missing-binary") == "unknown"


def test_manifest_versions_pin_the_toolchain(monkeypatch, tmp_path):
    from reaplab.core.config import ProviderCfg, PruneCfg, SweepSpec
    from reaplab.prune import gguf

    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: FakeProc(stderr="version: 9966 (deadbee)")
    )
    spec = SweepSpec(
        model_id="m/moe",
        domain_pack=str(tmp_path / "pack.yaml"),
        generator=ProviderCfg(kind="mock"),
        prune=PruneCfg(execution_profile="remote", reap_commit="1970473"),
    )
    tools = gguf.LlamaCppTools(
        convert_script=tmp_path / "convert_hf_to_gguf.py",
        quantize_bin=tmp_path / "llama-quantize.exe",
    )
    versions = runner._tool_versions(spec, tools)

    assert versions["llama_cpp_build"] == "b9966 (deadbee)"
    assert versions["reap_commit"] == "1970473"
    assert versions["reaplab"]  # our own version, so a manifest is traceable to a release
    assert versions["python"]
    assert "llama-quantize" in versions["llama_quantize"]


def test_mock_profile_says_so_rather_than_inventing_versions(tmp_path):
    from reaplab.core.config import ProviderCfg, PruneCfg, SweepSpec

    spec = SweepSpec(
        model_id="m/moe",
        domain_pack=str(tmp_path / "pack.yaml"),
        generator=ProviderCfg(kind="mock"),
        prune=PruneCfg(execution_profile="mock"),
    )
    versions = runner._tool_versions(spec, None)
    assert versions["gguf_tools"] == "mock"
    assert "llama_cpp_build" not in versions
