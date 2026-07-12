"""The claude-cli provider is the DEFAULT generator and judge, so its subprocess
contract deserves coverage even though the binary itself is never invoked here."""

from __future__ import annotations

import json
import subprocess

import pytest

from reaplab.core.config import ProviderCfg
from reaplab.core.providers import ProviderError, get_provider
from reaplab.core.providers.claude_cli import ClaudeCliProvider


class FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def provider(monkeypatch) -> ClaudeCliProvider:
    monkeypatch.setattr("shutil.which", lambda name: r"C:\bin\claude.exe" if name == "claude" else None)
    return get_provider(ProviderCfg(kind="claude-cli", model="claude-sonnet-5"))


def _capture(monkeypatch, proc: FakeProc) -> dict:
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def test_missing_binary_is_instructive(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    p = get_provider(ProviderCfg(kind="claude-cli"))
    with pytest.raises(ProviderError, match="not found on PATH"):
        p.complete("hi")


def test_prompt_goes_via_stdin_not_argv(provider, monkeypatch):
    """Windows has a command-length limit; a 2k-prompt batch must never hit argv."""
    payload = json.dumps({"result": "ok", "usage": {"input_tokens": 12, "output_tokens": 3}})
    seen = _capture(monkeypatch, FakeProc(stdout=payload))

    long_prompt = "generate 20 items " * 500
    resp = provider.complete(long_prompt, system="be terse")

    assert seen["kwargs"]["input"] == long_prompt
    assert long_prompt not in " ".join(seen["cmd"])
    assert resp.text == "ok"
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 3


def test_command_shape(provider, monkeypatch):
    seen = _capture(monkeypatch, FakeProc(stdout=json.dumps({"result": "x"})))
    provider.complete("hi", system="SYS")
    cmd = seen["cmd"]
    assert cmd[0].endswith("claude.exe")
    assert "-p" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5"
    assert cmd[cmd.index("--append-system-prompt") + 1] == "SYS"


def test_json_mode_appends_instruction(provider, monkeypatch):
    seen = _capture(monkeypatch, FakeProc(stdout=json.dumps({"result": "[]"})))
    provider.complete("give me items", json_mode=True)
    assert "single valid JSON value only" in seen["kwargs"]["input"]


def test_nonzero_exit_raises_with_stderr(provider, monkeypatch):
    _capture(monkeypatch, FakeProc(stderr="not logged in", returncode=1))
    with pytest.raises(ProviderError, match="not logged in"):
        provider.complete("hi")


def test_plain_text_stdout_is_tolerated(provider, monkeypatch):
    """Older CLI builds print the answer directly instead of a JSON envelope."""
    _capture(monkeypatch, FakeProc(stdout="  just text  "))
    assert provider.complete("hi").text == "just text"


def test_timeout_is_reported(provider, monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 300)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(ProviderError, match="timed out"):
        provider.complete("hi")
