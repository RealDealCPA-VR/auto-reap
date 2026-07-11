"""`reap-lab doctor` — environment checks with fix-it guidance for each finding.

Levels: OK (ready), WARN (works for some paths, e.g. mock/demo, but a real sweep
will hit this), FAIL (spec explicitly requires something that is missing).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from reaplab import __version__
from reaplab.core.config import DomainPack, ProviderCfg, SweepSpec
from reaplab.core.paths import free_disk_gb

Check = tuple[str, str, str]  # (name, level, message)

_LLAMA_RELEASES = "https://github.com/ggml-org/llama.cpp/releases (win-cuda zip + cudart zip)"


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _check_python() -> Check:
    v = sys.version_info
    return ("python / reap-lab", "OK", f"python {v.major}.{v.minor}.{v.micro}, reap-lab {__version__}")


def _check_claude_cli() -> Check:
    path = shutil.which("claude")
    if not path:
        return (
            "claude CLI",
            "WARN",
            "not on PATH - the default generator/judge provider needs it. Install Claude Code "
            "(https://claude.com/claude-code) or switch providers to openai-compat/anthropic-api.",
        )
    version = _run([path, "--version"]) or "version unknown"
    return ("claude CLI", "OK", f"{version.splitlines()[0]} ({path})")


def _check_llamacpp(spec: SweepSpec | None) -> Check:
    from reaplab.prune import LlamaCppTools, ToolNotFoundError  # noqa: PLC0415

    convert = spec.prune.convert_script if spec else None
    quantize = spec.prune.llama_quantize if spec else None
    try:
        tools = LlamaCppTools.discover(convert_script=convert, quantize_bin=quantize)
    except ToolNotFoundError as e:
        level = "FAIL" if spec and spec.prune.execution_profile != "mock" else "WARN"
        return ("llama.cpp", level, f"{e} Get binaries: {_LLAMA_RELEASES}")
    return ("llama.cpp", "OK", f"convert={tools.convert_script} quantize={tools.quantize_bin}")


def _check_llama_server() -> Check:
    path = shutil.which("llama-server")
    if path:
        return ("llama-server", "OK", path)
    return (
        "llama-server",
        "WARN",
        "not on PATH - needed for runtime kind 'llama-server'. Either install it "
        f"({_LLAMA_RELEASES}) or point runtime at a running server (kind: openai-compat, "
        "e.g. LM Studio at http://localhost:1234/v1).",
    )


def _check_gpu() -> Check:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if not out:
        return (
            "GPU",
            "WARN",
            "nvidia-smi not available - VRAM gates will read 'not measured'. Fine for the demo; "
            "real evals want the GPU box.",
        )
    return ("GPU", "OK", out.splitlines()[0].strip())


def _check_lmstudio(spec: SweepSpec | None) -> Check:
    configured = spec.promote.lmstudio_dir if spec and spec.promote.lmstudio_dir else None
    models_dir = Path(configured) if configured else Path.home() / ".lmstudio" / "models"
    bits: list[str] = []
    level = "OK"
    if models_dir.exists():
        bits.append(f"models dir {models_dir}")
    else:
        level = "WARN"
        bits.append(
            f"models dir {models_dir} missing - promotion will create it, but check LM Studio "
            "is installed if you expect the model to appear in its UI"
        )
    try:
        httpx.get("http://localhost:1234/v1/models", timeout=2.0)
        bits.append("server responding on :1234")
    except httpx.HTTPError:
        bits.append("server not responding on :1234 (fine unless runtime/base_url points at it)")
    return ("LM Studio", level, "; ".join(bits))


def _check_disk(spec: SweepSpec | None) -> Check:
    target = Path(spec.workspace) if spec else Path.cwd()
    free = free_disk_gb(target)
    need = spec.min_free_disk_gb if spec else 80.0
    if free < need:
        return (
            "disk",
            "FAIL" if spec else "WARN",
            f"{free:.0f} GB free at {target.resolve().anchor}, sweep guard wants {need:g} GB "
            "(candidates weigh 15-35 GB each). Point `workspace:` at a larger drive.",
        )
    return ("disk", "OK", f"{free:.0f} GB free (guard: {need:g} GB)")


def _check_git_uv() -> Check:
    missing = [t for t in ("git", "uv") if not shutil.which(t)]
    if missing:
        return (
            "git/uv",
            "WARN",
            f"missing: {', '.join(missing)} - needed for the local-offload/remote prune profiles "
            "(cloning + building the REAP repo).",
        )
    return ("git/uv", "OK", "both on PATH")


def _check_provider(name: str, cfg: ProviderCfg) -> Check:
    if cfg.kind == "mock":
        return (name, "OK", "mock (offline)")
    if cfg.kind == "claude-cli":
        ok = shutil.which("claude") is not None
        return (name, "OK" if ok else "FAIL", "claude CLI " + ("found" if ok else "NOT on PATH"))
    if cfg.kind == "anthropic-api":
        env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        ok = bool(os.environ.get(env))
        return (name, "OK" if ok else "FAIL", f"${env} " + ("set" if ok else "is empty"))
    base = (cfg.base_url or "http://localhost:1234/v1").rstrip("/")
    try:
        httpx.get(f"{base}/models", timeout=3.0)
        return (name, "OK", f"{base} reachable")
    except httpx.HTTPError as e:
        return (name, "FAIL", f"{base} unreachable ({type(e).__name__}) - start the server first")


def run_doctor(console: Console, spec_path: Path | None = None, strict: bool = False) -> int:
    """Print the checkup table; returns the exit code (1 when strict and any FAIL)."""
    spec: SweepSpec | None = None
    checks: list[Check] = [_check_python()]

    if spec_path is not None:
        try:
            spec = SweepSpec.from_yaml(spec_path)
            pack = DomainPack.from_yaml(spec.domain_pack)
            checks.append(("sweep spec", "OK", f"{spec_path} (pack '{pack.name}', {len(pack.domains)} domains)"))
        except Exception as e:  # noqa: BLE001 - any spec error is a finding, not a crash
            checks.append(("sweep spec", "FAIL", f"{spec_path}: {e}"))

    checks.append(_check_claude_cli())
    checks.append(_check_llamacpp(spec))
    checks.append(_check_llama_server())
    checks.append(_check_gpu())
    checks.append(_check_lmstudio(spec))
    checks.append(_check_disk(spec))
    checks.append(_check_git_uv())
    if spec is not None:
        checks.append(_check_provider("generator provider", spec.generator))
        checks.append(_check_provider("judge provider", spec.judge.provider))
        if spec.runtime.kind == "openai-compat":
            checks.append(
                _check_provider(
                    "eval runtime", ProviderCfg(kind="openai-compat", base_url=spec.runtime.base_url)
                )
            )

    table = Table(title="reap-lab doctor", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    styles = {"OK": "green", "WARN": "yellow", "FAIL": "red"}
    for name, level, message in checks:
        table.add_row(name, f"[{styles[level]}]{level}[/{styles[level]}]", message)
    console.print(table)

    fails = [c for c in checks if c[1] == "FAIL"]
    warns = [c for c in checks if c[1] == "WARN"]
    if fails:
        console.print(f"[red]{len(fails)} FAIL[/red], [yellow]{len(warns)} WARN[/yellow] - fix FAILs before a real sweep.")
    elif warns:
        console.print(f"[yellow]{len(warns)} WARN[/yellow] - the demo runs fine; real sweeps may need attention above.")
    else:
        console.print("[green]All clear.[/green]")
    return 1 if (strict and fails) else 0
