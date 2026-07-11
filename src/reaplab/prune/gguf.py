"""GGUF conversion + quantization wrappers around llama.cpp tooling (PRD FR-2.4).

Ground truth: docs/RESEARCH_BRIEF.md section 2.

- ``convert_hf_to_gguf.py <hf_dir> --outfile out.gguf --outtype bf16`` -- the
  converter reads the expert count from ``config.json``, so pruned MoE
  checkpoints need no special flag. K-quants happen in ``llama-quantize``.
- ``llama-quantize in-bf16.gguf out.gguf Q4_K_M`` with the exact llama.cpp
  quant spelling.

llama.cpp is always an external subprocess (list argv, no shell); it is never
imported. Mock mode fabricates tiny GGUF-magic files so hashing/promotion
paths run offline.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from reaplab.prune.errors import PruneError, ToolNotFoundError

#: llama-quantize type names confirmed against llama.cpp (research brief:
#: Q4_K_M / Q5_K_M / Q6_K confirmed, plus the long-standing standard set).
CONFIRMED_QUANTS: frozenset[str] = frozenset(
    {
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_0",
        "Q4_1",
        "Q4_K_S",
        "Q4_K_M",
        "Q5_0",
        "Q5_1",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
        "IQ4_NL",
        "IQ4_XS",
        "F16",
        "BF16",
        "F32",
    }
)

#: convert_hf_to_gguf.py --outtype choices (brief section 2).
CONVERT_OUTTYPES: frozenset[str] = frozenset({"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"})

GGUF_MAGIC = b"GGUF"

_INSTALL_HINT = (
    "How to install llama.cpp tooling:\n"
    "  * llama-quantize (binary): download from https://github.com/ggml-org/llama.cpp/releases --\n"
    "    on Windows grab BOTH 'llama-<tag>-bin-win-cuda-12.4-x64.zip' AND\n"
    "    'cudart-llama-bin-win-cuda-12.4-x64.zip' and unzip them into ONE folder (e.g. C:\\llama.cpp).\n"
    "    (winget's ggml.llamacpp package is CPU-only; use the release zips for GPU.)\n"
    "  * convert_hf_to_gguf.py (script): clone https://github.com/ggml-org/llama.cpp -- the script\n"
    "    sits in the repo root; run 'pip install -r requirements.txt' inside that clone once.\n"
    "Then add the folder(s) to PATH, or set the LLAMA_CPP_DIR environment variable to the folder,\n"
    "or set LLAMA_QUANTIZE / CONVERT_HF_TO_GGUF to the exact file paths."
)


def validate_quant(quant: str) -> str:
    """Normalize and validate a quant name against the confirmed llama.cpp set.

    Accepts case-insensitive spellings (``q4_k_m`` -> ``Q4_K_M``); rejects
    anything not in :data:`CONFIRMED_QUANTS` (e.g. ``q4km``) with the closest
    valid suggestion. Returns the canonical uppercase name.
    """
    canonical = quant.strip().upper()
    if canonical in CONFIRMED_QUANTS:
        return canonical
    suggestion = difflib.get_close_matches(canonical, sorted(CONFIRMED_QUANTS), n=1, cutoff=0.4)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    raise PruneError(
        f"Unknown quantization type: '{quant}'.{hint}\n"
        f"Use the exact llama.cpp spelling. Confirmed names: {', '.join(sorted(CONFIRMED_QUANTS))}."
    )


def _common_dirs() -> list[Path]:
    """Places we look for llama.cpp tools on Windows (after explicit config and PATH)."""
    dirs: list[Path] = []
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    for base_env in ("LOCALAPPDATA", "ProgramFiles", "USERPROFILE"):
        base = os.environ.get(base_env)
        if base:
            dirs.append(Path(base) / "llama.cpp")
    dirs += [
        Path("C:/llama.cpp"),
        Path("C:/tools/llama.cpp"),
        Path.home() / "llama.cpp",
        Path.home() / "tools" / "llama.cpp",
    ]
    return dirs


def _resolve_tool(
    explicit: str | Path | None,
    *,
    env_var: str,
    names: list[str],
    kind: str,
) -> Path:
    """Discovery order: explicit path -> env var -> PATH -> common Windows dirs."""
    if explicit is not None:
        p = Path(explicit)
        if p.exists():
            return p
        raise ToolNotFoundError(
            f"Configured {kind} path does not exist: {p}\nFix the path or unset it to use auto-discovery.\n"
            + _INSTALL_HINT
        )
    env_val = os.environ.get(env_var)
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
        raise ToolNotFoundError(
            f"{env_var} points at a missing file: {p}\nFix or unset the environment variable.\n"
            + _INSTALL_HINT
        )
    for name in names:
        hit = shutil.which(name)
        if hit:
            return Path(hit)
    for d in _common_dirs():
        for name in names:
            candidate = d / name
            if candidate.exists():
                return candidate
    raise ToolNotFoundError(f"Could not find {kind} ({' / '.join(names)}).\n" + _INSTALL_HINT)


@dataclass(frozen=True)
class LlamaCppTools:
    """Resolved paths to the two llama.cpp tools this pipeline needs.

    Use :meth:`discover` rather than constructing directly, unless you already
    hold verified paths (tests do).
    """

    convert_script: Path
    quantize_bin: Path
    python_exe: str = field(default_factory=lambda: sys.executable)

    @classmethod
    def discover(
        cls,
        convert_script: str | Path | None = None,
        quantize_bin: str | Path | None = None,
        python_exe: str | None = None,
    ) -> LlamaCppTools:
        """Locate ``convert_hf_to_gguf.py`` and ``llama-quantize``.

        Order per tool: explicit argument -> env var (``CONVERT_HF_TO_GGUF`` /
        ``LLAMA_QUANTIZE``) -> PATH -> ``LLAMA_CPP_DIR`` and common Windows
        install folders. Raises :class:`ToolNotFoundError` with download
        instructions (GitHub release zips) when a tool cannot be found.
        """
        conv = _resolve_tool(
            convert_script,
            env_var="CONVERT_HF_TO_GGUF",
            names=["convert_hf_to_gguf.py"],
            kind="llama.cpp converter script",
        )
        quant = _resolve_tool(
            quantize_bin,
            env_var="LLAMA_QUANTIZE",
            names=["llama-quantize.exe", "llama-quantize"],
            kind="llama-quantize binary",
        )
        return cls(convert_script=conv, quantize_bin=quant, python_exe=python_exe or sys.executable)


def _run_tool(argv: list[str], *, what: str) -> None:
    """Run one llama.cpp tool; on failure raise a PruneError carrying the output tail."""
    try:
        result = subprocess.run(  # noqa: S603 - list argv, no shell
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError as e:
        raise ToolNotFoundError(f"{what}: executable not found: {argv[0]}\n" + _INSTALL_HINT) from e
    if result.returncode != 0:
        tail = ((result.stderr or "") + (result.stdout or ""))[-2000:]
        raise PruneError(
            f"{what} failed (exit {result.returncode}).\nCommand: {' '.join(argv)}\nOutput tail:\n{tail}"
        )


def convert_to_gguf(
    hf_dir: Path,
    out_path: Path,
    tools: LlamaCppTools,
    outtype: str = "bf16",
) -> Path:
    """HF checkpoint dir -> GGUF via ``convert_hf_to_gguf.py`` (default bf16).

    Always convert from BF16 checkpoints, never FP8/NVFP4 variants (brief
    section 1). K-quants are produced afterwards by :func:`quantize`.
    """
    hf_dir = Path(hf_dir)
    out_path = Path(out_path)
    if outtype not in CONVERT_OUTTYPES:
        raise PruneError(
            f"Invalid --outtype '{outtype}'. convert_hf_to_gguf.py accepts: "
            f"{', '.join(sorted(CONVERT_OUTTYPES))}."
        )
    if not (hf_dir / "config.json").exists():
        raise PruneError(
            f"Not an HF checkpoint directory (no config.json): {hf_dir}\n"
            "Point convert_to_gguf at the folder that save_pretrained produced."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        tools.python_exe,
        str(tools.convert_script),
        str(hf_dir),
        "--outfile",
        str(out_path),
        "--outtype",
        outtype,
    ]
    _run_tool(argv, what="GGUF conversion")
    if not out_path.exists():
        raise PruneError(
            f"convert_hf_to_gguf.py exited 0 but produced no file at {out_path} -- "
            "check the converter output for the actual location."
        )
    return out_path


def quantize(src_gguf: Path, out_path: Path, quant: str, tools: LlamaCppTools) -> Path:
    """bf16 GGUF -> quantized GGUF via ``llama-quantize``. Returns ``out_path``."""
    src_gguf = Path(src_gguf)
    out_path = Path(out_path)
    canonical = validate_quant(quant)
    if not src_gguf.exists():
        raise PruneError(
            f"Source GGUF not found: {src_gguf}\nRun the bf16 conversion first (convert_to_gguf)."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(tools.quantize_bin), str(src_gguf), str(out_path), canonical]
    _run_tool(argv, what=f"Quantization to {canonical}")
    if not out_path.exists():
        raise PruneError(f"llama-quantize exited 0 but produced no file at {out_path}.")
    return out_path


def deterministic_bytes(seed: str, size: int) -> bytes:
    """Deterministic pseudo-random bytes: same seed -> same bytes, any size."""
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}|{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


def write_fake_gguf(out_path: Path, seed: str, size_kb: int = 4) -> Path:
    """Mock-mode GGUF: correct magic bytes (``GGUF``) plus deterministic filler.

    Not a loadable model -- just enough structure that downstream hashing,
    manifest, and promotion code paths are exercised for real.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = deterministic_bytes(seed, max(1, size_kb) * 1024 - len(GGUF_MAGIC))
    out_path.write_bytes(GGUF_MAGIC + body)
    return out_path


def detect_quant_from_name(filename: str) -> str | None:
    """Best-effort quant detection from a GGUF filename (longest match wins).

    ``"Qwen3-30B-Q4_K_M.gguf" -> "Q4_K_M"``; returns None when nothing matches.
    """
    upper = filename.upper().replace("-", "_")
    for quant in sorted(CONFIRMED_QUANTS, key=len, reverse=True):
        if quant in upper:
            return quant
    return None
