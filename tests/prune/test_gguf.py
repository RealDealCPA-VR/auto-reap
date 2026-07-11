"""gguf: quant validation, fake GGUF fabrication, tool discovery, subprocess argv."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reaplab.prune import gguf
from reaplab.prune.errors import PruneError, ToolNotFoundError
from reaplab.prune.gguf import (
    LlamaCppTools,
    convert_to_gguf,
    detect_quant_from_name,
    quantize,
    validate_quant,
    write_fake_gguf,
)


class TestValidateQuant:
    @pytest.mark.parametrize("name", ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q4_K_S"])
    def test_confirmed_names_pass(self, name: str):
        assert validate_quant(name) == name

    def test_lowercase_is_normalized(self):
        assert validate_quant("q4_k_m") == "Q4_K_M"

    def test_q4km_rejected_with_suggestion(self):
        with pytest.raises(PruneError) as exc:
            validate_quant("q4km")
        msg = str(exc.value)
        assert "q4km" in msg
        assert "Q4_K_M" in msg  # suggestion or confirmed list names the right spelling

    @pytest.mark.parametrize("bad", ["", "Q4KM", "int4", "GPTQ"])
    def test_garbage_rejected(self, bad: str):
        with pytest.raises(PruneError):
            validate_quant(bad)


class TestFakeGguf:
    def test_magic_bytes_and_size(self, tmp_path: Path):
        out = write_fake_gguf(tmp_path / "m.gguf", seed="s1", size_kb=4)
        blob = out.read_bytes()
        assert blob[:4] == b"GGUF"
        assert len(blob) == 4 * 1024

    def test_deterministic_per_seed(self, tmp_path: Path):
        a = write_fake_gguf(tmp_path / "a.gguf", seed="same").read_bytes()
        b = write_fake_gguf(tmp_path / "b.gguf", seed="same").read_bytes()
        c = write_fake_gguf(tmp_path / "c.gguf", seed="other").read_bytes()
        assert a == b
        assert a != c


class TestDiscovery:
    def test_missing_tools_error_points_at_release_zips(self, no_llama_tools):
        with pytest.raises(ToolNotFoundError) as exc:
            LlamaCppTools.discover()
        msg = str(exc.value)
        assert "github.com/ggml-org/llama.cpp/releases" in msg
        assert "cudart" in msg  # the Windows two-zip gotcha is spelled out

    def test_explicit_paths_win(self, no_llama_tools, tmp_path: Path):
        conv = tmp_path / "convert_hf_to_gguf.py"
        quant = tmp_path / "llama-quantize.exe"
        conv.touch()
        quant.touch()
        tools = LlamaCppTools.discover(convert_script=conv, quantize_bin=quant)
        assert tools.convert_script == conv
        assert tools.quantize_bin == quant

    def test_explicit_missing_path_is_instructive(self, no_llama_tools, tmp_path: Path):
        with pytest.raises(ToolNotFoundError, match="does not exist"):
            LlamaCppTools.discover(
                convert_script=tmp_path / "ghost.py", quantize_bin=tmp_path / "ghost.exe"
            )

    def test_env_var_discovery(self, no_llama_tools, monkeypatch, tmp_path: Path):
        conv = tmp_path / "convert_hf_to_gguf.py"
        quant = tmp_path / "llama-quantize.exe"
        conv.touch()
        quant.touch()
        monkeypatch.setenv("CONVERT_HF_TO_GGUF", str(conv))
        monkeypatch.setenv("LLAMA_QUANTIZE", str(quant))
        tools = LlamaCppTools.discover()
        assert tools.convert_script == conv
        assert tools.quantize_bin == quant

    def test_llama_cpp_dir_discovery(self, no_llama_tools, monkeypatch, tmp_path: Path):
        d = tmp_path / "llamacpp"
        d.mkdir()
        (d / "convert_hf_to_gguf.py").touch()
        (d / "llama-quantize.exe").touch()
        monkeypatch.setenv("LLAMA_CPP_DIR", str(d))
        monkeypatch.setattr(gguf, "_common_dirs", lambda: [Path(v) for v in [d]])
        tools = LlamaCppTools.discover()
        assert tools.quantize_bin == d / "llama-quantize.exe"


@pytest.fixture
def fake_tools(tmp_path: Path) -> LlamaCppTools:
    conv = tmp_path / "convert_hf_to_gguf.py"
    quant = tmp_path / "llama-quantize.exe"
    conv.touch()
    quant.touch()
    return LlamaCppTools(convert_script=conv, quantize_bin=quant, python_exe="python")


class TestSubprocessArgv:
    def test_convert_builds_exact_argv(self, fake_tools: LlamaCppTools, tmp_path: Path, monkeypatch):
        hf_dir = tmp_path / "ckpt"
        hf_dir.mkdir()
        (hf_dir / "config.json").write_text("{}", encoding="utf-8")
        out = tmp_path / "model-bf16.gguf"
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            out.write_bytes(b"GGUF-real-enough")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(gguf.subprocess, "run", fake_run)
        convert_to_gguf(hf_dir, out, fake_tools, outtype="bf16")
        assert calls == [
            [
                "python",
                str(fake_tools.convert_script),
                str(hf_dir),
                "--outfile",
                str(out),
                "--outtype",
                "bf16",
            ]
        ]

    def test_quantize_builds_exact_argv(self, fake_tools: LlamaCppTools, tmp_path: Path, monkeypatch):
        src = tmp_path / "in-bf16.gguf"
        src.write_bytes(b"GGUF")
        out = tmp_path / "out-Q4_K_M.gguf"
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            out.write_bytes(b"GGUF")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(gguf.subprocess, "run", fake_run)
        quantize(src, out, "q4_k_m", fake_tools)
        assert calls == [[str(fake_tools.quantize_bin), str(src), str(out), "Q4_K_M"]]

    def test_convert_rejects_bad_outtype(self, fake_tools: LlamaCppTools, tmp_path: Path):
        with pytest.raises(PruneError, match="outtype"):
            convert_to_gguf(tmp_path, tmp_path / "o.gguf", fake_tools, outtype="q4_k_m")

    def test_convert_rejects_non_checkpoint_dir(self, fake_tools: LlamaCppTools, tmp_path: Path):
        with pytest.raises(PruneError, match="config.json"):
            convert_to_gguf(tmp_path / "empty", tmp_path / "o.gguf", fake_tools)

    def test_tool_failure_carries_output_tail(self, fake_tools: LlamaCppTools, tmp_path: Path, monkeypatch):
        src = tmp_path / "in.gguf"
        src.write_bytes(b"GGUF")

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="CUDA error: OOM")

        monkeypatch.setattr(gguf.subprocess, "run", fake_run)
        with pytest.raises(PruneError, match="CUDA error: OOM"):
            quantize(src, tmp_path / "o.gguf", "Q4_K_M", fake_tools)


class TestDetectQuantFromName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Qwen3-30B-A3B-Q4_K_M.gguf", "Q4_K_M"),
            ("model-q5_k_m.gguf", "Q5_K_M"),
            ("model.bf16.gguf", "BF16"),
            ("mystery.gguf", None),
        ],
    )
    def test_detection(self, name: str, expected: str | None):
        assert detect_quant_from_name(name) == expected
