"""`reap-lab init` — draft a domain pack + sweep spec from a plain-English
description of the user's workload.

Interactive by default; fully scriptable with --yes plus flags (used by tests and
CI). When a real provider is configured, it drafts the pack; otherwise (or on
provider failure) a sensible template pack is emitted, clearly marked for editing.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import typer
import yaml
from rich.console import Console

from reaplab.core.config import DomainPack, ProviderCfg, SweepSpec
from reaplab.core.providers import ProviderError, extract_json, get_provider

PACK_DRAFT_SYSTEM = """You design evaluation workload packs for LLM expert-pruning runs.
Given a user's description of what their local model does all day, produce a JSON object:
{"name": "<kebab-case>", "description": "<one sentence>", "domains": [
  {"name": "<snake_case>", "description": "...", "task_type": "exact"|"json_schema"|"open_ended"|"tool_call",
   "weight": <float, share of the workload>, "prompt_guidance": "<coverage/style notes for a prompt generator>",
   "long_context": <bool>} ]}
Rules: 4-7 domains proportional to the described workload; prefer open_ended unless outputs are
clearly a single verifiable answer (exact), a JSON document (json_schema - also include a
"json_schema" field holding a small JSON Schema), or a tool/function call (tool_call - also include
a "tools" field with 1-3 OpenAI-format function definitions the workload plausibly uses).
Set long_context true on at most 2 domains that plausibly involve long documents.
JSON only, no commentary."""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "my-workload"


def _detect_vram_gb() -> float | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return round(float(proc.stdout.strip().splitlines()[0]) / 1024, 1)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def _template_pack(name: str, description: str) -> DomainPack:
    """Fallback pack when no provider can draft one: broad, editable, valid."""
    return DomainPack.model_validate(
        {
            "name": name,
            "description": description or "Custom workload (edit me).",
            "domains": [
                {
                    "name": "primary_tasks",
                    "description": "The core work described by the user (EDIT: split into real domains).",
                    "task_type": "open_ended",
                    "weight": 3.0,
                    "prompt_guidance": description or "Realistic day-to-day requests.",
                },
                {
                    "name": "structured_extraction",
                    "description": "Pull structured JSON out of documents (EDIT or remove).",
                    "task_type": "json_schema",
                    "weight": 1.0,
                    "long_context": True,
                    "json_schema": {
                        "type": "object",
                        "required": ["summary", "entities"],
                        "properties": {
                            "summary": {"type": "string"},
                            "entities": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                {
                    "name": "general_chat",
                    "description": "Everyday assistant tasks for distribution balance.",
                    "task_type": "open_ended",
                    "weight": 1.0,
                    "prompt_guidance": "Summaries, rewrites, quick questions.",
                },
            ],
        }
    )


def _draft_pack_via_provider(
    console: Console, provider_cfg: ProviderCfg, name: str, description: str
) -> DomainPack | None:
    provider = get_provider(provider_cfg)
    prompt = f"Workload description:\n{description}\n\nPack name to use: {name}"
    last_error = ""
    for attempt in (1, 2):
        try:
            resp = provider.complete(
                prompt if attempt == 1 else f"{prompt}\n\nYour previous draft failed validation: "
                f"{last_error}. Fix it and return corrected JSON only.",
                system=PACK_DRAFT_SYSTEM,
                json_mode=True,
                max_tokens=3000,
            )
            data = extract_json(resp.text)
            if not isinstance(data, dict) or "domains" not in data:
                raise ValueError("provider returned JSON without a 'domains' list")
            data.setdefault("name", name)
            for dom in data["domains"]:
                # tool_call without tools cannot be scored - downgrade rather than fail
                if dom.get("task_type") == "tool_call" and not dom.get("tools"):
                    dom["task_type"] = "open_ended"
            return DomainPack.model_validate(data)
        except (ProviderError, ValueError, Exception) as e:  # noqa: BLE001 - fall back to template
            last_error = str(e)[:300]
            console.print(f"[yellow]pack draft attempt {attempt} failed:[/yellow] {last_error}")
    return None


def run_init(
    console: Console,
    out_dir: Path,
    *,
    name: str | None = None,
    model_id: str | None = None,
    describe: str | None = None,
    provider: str | None = None,
    yes: bool = False,
) -> tuple[Path, Path]:
    """Create <name>-pack.yaml and <name>-sweep.yaml in out_dir; returns both paths."""
    if not yes:
        console.print(
            "[bold]reap-lab init[/bold] - answer a few questions and get a domain pack "
            "+ sweep spec tailored to your workload.\n"
        )
        name = typer.prompt("Project name", default=name or "my-workload")
        model_id = typer.prompt("Base MoE model (HF id)", default=model_id or "Qwen/Qwen3-30B-A3B")
        describe = typer.prompt(
            "Describe what your local model does all day (one paragraph)",
            default=describe or "",
        )
        provider = typer.prompt(
            "Provider for generation/judging [claude-cli / openai-compat / anthropic-api / mock]",
            default=provider or "claude-cli",
        )
    name = _slug(name or "my-workload")
    model_id = model_id or "Qwen/Qwen3-30B-A3B"
    describe = describe or ""
    provider = (provider or "claude-cli").strip()
    if provider not in ("claude-cli", "openai-compat", "anthropic-api", "mock"):
        raise typer.BadParameter(f"unknown provider kind {provider!r}")
    provider_cfg = ProviderCfg(kind=provider)  # type: ignore[arg-type]

    pack: DomainPack | None = None
    if provider != "mock":
        console.print(f"Drafting a domain pack from your description via {provider}...")
        pack = _draft_pack_via_provider(console, provider_cfg, name, describe)
    if pack is None:
        console.print("Using the editable template pack (mock provider or draft failed).")
        pack = _template_pack(name, describe)

    vram = _detect_vram_gb()
    gates_vram = round(vram * 0.85, 1) if vram else 40.0
    if vram:
        console.print(f"Detected {vram} GB VRAM -> gate max_vram_gb={gates_vram}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pack_path = out_dir / f"{name}-pack.yaml"
    sweep_path = out_dir / f"{name}-sweep.yaml"

    pack_path.write_text(
        yaml.safe_dump(pack.model_dump(mode="json", exclude_none=True), sort_keys=False, width=100),
        encoding="utf-8",
    )
    spec = SweepSpec(
        model_id=model_id,
        domain_pack=pack_path.name,  # sibling file; SweepSpec.from_yaml resolves it
        generator=provider_cfg,
        workspace="workspace",
    )
    spec.judge.provider = provider_cfg
    spec.gates.max_vram_gb = gates_vram
    sweep_path.write_text(
        yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True), sort_keys=False, width=100),
        encoding="utf-8",
    )

    console.print(f"\n[green]Wrote[/green] {pack_path}")
    console.print(f"[green]Wrote[/green] {sweep_path}")
    console.print(
        "\nNext steps:\n"
        f"  1. Review/edit the pack - it defines what 'good' means for your model.\n"
        f"  2. reap-lab doctor {sweep_path}\n"
        f"  3. reap-lab generate {sweep_path}   (datasets only; audit the sample it prints)\n"
        f"  4. reap-lab sweep {sweep_path}"
    )
    return pack_path, sweep_path
