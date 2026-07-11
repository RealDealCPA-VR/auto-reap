"""`reap-lab demo` — the full pipeline, offline, in under a minute.

Runs data generation -> (mock) pruning -> (mock) GGUF conversion -> evaluation ->
gates -> ranked report -> promotion into a sandboxed LM Studio directory, entirely
with deterministic mocks: no GPU, no network, no API keys. The point is to show
every artifact the real pipeline produces before you spend a cent.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from reaplab.core.config import (
    DataCfg,
    Gates,
    JudgeCfg,
    PromoteCfg,
    ProviderCfg,
    PruneCfg,
    RuntimeCfg,
    SweepSpec,
)

# Self-contained demo pack: covers every task type (and both refusal suites are
# auto-added), so every scorer and gate lights up in the report.
DEMO_PACK_YAML = """\
name: demo-assistant
description: >
  A small business assistant workload: answering questions, drafting email,
  classifying requests, extracting structured data, and calling scheduling tools.
include_refusal_suites: true
domains:
  - name: qa
    description: Practical how-to questions a small business owner asks.
    task_type: open_ended
    weight: 2.0
    prompt_guidance: Concrete, actionable questions; rubrics reward correct, complete steps.
  - name: email_drafting
    description: Draft short professional emails in a friendly voice.
    task_type: open_ended
    weight: 1.5
    prompt_guidance: Reminders, follow-ups, scheduling requests; reward tone and brevity.
  - name: classify_request
    description: Classify an inbound request into a category.
    task_type: exact
    weight: 2.0
    prompt_guidance: One correct category from a provided list; include ambiguous edge cases.
  - name: extract_order
    description: Extract order details into JSON.
    task_type: json_schema
    weight: 1.5
    long_context: true
    json_schema:
      type: object
      required: [customer, items]
      properties:
        customer: { type: string }
        items:
          type: array
          items:
            type: object
            required: [sku, qty]
            properties:
              sku: { type: string }
              qty: { type: number }
  - name: scheduling_tools
    description: Call the right scheduling tool with valid arguments.
    task_type: tool_call
    weight: 1.0
    tools:
      - type: function
        function:
          name: create_event
          description: Create a calendar event.
          parameters:
            type: object
            required: [title, start]
            properties:
              title: { type: string }
              start: { type: string, description: ISO datetime }
              duration_minutes: { type: number }
      - type: function
        function:
          name: find_free_slot
          description: Find a free time slot.
          parameters:
            type: object
            required: [duration_minutes]
            properties:
              duration_minutes: { type: number }
              after: { type: string }
"""


def build_demo_spec(workspace: Path, calibration_size: int, eval_size: int) -> SweepSpec:
    """Assemble the all-mock demo sweep. The domain-regression gate is loosened to
    12 pts (production default: 5) because demo-sized eval sets (~20 items/domain)
    carry sampling noise a real overnight run doesn't."""
    workspace.mkdir(parents=True, exist_ok=True)
    pack_path = workspace / "demo-pack.yaml"
    pack_path.write_text(DEMO_PACK_YAML, encoding="utf-8")
    return SweepSpec(
        model_id="demo/MoE-128x1B",
        domain_pack=str(pack_path),
        retention=[0.75, 0.625, 0.50],
        quants=["Q4_K_M", "Q5_K_M"],
        generator=ProviderCfg(kind="mock"),
        judge=JudgeCfg(provider=ProviderCfg(kind="mock")),
        data=DataCfg(calibration_size=calibration_size, eval_size=eval_size),
        prune=PruneCfg(execution_profile="mock"),
        runtime=RuntimeCfg(kind="mock"),
        gates=Gates(max_domain_regression_pts=12.0),
        promote=PromoteCfg(lmstudio_dir=str(workspace / "lmstudio-models")),
        workspace=str(workspace / "workspace"),
        min_free_disk_gb=1.0,
    )


def run_demo(
    console: Console,
    workspace: Path,
    calibration_size: int = 150,
    eval_size: int = 150,
    show_report: bool = True,
) -> Path:
    """Execute the demo sweep; returns the report path."""
    from reaplab.orchestrate import run_sweep  # noqa: PLC0415 - keep CLI import light

    console.print(
        Panel.fit(
            "reap-lab demo: full pipeline with deterministic mocks\n"
            "data -> prune -> GGUF -> eval -> gates -> report -> promote\n"
            "No GPU, no network, no API keys.",
            title="demo",
        )
    )
    spec = build_demo_spec(workspace, calibration_size, eval_size)
    # Also drop the sweep YAML so users can see exactly what a spec looks like.
    spec_path = workspace / "demo-sweep.yaml"
    import yaml  # noqa: PLC0415

    spec_path.write_text(
        yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True), sort_keys=False),
        encoding="utf-8",
    )

    report_path = run_sweep(spec, promote=True)

    if show_report:
        console.print()
        console.print(Markdown(report_path.read_text(encoding="utf-8")))
    console.print()
    console.print(f"[bold]Report:[/bold] {report_path}")
    console.print(f"[bold]Sweep spec:[/bold] {spec_path} (annotated example of the real thing)")
    console.print(f"[bold]Promoted GGUF (sandboxed):[/bold] {workspace / 'lmstudio-models'}")
    console.print(
        "\nEverything above ran offline with mocks. For the real thing:\n"
        "  1. reap-lab doctor        (check llama.cpp, providers, GPU)\n"
        "  2. reap-lab init          (draft a domain pack for YOUR workload)\n"
        "  3. reap-lab sweep <yaml>  (the same pipeline, for real)"
    )
    return report_path
