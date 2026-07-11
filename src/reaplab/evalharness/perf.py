"""Performance capture (PRD FR-3.4): prefill/decode tok/s, load time, peak VRAM.

Prefers llama-server's per-request timings (prompt_per_second / predicted_per_second,
present when the request sets "timings_per_token": true — see docs/RESEARCH_BRIEF.md §2).
Servers without timings fall back to a coarse derivation from token counts over total
request latency (an underestimate, flagged in the docstring so reports can say so).
"""

from __future__ import annotations

from reaplab.core.records import PerfMetrics
from reaplab.evalharness.runners import ModelRunner, RunnerError

_GEN_TOKENS = 128  # short, fixed-length generation keeps perf sampling cheap and comparable


def capture_perf(runner: ModelRunner, context: int, sample_prompts: list[str]) -> PerfMetrics:
    """Run a few short completions and aggregate throughput for one context size.

    Constraints: the runner must already be started at `context`; sample_prompts
    must be non-empty (a default prompt is substituted if not). Individual failed
    samples are skipped; if every sample fails, throughput fields stay None rather
    than raising — perf capture must never sink an otherwise-good eval.
    """
    prompts = [p for p in sample_prompts if p and p.strip()] or [
        "Summarize the key deadlines a small business faces in a typical fiscal quarter."
    ]
    prefill: list[float] = []
    decode: list[float] = []
    for prompt in prompts:
        try:
            resp = runner.complete(prompt, max_tokens=_GEN_TOKENS, temperature=0.0)
        except RunnerError:
            continue
        t = resp.timings or {}
        pp, dp = t.get("prompt_per_second"), t.get("predicted_per_second")
        if isinstance(pp, (int, float)) and pp > 0:
            prefill.append(float(pp))
        if isinstance(dp, (int, float)) and dp > 0:
            decode.append(float(dp))
        elif resp.completion_tokens and resp.latency_ms and resp.latency_ms > 0:
            # coarse fallback: whole-request latency spans prefill+decode, so this
            # underestimates true decode speed — better than nothing on servers
            # that ignore timings_per_token.
            decode.append(resp.completion_tokens / (resp.latency_ms / 1000.0))
            if resp.prompt_tokens and not (isinstance(pp, (int, float)) and pp > 0):
                prefill.append(resp.prompt_tokens / (resp.latency_ms / 1000.0))

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    return PerfMetrics(
        context=context,
        load_time_s=getattr(runner, "load_time_s", None),
        prefill_tps=mean(prefill),
        decode_tps=mean(decode),
        peak_vram_mb=getattr(runner, "peak_vram_mb", None),
    )
