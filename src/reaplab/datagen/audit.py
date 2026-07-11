"""Human-audit sampling (PRD M1): a stratified ~5% markdown sample of the eval set.

The exit criterion for datasets v1 is a human audit of a 5% sample. This module
writes that sample as reviewer-friendly markdown: stratified by domain
(largest-remainder over domain counts, every non-empty domain represented when
the budget allows), minimum 10 items, deterministic under the run seed.
"""

from __future__ import annotations

from pathlib import Path

from reaplab.core.records import EvalRecord
from reaplab.datagen.planning import largest_remainder
from reaplab.datagen.procedural import rng_for

_PROMPT_PREVIEW_CHARS = 1200


def _sample_counts(by_domain: dict[str, int], target: int) -> dict[str, int]:
    """Stratified allocation: proportional largest-remainder, then guarantee every
    non-empty domain at least one slot while the target budget allows."""
    counts = largest_remainder(target, {d: float(n) for d, n in by_domain.items()})
    counts = {d: min(c, by_domain[d]) for d, c in counts.items()}
    if target >= len(by_domain):
        for d, n in by_domain.items():
            if n > 0 and counts[d] == 0:
                donor = max(counts, key=lambda k: counts[k])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[d] = 1
    return counts


def _preview(text: str) -> str:
    if len(text) <= _PROMPT_PREVIEW_CHARS:
        return text
    return (
        text[:_PROMPT_PREVIEW_CHARS]
        + f"\n... [truncated for review; full prompt is {len(text):,} chars]"
    )


def write_audit_sample(
    records: list[EvalRecord],
    path: str | Path,
    *,
    seed: int = 42,
    fraction: float = 0.05,
    minimum: int = 10,
) -> Path:
    """Write a stratified audit sample of `records` to markdown at `path`.

    Sample size = min(len(records), max(minimum, round(fraction * len(records)))).
    Deterministic for a given seed. Returns the written path. An empty eval set
    still produces a (clearly marked) file so the pipeline stays inspectable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = len(records)
    target = min(total, max(minimum, round(fraction * total)))

    domains: dict[str, list[EvalRecord]] = {}
    for r in records:
        domains.setdefault(r.domain, []).append(r)

    lines: list[str] = [
        "# Eval dataset — human audit sample",
        "",
        f"{target} of {total} items (~{fraction:.0%}, stratified by domain, min {minimum}).",
        "For each item check: is the prompt realistic for the workload? Is the gold answer /",
        "rubric / expected tool actually correct? Flag anything synthetic-looking or leaky.",
        "",
    ]
    if total == 0:
        lines.append("_No eval records were available to sample — generation produced an empty set._")
    else:
        counts = _sample_counts({d: len(v) for d, v in domains.items()}, target)
        rng = rng_for(seed, "audit-sample")
        for domain, recs in domains.items():
            k = counts.get(domain, 0)
            if k <= 0:
                continue
            chosen = sorted(rng.sample(recs, k), key=lambda r: r.id)
            lines.append(f"## {domain} — {k} of {len(recs)} items")
            lines.append("")
            for r in chosen:
                lines.append(f"### {r.id} — {r.task_type.value}, {r.difficulty.value}")
                if r.tags:
                    lines.append(f"**Tags:** {', '.join(r.tags)}")
                lines.append("")
                lines.append("**Prompt:**")
                lines.append("```")
                lines.append(_preview(r.prompt))
                lines.append("```")
                if r.gold is not None:
                    lines.append(f"**Gold:** `{_preview(r.gold)}`")
                if r.rubric:
                    lines.append(f"**Rubric:** {r.rubric}")
                if r.expected_tool:
                    lines.append(f"**Expected tool:** `{r.expected_tool}`")
                lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path
