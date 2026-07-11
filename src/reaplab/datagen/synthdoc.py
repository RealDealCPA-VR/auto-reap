"""Synthetic long documents for >=16k-token prompts (PRD FR-1.4).

Long-context items are built by prepending a large, clearly-synthetic reference
document to an ordinary prompt. The document is wrapped in sentinel markers so
that:

- the near-duplicate filter can compare only the *task* part of a prompt
  (two long items sharing paragraph templates must not collapse into one), and
- humans auditing the sample can skip the filler at a glance.

Token counts are estimated at ~4 characters/token, the usual BPE ballpark; the
builder overshoots the target by ~10% so estimates stay comfortably above the
16k floor.
"""

from __future__ import annotations

import re
from random import Random

DOC_BEGIN = "<<<BEGIN SYNTHETIC REFERENCE DOCUMENT>>>"
DOC_END = "<<<END SYNTHETIC REFERENCE DOCUMENT>>>"

_DOC_SPAN = re.compile(re.escape(DOC_BEGIN) + r".*?" + re.escape(DOC_END), re.DOTALL)

_ENTITIES = [
    "Meridian", "Blue Harbor", "Stonebridge", "Cascade", "Ironwood", "Lakeshore",
    "Summit Ridge", "Copperfield", "Northgate", "Silver Birch", "Redwood", "Harborview",
]
_SUFFIXES = ["LLC", "Inc.", "Partners", "Group", "Holdings"]
_PEOPLE = [
    "Alvarez", "Chen", "Osei", "Novak", "Whitfield", "Iyer",
    "Marchetti", "Okafor", "Delgado", "Fitzgerald", "Yamada", "Kowalski",
]
_CITIES = [
    "Austin", "Tacoma", "Columbus", "Mesa", "Providence", "Boise",
    "Savannah", "Duluth", "Fresno", "Richmond", "Omaha", "Trenton",
]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_PARA_TEMPLATES = [
    "{entity} ({city}) reported a period-end balance of ${amount} for {month} {year}. "
    "The supporting schedule was prepared by {person} and cross-referenced to workpaper {ref}. "
    "No exceptions were noted during the tie-out, though two immaterial rounding differences remain open.",
    "Item {ref}: correspondence received from {entity} on {month} {day}, {year} regarding account "
    "reconciliation for the {city} location. {person} logged the request, attached the source detail, "
    "and flagged a follow-up for the amount of ${amount} pending confirmation.",
    "Summary of activity -- {month} {year}: {entity} recorded {count} transactions totaling ${amount}. "
    "The largest single entry, reviewed by {person}, relates to the {city} facility and carries "
    "reference {ref}. Classification is consistent with the prior period.",
    "Note {ref}: during the {month} {year} review, {person} observed that documentation supplied by "
    "{entity} was partially truncated; pages 3-5 of the {city} statement were re-requested. Interim "
    "totals of ${amount} are carried forward subject to that receipt.",
    "Approval trail: the ${amount} adjustment proposed for {entity} ({city}) was drafted on {month} "
    "{day}, {year}, seconded by {person}, and posted under reference {ref}. Reversal criteria are "
    "documented in the standing procedures memo.",
    "Aging detail for {entity}: {count} open items as of {month} {day}, {year}, aggregate ${amount}. "
    "{person} confirmed the two oldest items with the {city} office; both carry reference {ref} and "
    "are expected to clear next cycle.",
]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Deterministic; no tokenizer dep."""
    return max(1, len(text) // 4)


def build_long_document(rng: Random, *, min_tokens: int = 16_000, topic: str = "") -> str:
    """Deterministically build a synthetic reference document of >= min_tokens
    estimated tokens. `rng` drives all variation -- same rng state, same text."""
    target_chars = int(min_tokens * 4 * 1.1)  # overshoot ~10% above the floor
    parts: list[str] = []
    if topic:
        parts.append(f"Reference dossier -- background material related to: {topic}\n")
    section = 0
    total = sum(len(p) for p in parts)
    while total < target_chars:
        section += 1
        heading = f"\n== Section {section} ==\n"
        para = rng.choice(_PARA_TEMPLATES).format(
            entity=f"{rng.choice(_ENTITIES)} {rng.choice(_SUFFIXES)}",
            city=rng.choice(_CITIES),
            person=f"{rng.choice('ABCDEFGHJKLM')}. {rng.choice(_PEOPLE)}",
            month=rng.choice(_MONTHS),
            day=rng.randint(1, 28),
            year=rng.randint(2019, 2025),
            amount=f"{rng.randint(100, 999_999):,}.{rng.randint(0, 99):02d}",
            count=rng.randint(3, 240),
            ref=f"{rng.choice('QRSTX')}{rng.randint(1000, 9999)}",
        )
        parts.append(heading + para + "\n")
        total += len(heading) + len(para) + 1
    return "".join(parts)


def with_long_document(prompt: str, rng: Random, *, topic: str = "", min_tokens: int = 16_000) -> str:
    """Wrap `prompt` with a synthetic long document so the full prompt is
    >= min_tokens estimated tokens. The document sits between DOC_BEGIN/DOC_END
    markers; `comparable_text` strips it back out for dedup purposes."""
    doc = build_long_document(rng, min_tokens=min_tokens, topic=topic)
    return (
        f"{DOC_BEGIN}\n{doc}\n{DOC_END}\n\n"
        "Consult the reference document above where it is relevant, then complete "
        f"the following task.\n\n{prompt}"
    )


def comparable_text(prompt: str) -> str:
    """Normalize a prompt for near-duplicate comparison: drop embedded synthetic
    documents, collapse whitespace, lowercase. Comparing the task part only keeps
    long-context items from colliding on shared filler paragraphs."""
    stripped = _DOC_SPAN.sub(" ", prompt)
    return " ".join(stripped.lower().split())
