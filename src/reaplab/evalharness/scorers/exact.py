"""Exact/normalized-match scorer for structured short answers (PRD FR-3.2).

Normalization: casefold, collapse all whitespace runs to single spaces, strip edge
punctuation. Numeric tolerance: values that both parse as numbers compare numerically,
so "1,000.00" == "1000" and "$1,000.00" == "1000".

Unit classes are NOT interchangeable, though: separators are noise, but a currency
marker and a percent marker are meaning. "$10" against gold "10%" is a wrong answer
(0.0, detail flags unit_mismatch), not a full-credit numeric match. A value with no
marker at all still matches either class — the gold's own marker is authoritative and
a bare number is simply unmarked.
"""

from __future__ import annotations

import math
import re
from typing import Any

from reaplab.core.records import EvalRecord

_EDGE_PUNCT = ".,;:!?\"'`()[]{}<>*_~"
_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

CURRENCY = "currency"
PERCENT = "percent"
_CURRENCY_SYMBOLS = "$€£¥"
_CURRENCY_WORDS = re.compile(r"\b(?:usd|eur|gbp|dollars?|euros?|pounds?)\b")
_PERCENT_WORDS = re.compile(r"\b(?:percent|pct)\b")
_TRAILING_UNIT_WORDS = re.compile(r"(?:percent|pct|usd|eur|gbp|dollars?|euros?|pounds?)$")


def normalize(text: str) -> str:
    """Casefold, collapse whitespace, strip punctuation from both edges."""
    s = " ".join(text.split()).casefold()
    return s.strip(_EDGE_PUNCT + " ")


def as_measure(text: str) -> tuple[float, str | None] | None:
    """Parse a string as (value, unit_class) — unit_class is currency/percent/None.

    Separators (commas, spaces) and the unit marker itself are stripped before the
    numeric parse; "$1,000.00" -> (1000.0, "currency"), "10 percent" -> (10.0, "percent"),
    "1000" -> (1000.0, None). A value carrying BOTH kinds of marker ("$10%") parses as
    unit "mixed", which never equals anything. Non-numeric text -> None.
    """
    s = normalize(text)
    percent = "%" in s or bool(_PERCENT_WORDS.search(s))
    currency = any(c in s for c in _CURRENCY_SYMBOLS) or bool(_CURRENCY_WORDS.search(s))
    unit: str | None
    if percent and currency:
        unit = "mixed"
    elif percent:
        unit = PERCENT
    elif currency:
        unit = CURRENCY
    else:
        unit = None

    core = re.sub(r"[,\s]", "", s)
    core = _TRAILING_UNIT_WORDS.sub("", core)
    core = core.strip(_CURRENCY_SYMBOLS + "%")
    if not _NUM_RE.match(core):
        return None
    try:
        return float(core), unit
    except ValueError:
        return None


def as_number(text: str) -> float | None:
    """Numeric value of a string, ignoring unit markers ("$1,000.00" -> 1000.0)."""
    measure = as_measure(text)
    return measure[0] if measure else None


class ExactScorer:
    """Full credit iff the normalized response equals the normalized gold
    (or both parse as numbers of a compatible unit class, within tolerance).
    No partial credit."""

    name = "exact"

    def score(self, item: EvalRecord, response: Any) -> tuple[float, bool, dict[str, Any]]:
        if item.gold is None:
            return 0.0, False, {
                "error": f"eval item {item.id!r} has task_type=exact but no gold answer; "
                "fix the eval dataset (every exact item needs a gold string)."
            }
        got, want = normalize(response.text), normalize(item.gold)
        if got == want:
            return 1.0, True, {"normalized_match": True}
        g_measure, w_measure = as_measure(response.text), as_measure(item.gold)
        if g_measure is not None and w_measure is not None:
            g_num, g_unit = g_measure
            w_num, w_unit = w_measure
            if g_unit is not None and w_unit is not None and g_unit != w_unit:
                return 0.0, False, {
                    "numeric_match": False, "unit_mismatch": True,
                    "got": g_num, "want": w_num, "got_unit": g_unit, "want_unit": w_unit,
                }
            if math.isclose(g_num, w_num, rel_tol=1e-9, abs_tol=1e-9):
                return 1.0, True, {"numeric_match": True, "value": w_num}
            return 0.0, False, {"numeric_match": False, "got": g_num, "want": w_num}
        return 0.0, False, {"normalized_match": False, "got": got[:200], "want": want[:200]}
