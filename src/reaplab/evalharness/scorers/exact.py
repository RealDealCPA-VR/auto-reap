"""Exact/normalized-match scorer for structured short answers (PRD FR-3.2).

Normalization: casefold, collapse all whitespace runs to single spaces, strip edge
punctuation. Numeric tolerance: values that both parse as numbers (commas and
currency symbols stripped) compare numerically, so "1,000.00" == "1000".
"""

from __future__ import annotations

import math
import re
from typing import Any

from reaplab.core.records import EvalRecord

_EDGE_PUNCT = ".,;:!?\"'`()[]{}<>*_~"
_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def normalize(text: str) -> str:
    """Casefold, collapse whitespace, strip punctuation from both edges."""
    s = " ".join(text.split()).casefold()
    return s.strip(_EDGE_PUNCT + " ")


def as_number(text: str) -> float | None:
    """Parse a normalized string as a number, tolerating commas/currency ("$1,000.00")."""
    s = normalize(text).replace(",", "").replace("$", "").replace("%", "").strip()
    if _NUM_RE.match(s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


class ExactScorer:
    """Full credit iff the normalized response equals the normalized gold
    (or both parse as numbers within tolerance). No partial credit."""

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
        g_num, w_num = as_number(response.text), as_number(item.gold)
        if g_num is not None and w_num is not None:
            if math.isclose(g_num, w_num, rel_tol=1e-9, abs_tol=1e-9):
                return 1.0, True, {"numeric_match": True, "value": w_num}
            return 0.0, False, {"numeric_match": False, "got": g_num, "want": w_num}
        return 0.0, False, {"normalized_match": False, "got": got[:200], "want": want[:200]}
