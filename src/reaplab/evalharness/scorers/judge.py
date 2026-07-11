"""Pairwise LLM judge vs. the unpruned baseline for open-ended items (PRD FR-3.3).

- Rubric-guided prompt; the judge sees two anonymized responses (A/B).
- Position bias control: candidate/baseline swap A/B slots on alternate votes.
- n votes, majority of win/tie/loss -> score 1.0 / 0.5 / 0.0.
- Judgments cached on disk keyed sha256(item_id, artifact_hash, judge_version):
  a cache hit never calls the provider, so re-runs cost nothing.
- Malformed judge output (no parseable {"winner": ...}) counts as a tie for that
  vote and is flagged in detail — one flaky judgment never sinks an item.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reaplab.core.providers import LLMProvider, ProviderError, extract_json
from reaplab.core.records import EvalRecord

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator comparing two assistant responses to the same "
    "prompt. Judge only quality: accuracy, completeness, instruction-following, and "
    "(when given) the rubric. Ignore response order, length for its own sake, and style. "
    'Reply with a single JSON object: {"winner": "A" | "B" | "tie", "reason": "<one sentence>"}.'
)

_PROMPT_TEMPLATE = """Task prompt:
{prompt}
{rubric_block}
Response A:
{a}

Response B:
{b}

Which response is better? Reply with JSON only: {{"winner": "A" | "B" | "tie", "reason": "..."}}"""


def cache_key(item_id: str, artifact_hash: str, judge_version: str) -> str:
    """Stable cache key per PRD FR-3.3: (item_id, artifact_hash, judge_version)."""
    payload = json.dumps([item_id, artifact_hash, judge_version], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _one_vote(
    provider: LLMProvider,
    item: EvalRecord,
    candidate_text: str,
    baseline_text: str,
    *,
    swap: bool,
) -> tuple[str, dict[str, Any]]:
    """One judged comparison. Returns (outcome, info) with outcome in win/tie/loss
    from the CANDIDATE's perspective. swap=True puts the baseline in slot A."""
    a, b = (baseline_text, candidate_text) if swap else (candidate_text, baseline_text)
    rubric_block = f"\nScoring rubric:\n{item.rubric}\n" if item.rubric else ""
    prompt = _PROMPT_TEMPLATE.format(prompt=item.prompt, rubric_block=rubric_block, a=a, b=b)
    try:
        resp = provider.complete(prompt, system=JUDGE_SYSTEM, temperature=0.0, json_mode=True)
        verdict = extract_json(resp.text)
        winner = str(verdict.get("winner", "")).strip().upper() if isinstance(verdict, dict) else ""
    except (ProviderError, ValueError) as e:
        return "tie", {"malformed": True, "error": str(e)[:200], "swapped": swap}
    if winner not in ("A", "B", "TIE"):
        return "tie", {"malformed": True, "raw_winner": winner[:40], "swapped": swap}
    if winner == "TIE":
        return "tie", {"swapped": swap}
    candidate_slot = "B" if swap else "A"
    outcome = "win" if winner == candidate_slot else "loss"
    reason = verdict.get("reason") if isinstance(verdict, dict) else None
    return outcome, {"swapped": swap, "reason": str(reason)[:200] if reason else None}


def judge_item(
    item: EvalRecord,
    candidate_text: str,
    baseline_text: str,
    provider: LLMProvider,
    votes: int,
    judge_version: str,
    cache_dir: str | Path,
    artifact_hash: str,
) -> tuple[float, dict[str, Any]]:
    """Pairwise-judge one item; returns (score in {0.0, 0.5, 1.0}, detail).

    Constraints: votes >= 1; cache_dir is created if missing; a cache hit returns
    the stored result WITHOUT any provider call. Corrupt cache files are ignored
    and re-judged (then overwritten).
    """
    if votes < 1:
        raise ValueError(f"judge votes must be >= 1, got {votes}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key(item.id, artifact_hash, judge_version)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            detail = dict(cached["detail"])
            detail["cached"] = True
            return float(cached["score"]), detail
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass  # corrupt cache entry: re-judge and overwrite

    outcomes: list[str] = []
    vote_infos: list[dict[str, Any]] = []
    for v in range(votes):
        outcome, info = _one_vote(provider, item, candidate_text, baseline_text, swap=bool(v % 2))
        outcomes.append(outcome)
        vote_infos.append(info)

    counts = Counter(outcomes)
    top = counts.most_common()
    majority = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "tie"
    score = {"win": 1.0, "tie": 0.5, "loss": 0.0}[majority]
    detail: dict[str, Any] = {
        "majority": majority,
        "votes": outcomes,
        "vote_details": vote_infos,
        "judge_version": judge_version,
        "cached": False,
    }
    tmp = cache_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"score": score, "detail": detail}), encoding="utf-8")
    tmp.replace(cache_file)
    return score, detail
