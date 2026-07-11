from __future__ import annotations

from typing import Any

import pytest

from reaplab.core.config import ProviderCfg
from reaplab.core.providers import LLMProvider, LLMResponse
from reaplab.core.providers.mock import MockProvider
from reaplab.core.records import TaskType
from reaplab.evalharness.scorers.judge import cache_key, judge_item


class ScriptedJudge(LLMProvider):
    """Returns scripted replies in order and records every prompt it sees."""

    name = "scripted"

    def __init__(self, replies: list[str]):
        super().__init__(ProviderCfg(kind="mock"))
        self.replies = replies
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kw: Any) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(text=self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)])


class CountingMock(MockProvider):
    def __init__(self, cfg: ProviderCfg):
        super().__init__(cfg)
        self.calls = 0

    def complete(self, prompt: str, **kw: Any) -> LLMResponse:
        self.calls += 1
        return super().complete(prompt, **kw)


@pytest.fixture
def item(make_record):
    return make_record(
        id="ev-open-1",
        task_type=TaskType.OPEN_ENDED,
        prompt="Draft a polite payment reminder.",
        rubric="Prefer the response that is professional, complete, and actionable.",
    )


def _judge(item, provider, votes, tmp_path, artifact_hash="art123"):
    return judge_item(
        item, "CANDIDATE_TEXT", "BASELINE_TEXT", provider,
        votes=votes, judge_version="j1", cache_dir=tmp_path / "cache", artifact_hash=artifact_hash,
    )


def test_majority_win(item, tmp_path):
    # v0 unswapped: A=candidate -> "A" is a win; v1 swapped: "B" is a win; v2 unswapped: "A".
    judge = ScriptedJudge(['{"winner": "A"}', '{"winner": "B"}', '{"winner": "A"}'])
    score, detail = _judge(item, judge, 3, tmp_path)
    assert score == 1.0
    assert detail["majority"] == "win"
    assert detail["votes"] == ["win", "win", "win"]
    assert len(judge.prompts) == 3


def test_majority_loss_and_mixed_tie(item, tmp_path):
    # all three votes name the BASELINE'S slot: loss regardless of position
    judge = ScriptedJudge(['{"winner": "B"}', '{"winner": "A"}', '{"winner": "B"}'])
    score, detail = _judge(item, judge, 3, tmp_path)
    assert score == 0.0 and detail["majority"] == "loss"

    # win / loss / tie -> no strict majority -> tie
    judge2 = ScriptedJudge(['{"winner": "A"}', '{"winner": "A"}', '{"winner": "tie"}'])
    score2, detail2 = _judge(item, judge2, 3, tmp_path, artifact_hash="art456")
    assert detail2["votes"] == ["win", "loss", "tie"]
    assert (score2, detail2["majority"]) == (0.5, "tie")


def test_position_swap_on_alternate_votes(item, tmp_path):
    judge = ScriptedJudge(['{"winner": "tie"}'] * 2)
    _judge(item, judge, 2, tmp_path)
    # vote 0: candidate in slot A; vote 1: swapped, baseline in slot A
    assert "Response A:\nCANDIDATE_TEXT" in judge.prompts[0]
    assert "Response B:\nBASELINE_TEXT" in judge.prompts[0]
    assert "Response A:\nBASELINE_TEXT" in judge.prompts[1]
    assert "Response B:\nCANDIDATE_TEXT" in judge.prompts[1]


def test_rubric_included_in_prompt(item, tmp_path):
    judge = ScriptedJudge(['{"winner": "tie"}'])
    _judge(item, judge, 1, tmp_path)
    assert "professional, complete, and actionable" in judge.prompts[0]


def test_malformed_judge_output_counts_as_tie(item, tmp_path):
    judge = ScriptedJudge(["utter nonsense, no json here", '{"verdict": "A"}', '{"winner": "Q"}'])
    score, detail = _judge(item, judge, 3, tmp_path)
    assert score == 0.5
    assert detail["votes"] == ["tie", "tie", "tie"]
    assert all(v.get("malformed") for v in detail["vote_details"])


def test_cache_hit_never_calls_provider(item, tmp_path):
    provider = CountingMock(ProviderCfg(kind="mock"))
    score1, detail1 = _judge(item, provider, 3, tmp_path)
    calls_after_first = provider.calls
    assert calls_after_first == 3
    assert detail1["cached"] is False

    score2, detail2 = _judge(item, provider, 3, tmp_path)
    assert provider.calls == calls_after_first  # ZERO new calls
    assert detail2["cached"] is True
    assert score2 == score1


def test_cache_key_dimensions(item, tmp_path):
    provider = CountingMock(ProviderCfg(kind="mock"))
    _judge(item, provider, 1, tmp_path, artifact_hash="art-A")
    _judge(item, provider, 1, tmp_path, artifact_hash="art-B")  # different artifact -> re-judge
    assert provider.calls == 2
    assert cache_key("i", "a", "j1") != cache_key("i", "a", "j2")
    assert cache_key("i", "a", "j1") == cache_key("i", "a", "j1")


def test_corrupt_cache_file_is_rejudged(item, tmp_path):
    provider = CountingMock(ProviderCfg(kind="mock"))
    _judge(item, provider, 1, tmp_path)
    cache_dir = tmp_path / "cache"
    [cache_file] = list(cache_dir.glob("*.json"))
    cache_file.write_text("{ not json", encoding="utf-8")
    _judge(item, provider, 1, tmp_path)
    assert provider.calls == 2


def test_votes_must_be_positive(item, tmp_path):
    with pytest.raises(ValueError, match="votes"):
        _judge(item, ScriptedJudge(["x"]), 0, tmp_path)
