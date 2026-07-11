"""Procedural (offline) generation: determinism, scoreability, variety, refusal suites."""

from __future__ import annotations

import json

import jsonschema

from reaplab.core.records import EvalRecord, TaskType
from reaplab.datagen.dedup import filter_near_duplicates
from reaplab.datagen.procedural import generate_procedural_items, generate_refusal_suite
from reaplab.datagen.synthdoc import (
    DOC_BEGIN,
    DOC_END,
    build_long_document,
    comparable_text,
    estimate_tokens,
    with_long_document,
)


def _spec(pack, name):
    return next(d for d in pack.domains if d.name == name)


def test_same_seed_same_items(mini_pack):
    spec = _spec(mini_pack, "txn_classify")
    a = generate_procedural_items(spec, mini_pack, 42, "eval", 12)
    b = generate_procedural_items(spec, mini_pack, 42, "eval", 12)
    assert a == b


def test_different_seed_different_items(mini_pack):
    spec = _spec(mini_pack, "txn_classify")
    a = generate_procedural_items(spec, mini_pack, 42, "eval", 12)
    b = generate_procedural_items(spec, mini_pack, 43, "eval", 12)
    assert a != b


def test_calibration_and_eval_streams_are_independent(mini_pack):
    spec = _spec(mini_pack, "txn_classify")
    cal = generate_procedural_items(spec, mini_pack, 42, "calibration", 8)
    ev = generate_procedural_items(spec, mini_pack, 42, "eval", 8)
    assert {i["prompt"] for i in cal}.isdisjoint({i["prompt"] for i in ev})


def test_exact_items_are_scoreable(mini_pack):
    spec = _spec(mini_pack, "txn_classify")
    for item in generate_procedural_items(spec, mini_pack, 42, "eval", 10):
        assert item["gold"], "exact items must carry a gold answer"
        assert item["gold"] in item["prompt"], "the gold label must appear among the options"
        assert item["difficulty"] in {"easy", "medium", "hard"}


def test_json_schema_items_validate_against_domain_schema(mini_pack):
    spec = _spec(mini_pack, "report_extract")
    for item in generate_procedural_items(spec, mini_pack, 42, "eval", 8):
        gold = json.loads(item["gold"])
        jsonschema.validate(gold, spec.json_schema)  # raises on mismatch
        assert item["json_schema"] == spec.json_schema


def test_tool_call_items_carry_tools_and_expected_tool(mini_pack):
    spec = _spec(mini_pack, "ops_tools")
    names = {t["function"]["name"] for t in spec.tools}
    items = generate_procedural_items(spec, mini_pack, 42, "eval", 9)
    for item in items:
        assert item["tools"] == spec.tools
        assert item["expected_tool"] in names
    assert len({i["expected_tool"] for i in items}) > 1, "should exercise multiple tools"


def test_open_ended_items_have_rubrics(mini_pack):
    spec = _spec(mini_pack, "advisory")
    for item in generate_procedural_items(spec, mini_pack, 42, "eval", 6):
        assert "Reward" in item["rubric"] and "Penalize" in item["rubric"]
        # domain guidance flows into the rubric
        assert "authority" in item["rubric"]


def test_refusal_suites_are_canned_and_parameterized(mini_pack):
    benign = generate_refusal_suite(TaskType.REFUSAL_BENIGN, mini_pack, 42, 12)
    refuse = generate_refusal_suite(TaskType.SHOULD_REFUSE, mini_pack, 42, 15)
    assert len(benign) == 12 and len(refuse) == 15
    for item in benign + refuse:
        assert "refusal_suite" in item["tags"]
        assert mini_pack.name in item["prompt"]  # parameterized by pack name
    benign_prompts = {i["prompt"] for i in benign}
    assert len(benign_prompts) == 12, "suite prompts must be distinct"
    assert benign_prompts.isdisjoint({i["prompt"] for i in refuse})


def test_procedural_variety_survives_default_dedup(mini_pack):
    """30 items from one template family must not collapse under the 0.90 filter."""
    spec = _spec(mini_pack, "txn_classify")
    items = generate_procedural_items(spec, mini_pack, 42, "eval", 30)
    records = [
        EvalRecord(
            id=f"ev-{i:06d}", domain=spec.name, prompt=it["prompt"],
            task_type=TaskType.EXACT, gold=it["gold"],
        )
        for i, it in enumerate(items, 1)
    ]
    kept, report = filter_near_duplicates(records, [], threshold=0.90, backend="fuzzy")
    assert len(kept) == 30, f"procedural items collapsed: {report.dropped}"


def test_tool_call_variety_survives_default_dedup(mini_pack):
    spec = _spec(mini_pack, "ops_tools")
    items = generate_procedural_items(spec, mini_pack, 42, "eval", 20)
    records = [
        EvalRecord(
            id=f"ev-{i:06d}", domain=spec.name, prompt=it["prompt"],
            task_type=TaskType.TOOL_CALL, tools=spec.tools, expected_tool=it["expected_tool"],
        )
        for i, it in enumerate(items, 1)
    ]
    kept, report = filter_near_duplicates(records, [], threshold=0.90, backend="fuzzy")
    assert len(kept) == 20, f"tool-call items collapsed: {report.dropped}"


# ---------------------------------------------------------------------------
# synthdoc
# ---------------------------------------------------------------------------


def test_long_document_reaches_16k_tokens():
    import random

    doc = build_long_document(random.Random("t"), min_tokens=16_000)
    assert estimate_tokens(doc) >= 16_000


def test_with_long_document_wraps_and_strips():
    import random

    core = "Classify the payment to Blue Harbor LLC as described in workpaper Q4417."
    wrapped = with_long_document(core, random.Random("t"), topic="reviews")
    assert wrapped.startswith(DOC_BEGIN)
    assert DOC_END in wrapped
    assert estimate_tokens(wrapped) >= 16_000
    stripped = comparable_text(wrapped)
    assert "blue harbor llc" in stripped
    assert DOC_BEGIN.lower() not in stripped
    assert len(stripped) < 400, "comparable text must drop the embedded document"
    # deterministic under the same rng seed
    assert wrapped == with_long_document(core, random.Random("t"), topic="reviews")
