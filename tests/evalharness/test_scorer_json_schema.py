from __future__ import annotations

import json

from reaplab.core.records import TaskType
from reaplab.evalharness.scorers.json_schema_scorer import JsonSchemaScorer

scorer = JsonSchemaScorer()

SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "amount": {"type": "number"},
        "memo": {"type": "string"},
    },
    "required": ["vendor", "amount"],
}
GOLD = json.dumps({"vendor": "Staples", "amount": 43.75})


def _item(make_record, **kw):
    defaults = dict(task_type=TaskType.JSON_SCHEMA, json_schema=SCHEMA, gold=GOLD)
    defaults.update(kw)
    return make_record(**defaults)


def test_valid_and_matching_full_credit(make_record, rresp):
    item = _item(make_record)
    score, passed, detail = scorer.score(item, rresp(GOLD))
    assert (score, passed) == (1.0, True)
    assert detail["schema_valid"] and detail["gold_match"]


def test_fenced_json_is_extracted(make_record, rresp):
    item = _item(make_record)
    text = f"Here you go:\n```json\n{GOLD}\n```\nLet me know if that helps."
    score, passed, _ = scorer.score(item, rresp(text))
    assert (score, passed) == (1.0, True)


def test_optional_field_difference_still_full_credit(make_record, rresp):
    item = _item(make_record)
    resp = json.dumps({"vendor": "Staples", "amount": 43.75, "memo": "extra optional field"})
    score, passed, _ = scorer.score(item, rresp(resp))
    assert (score, passed) == (1.0, True)


def test_valid_but_wrong_value_half_credit(make_record, rresp):
    item = _item(make_record)
    resp = json.dumps({"vendor": "Office Depot", "amount": 43.75})
    score, passed, detail = scorer.score(item, rresp(resp))
    assert (score, passed) == (0.5, False)
    assert detail["schema_valid"] is True
    assert detail["mismatched_fields"] == ["vendor"]


def test_schema_invalid_zero(make_record, rresp):
    item = _item(make_record)
    resp = json.dumps({"vendor": "Staples"})  # missing required "amount"
    score, passed, detail = scorer.score(item, rresp(resp))
    assert (score, passed) == (0.0, False)
    assert detail["schema_valid"] is False


def test_wrong_type_invalid(make_record, rresp):
    item = _item(make_record)
    resp = json.dumps({"vendor": "Staples", "amount": "forty-three"})
    score, passed, detail = scorer.score(item, rresp(resp))
    assert (score, passed) == (0.0, False)
    assert detail["schema_valid"] is False


def test_no_json_at_all_zero(make_record, rresp):
    item = _item(make_record)
    score, passed, detail = scorer.score(item, rresp("I would say the vendor is Staples."))
    assert (score, passed) == (0.0, False)
    assert detail["schema_valid"] is False


def test_valid_without_gold_passes_flagged(make_record, rresp):
    item = _item(make_record, gold=None)
    score, passed, detail = scorer.score(item, rresp(GOLD))
    assert (score, passed) == (1.0, True)
    assert detail["no_gold"] is True


def test_unparseable_gold_scores_validity_only_and_is_flagged(make_record, rresp):
    """Broken gold is bad DATA. Awarding full credit would hide it behind a perfect
    score; half credit (schema validity only) keeps it visible in the report."""
    item = _item(make_record, gold="the vendor should be Staples")
    score, passed, detail = scorer.score(item, rresp(GOLD))
    assert (score, passed) == (0.5, False)
    assert detail["schema_valid"] is True
    assert detail["gold_unparseable"] is True
    assert "no_gold" not in detail
    assert "not valid JSON" in detail["error"] and item.id in detail["error"]


def test_unparseable_gold_does_not_rescue_an_invalid_response(make_record, rresp):
    item = _item(make_record, gold="not json at all")
    resp = json.dumps({"vendor": "Staples"})  # missing required "amount"
    score, passed, detail = scorer.score(item, rresp(resp))
    assert (score, passed) == (0.0, False)
    assert detail["schema_valid"] is False


def test_missing_schema_is_dataset_error(make_record, rresp):
    item = _item(make_record, json_schema=None)
    score, passed, detail = scorer.score(item, rresp(GOLD))
    assert (score, passed) == (0.0, False)
    assert "json_schema" in detail["error"]
