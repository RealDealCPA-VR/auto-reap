from __future__ import annotations

import json

from reaplab.core.records import TaskType
from reaplab.evalharness.scorers.tool_call import ToolCallScorer, normalize_tool_calls, synth_args

scorer = ToolCallScorer()

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
            "required": ["city"],
        },
    },
}
LEDGER_TOOL = {
    "type": "function",
    "function": {
        "name": "post_journal_entry",
        "parameters": {
            "type": "object",
            "properties": {"account": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["account", "amount"],
        },
    },
}
TOOLS = [WEATHER_TOOL, LEDGER_TOOL]


def _item(make_record, **kw):
    defaults = dict(task_type=TaskType.TOOL_CALL, tools=TOOLS, expected_tool="get_weather")
    defaults.update(kw)
    return make_record(**defaults)


def _openai_call(name: str, args: dict) -> list[dict]:
    return [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]


def test_correct_tool_valid_args(make_record, rresp):
    item = _item(make_record)
    score, passed, detail = scorer.score(item, rresp(tool_calls=_openai_call("get_weather", {"city": "Reno"})))
    assert (score, passed) == (1.0, True)
    assert detail["schema_valid"] is True


def test_wrong_tool_but_schema_valid_half_credit(make_record, rresp):
    item = _item(make_record)
    call = _openai_call("post_journal_entry", {"account": "6100 Utilities", "amount": 120.5})
    score, passed, detail = scorer.score(item, rresp(tool_calls=call))
    assert (score, passed) == (0.5, False)
    assert detail["schema_valid"] is True  # feeds the tool_call_validity gate anyway
    assert detail["called_tool"] == "post_journal_entry"


def test_invalid_args_zero(make_record, rresp):
    item = _item(make_record)
    score, passed, detail = scorer.score(item, rresp(tool_calls=_openai_call("get_weather", {"units": "C"})))
    assert (score, passed) == (0.0, False)
    assert detail["schema_valid"] is False


def test_unknown_tool_zero(make_record, rresp):
    item = _item(make_record)
    score, passed, detail = scorer.score(item, rresp(tool_calls=_openai_call("format_disk", {})))
    assert (score, passed) == (0.0, False)
    assert detail["schema_valid"] is False


def test_text_fallback_json_call(make_record, rresp):
    item = _item(make_record)
    text = json.dumps({"name": "get_weather", "arguments": {"city": "Reno"}})
    score, passed, detail = scorer.score(item, rresp(text))
    assert (score, passed) == (1.0, True)
    assert detail["schema_valid"] is True


def test_text_fallback_fenced(make_record, rresp):
    item = _item(make_record)
    text = '```json\n{"name": "get_weather", "arguments": {"city": "Reno", "units": "F"}}\n```'
    score, passed, _ = scorer.score(item, rresp(text))
    assert (score, passed) == (1.0, True)


def test_no_call_at_all_zero(make_record, rresp):
    item = _item(make_record)
    score, passed, detail = scorer.score(item, rresp("The weather in Reno is probably sunny."))
    assert (score, passed) == (0.0, False)
    assert detail["schema_valid"] is False


def test_no_expected_tool_any_valid_listed_tool_passes(make_record, rresp):
    item = _item(make_record, expected_tool=None)
    call = _openai_call("post_journal_entry", {"account": "6100", "amount": 10})
    score, passed, _ = scorer.score(item, rresp(tool_calls=call))
    assert (score, passed) == (1.0, True)


def test_multiple_calls_best_one_counts(make_record, rresp):
    item = _item(make_record)
    calls = _openai_call("format_disk", {}) + _openai_call("get_weather", {"city": "Reno"})
    score, passed, _ = scorer.score(item, rresp(tool_calls=calls))
    assert (score, passed) == (1.0, True)


def test_missing_tools_is_dataset_error(make_record, rresp):
    item = _item(make_record, tools=None)
    score, passed, detail = scorer.score(item, rresp("anything"))
    assert (score, passed) == (0.0, False)
    assert "tools" in detail["error"]


def test_normalize_handles_string_and_dict_arguments(rresp):
    r = rresp(tool_calls=[{"function": {"name": "t", "arguments": '{"a": 1}'}}])
    assert normalize_tool_calls(r) == [{"name": "t", "arguments": {"a": 1}}]
    r2 = rresp(tool_calls=[{"name": "t", "arguments": {"a": 1}}])  # flat, pre-parsed
    assert normalize_tool_calls(r2) == [{"name": "t", "arguments": {"a": 1}}]


def test_synth_args_satisfies_schema():
    import jsonschema

    for tool in TOOLS:
        params = tool["function"]["parameters"]
        jsonschema.validate(synth_args(params), params)
    nested = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["a", "b"]},
            "rows": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
            "flag": {"type": "boolean"},
        },
        "required": ["kind", "rows"],
    }
    jsonschema.validate(synth_args(nested), nested)
    assert synth_args(nested)["kind"] == "a"  # enum -> first value, deterministic
