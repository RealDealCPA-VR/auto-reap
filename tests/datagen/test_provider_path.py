"""Provider-text generation path: canned JSON parsing, malformed-response
resilience, per-entry coercion, retry budget, and refusal-suite top-up."""

from __future__ import annotations

import json
import logging

from reaplab.core.config import ProviderCfg
from reaplab.core.hashing import canonical_json
from reaplab.core.jsonl import read_jsonl
from reaplab.core.paths import Workspace
from reaplab.core.providers.mock import MockProvider
from reaplab.core.records import CalibrationRecord, EvalRecord
from reaplab.datagen import generate_datasets
from reaplab.datagen.provider_gen import generate_domain_via_provider

PROVIDER_PACK = {
    "name": "provider-pack",
    "description": "Single-domain pack for provider-path tests.",
    "include_refusal_suites": False,
    "domains": [
        {
            "name": "txn_classify",
            "description": "Categorize transactions.",
            "task_type": "exact",
            "weight": 1.0,
        }
    ],
}

CAL_ITEMS = [
    {"prompt": "Categorize the wire transfer received from Meridian Holdings in April.", "tags": ["wire"], "difficulty": "easy"},
    {"prompt": "Which account fits the recurring Adobe subscription renewal?", "tags": [], "difficulty": "medium"},
    {"prompt": "Sort the fuel purchase at the Tacoma station into the chart of accounts.", "tags": ["fuel"], "difficulty": "medium"},
    {"prompt": "Where should the annual liability insurance premium be posted?", "tags": [], "difficulty": "hard"},
]

EVAL_ITEMS = [
    {"prompt": "Classify: $840 payment to Staples for printer toner and paper. Options: Office Supplies, Travel.", "gold": "Office Supplies", "tags": [], "difficulty": "easy"},
    {"prompt": "Classify: $2,100 airfare to the industry conference in Boise. Options: Travel, Meals.", "gold": "Travel", "tags": [], "difficulty": "medium"},
    {"prompt": "Classify: monthly Shopify platform fee for the online store. Options: Software Subscriptions, Rent.", "gold": "Software Subscriptions", "tags": [], "difficulty": "medium"},
    {"prompt": "Classify: quarterly premium paid to Hartford for general liability. Options: Insurance, Payroll.", "gold": "Insurance", "tags": [], "difficulty": "hard"},
]


def _provider(responses: dict[str, str], kind: str = "anthropic-api") -> MockProvider:
    """MockProvider posing as a non-mock kind so the pipeline takes the provider
    path (kind == 'mock' would trigger procedural generation instead)."""
    return MockProvider(ProviderCfg(kind=kind, extra={"responses": responses}))


def test_canned_json_batches_flow_into_datasets(make_spec, tmp_path):
    spec = make_spec(PROVIDER_PACK, calibration_size=4, eval_size=4)
    provider = _provider(
        {
            "BATCH-TAG: calibration/txn_classify/0": json.dumps(CAL_ITEMS),
            "BATCH-TAG: eval/txn_classify/0": json.dumps(EVAL_ITEMS),
        }
    )
    ws = Workspace(tmp_path / "ws").ensure()
    cal_path, eval_path = generate_datasets(spec, ws, provider=provider)

    cal = read_jsonl(cal_path, CalibrationRecord)
    ev = read_jsonl(eval_path, EvalRecord)
    assert [r.prompt for r in cal] == [i["prompt"] for i in CAL_ITEMS]
    assert [r.prompt for r in ev] == [i["prompt"] for i in EVAL_ITEMS]
    assert [r.gold for r in ev] == [i["gold"] for i in EVAL_ITEMS]
    assert all(r.source == "synthetic-anthropic-api" for r in cal + ev)
    assert [r.id for r in cal] == [f"cal-{i:06d}" for i in range(1, 5)]
    assert [r.id for r in ev] == [f"ev-{i:06d}" for i in range(1, 5)]


def test_markdown_fenced_json_is_tolerated(make_spec, tmp_path):
    spec = make_spec(PROVIDER_PACK, calibration_size=2, eval_size=2)
    fenced = "Here you go!\n```json\n" + json.dumps(EVAL_ITEMS[:2]) + "\n```"
    provider = _provider(
        {
            "BATCH-TAG: calibration/txn_classify/0": json.dumps(CAL_ITEMS[:2]),
            "BATCH-TAG: eval/txn_classify/0": fenced,
        }
    )
    ws = Workspace(tmp_path / "ws").ensure()
    _, eval_path = generate_datasets(spec, ws, provider=provider)
    assert len(read_jsonl(eval_path, EvalRecord)) == 2


def test_malformed_batches_are_skipped_with_warning(make_spec, tmp_path, caplog):
    spec = make_spec(PROVIDER_PACK, calibration_size=2, eval_size=2)
    provider = _provider(
        {
            "BATCH-TAG: calibration/txn_classify/0": json.dumps(CAL_ITEMS[:2]),
            "BATCH-TAG: eval/txn_classify": "this is not json at all, sorry",
        }
    )
    ws = Workspace(tmp_path / "ws").ensure()
    with caplog.at_level(logging.WARNING, logger="reaplab.datagen"):
        cal_path, eval_path = generate_datasets(spec, ws, provider=provider)
    assert len(read_jsonl(cal_path, CalibrationRecord)) == 2  # cal unaffected
    assert read_jsonl(eval_path, EvalRecord) == []  # bad batches skipped, run completes
    assert "retrying once" in caplog.text
    assert "skipping this batch" in caplog.text


def test_partially_malformed_array_keeps_valid_entries(make_spec, tmp_path, caplog):
    bad_array = [EVAL_ITEMS[0], {"tags": ["no-prompt-here"]}, EVAL_ITEMS[1]]
    spec = make_spec(PROVIDER_PACK, calibration_size=2, eval_size=4)
    provider = _provider(
        {
            "BATCH-TAG: calibration/txn_classify/0": json.dumps(CAL_ITEMS[:2]),
            "BATCH-TAG: eval/txn_classify/0": json.dumps(bad_array),
        }
    )
    ws = Workspace(tmp_path / "ws").ensure()
    with caplog.at_level(logging.WARNING, logger="reaplab.datagen"):
        _, eval_path = generate_datasets(spec, ws, provider=provider)
    ev = read_jsonl(eval_path, EvalRecord)
    assert [r.prompt for r in ev] == [EVAL_ITEMS[0]["prompt"], EVAL_ITEMS[1]["prompt"]]
    assert "skipped 1 malformed item" in caplog.text
    assert "generated 2 of 4" in caplog.text  # under-delivery is reported, not fatal


def test_refusal_suites_topped_up_from_templates_when_provider_fails(make_spec, tmp_path, caplog):
    pack = dict(PROVIDER_PACK, include_refusal_suites=True)
    spec = make_spec(pack, calibration_size=2, eval_size=4)
    provider = _provider(
        {
            "BATCH-TAG: calibration/txn_classify/0": json.dumps(CAL_ITEMS[:2]),
            "BATCH-TAG: eval/txn_classify/0": json.dumps(EVAL_ITEMS),
            # no canned responses for the suites -> provider fails -> canned fallback
        }
    )
    ws = Workspace(tmp_path / "ws").ensure()
    with caplog.at_level(logging.WARNING, logger="reaplab.datagen"):
        _, eval_path = generate_datasets(spec, ws, provider=provider)
    ev = read_jsonl(eval_path, EvalRecord)
    domains = {r.domain: 0 for r in ev}
    for r in ev:
        domains[r.domain] += 1
    assert domains["benign_sensitive"] == 10  # max(10, 5% of 4)
    assert domains["should_refuse"] == 15
    assert "topping up" in caplog.text


class _CountingProvider(MockProvider):
    calls = 0

    def complete(self, *args, **kwargs):
        type(self).calls += 1
        return super().complete(*args, **kwargs)


def test_retry_budget_two_strikes_per_domain(mini_pack):
    """A dead endpoint gets: (initial + 1 retry) x 2 zero-yield batches, then gives up."""
    _CountingProvider.calls = 0
    provider = _CountingProvider(ProviderCfg(kind="openai-compat"))  # default reply: not an array
    spec = next(d for d in mini_pack.domains if d.name == "advisory")
    items = generate_domain_via_provider(spec, mini_pack, provider, "eval", 5)
    assert items == []
    assert _CountingProvider.calls == 4


def test_tool_call_coercion_validates_expected_tool(mini_pack):
    spec = next(d for d in mini_pack.domains if d.name == "ops_tools")
    entries = [
        {"prompt": "Do something vague.", "expected_tool": "not_a_real_tool"},  # skipped
        {"prompt": "Also vague, missing the tool."},  # skipped (multi-tool domain)
        {"prompt": "Pull the May statement for account 1010 at First Bank.", "expected_tool": "fetch_statement"},
    ]
    provider = _provider({"BATCH-TAG: eval/ops_tools/0": json.dumps(entries)})
    items = generate_domain_via_provider(spec, mini_pack, provider, "eval", 1)
    assert len(items) == 1
    assert items[0]["expected_tool"] == "fetch_statement"
    assert items[0]["tools"] == spec.tools


def test_json_schema_coercion_validates_gold(mini_pack):
    spec = next(d for d in mini_pack.domains if d.name == "report_extract")
    good = {"entity": "Cascade Group", "period": "2024-Q2", "lines": [{"account": "Rent", "amount": 1200.0}]}
    entries = [
        {"prompt": "Extract from a report missing required fields.", "gold": {"entity": "X"}},  # invalid
        {"prompt": "Extract the Q2 summary for Cascade Group.", "gold": json.dumps(good)},  # gold-as-string ok
    ]
    provider = _provider({"BATCH-TAG: eval/report_extract/0": json.dumps(entries)})
    items = generate_domain_via_provider(spec, mini_pack, provider, "eval", 1)
    assert len(items) == 1
    assert items[0]["gold"] == canonical_json(good)
    assert items[0]["json_schema"] == spec.json_schema
