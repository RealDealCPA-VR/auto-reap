from __future__ import annotations

import pytest

from reaplab.core.records import TaskType
from reaplab.evalharness.scorers.exact import ExactScorer, as_measure, as_number, normalize

scorer = ExactScorer()


def test_plain_match(make_record, rresp):
    item = make_record(task_type=TaskType.EXACT, gold="Office Supplies")
    score, passed, _ = scorer.score(item, rresp("Office Supplies"))
    assert (score, passed) == (1.0, True)


@pytest.mark.parametrize(
    "response",
    ["  office   supplies.", "OFFICE SUPPLIES", "'Office Supplies'", "office\tsupplies!"],
)
def test_normalization_case_whitespace_punct(make_record, rresp, response):
    item = make_record(task_type=TaskType.EXACT, gold="Office Supplies")
    score, passed, _ = scorer.score(item, rresp(response))
    assert (score, passed) == (1.0, True)


@pytest.mark.parametrize(
    ("response", "gold"),
    [
        ("1,000.00", "1000"),
        ("$1,000.00", "1000"),
        ("1000", "1,000"),
        ("42.0", "42"),
        ("-3.50", "-3.5"),
    ],
)
def test_numeric_tolerance_commas_currency(make_record, rresp, response, gold):
    item = make_record(task_type=TaskType.EXACT, gold=gold)
    score, passed, detail = scorer.score(item, rresp(response))
    assert (score, passed) == (1.0, True), detail


def test_numeric_mismatch_fails(make_record, rresp):
    item = make_record(task_type=TaskType.EXACT, gold="1000")
    score, passed, detail = scorer.score(item, rresp("1001"))
    assert (score, passed) == (0.0, False)
    assert detail["numeric_match"] is False


def test_text_mismatch_fails(make_record, rresp):
    item = make_record(task_type=TaskType.EXACT, gold="Office Supplies")
    score, passed, detail = scorer.score(item, rresp("Meals & Entertainment"))
    assert (score, passed) == (0.0, False)
    assert detail["normalized_match"] is False


def test_gold_none_is_dataset_error(make_record, rresp):
    item = make_record(task_type=TaskType.EXACT, gold=None)
    score, passed, detail = scorer.score(item, rresp("anything"))
    assert (score, passed) == (0.0, False)
    assert "gold" in detail["error"]


def test_normalize_and_as_number_helpers():
    assert normalize("  Hello,   WORLD!  ") == "hello, world"
    assert as_number("$1,234.50") == 1234.5
    assert as_number("not a number") is None
    assert as_number("1,00,0") == 1000.0  # commas stripped wherever they sit


# -- unit classes: separators are noise, currency/percent markers are meaning ------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1000", (1000.0, None)),
        ("1,000.00", (1000.0, None)),
        ("$10", (10.0, "currency")),
        ("10 USD", (10.0, "currency")),
        ("10%", (10.0, "percent")),
        ("10 percent", (10.0, "percent")),
        ("$10%", (10.0, "mixed")),
        ("office supplies", None),
    ],
)
def test_as_measure_reports_the_unit_class(text, expected):
    assert as_measure(text) == expected


@pytest.mark.parametrize(("response", "gold"), [("$10", "10%"), ("10%", "$10"), ("$0.07", "7%")])
def test_currency_vs_percent_is_not_a_numeric_match(make_record, rresp, response, gold):
    item = make_record(task_type=TaskType.EXACT, gold=gold)
    score, passed, detail = scorer.score(item, rresp(response))
    assert (score, passed) == (0.0, False)
    assert detail["unit_mismatch"] is True
    assert detail["got_unit"] != detail["want_unit"]


@pytest.mark.parametrize(
    ("response", "gold"),
    [
        ("1,000.00", "1000"),  # separators still don't matter
        ("$1,000.00", "1000"),  # an UNMARKED gold accepts either marker
        ("10", "10%"),
        ("7.5%", "7.50%"),  # same class, same value
        ("$1,200", "1200 USD"),
    ],
)
def test_compatible_units_still_match(make_record, rresp, response, gold):
    item = make_record(task_type=TaskType.EXACT, gold=gold)
    score, passed, detail = scorer.score(item, rresp(response))
    assert (score, passed) == (1.0, True), detail
