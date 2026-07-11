"""Planning: proportional counts, refusal suites, long-context shares, pack validation."""

from __future__ import annotations

import copy

import pytest

from reaplab.core.config import DataCfg, DomainPack
from reaplab.core.records import TaskType
from reaplab.datagen.planning import (
    BENIGN_SENSITIVE_DOMAIN,
    SHOULD_REFUSE_COUNT,
    SHOULD_REFUSE_DOMAIN,
    benign_suite_size,
    largest_remainder,
    plan_counts,
)


def test_counts_proportional_to_weights(mini_pack):
    plan = plan_counts(mini_pack, DataCfg(calibration_size=20, eval_size=10))
    cal = {a.spec.name: a.cal_count for a in plan.allocations if not a.suite}
    ev = {a.spec.name: a.eval_count for a in plan.allocations if not a.suite}
    # weights 3/2/2/2/1 over totals 20 and 10 divide exactly
    assert cal == {
        "txn_classify": 6, "report_extract": 4, "advisory": 4, "ops_tools": 4, "long_review": 2,
    }
    assert ev == {
        "txn_classify": 3, "report_extract": 2, "advisory": 2, "ops_tools": 2, "long_review": 1,
    }
    assert sum(cal.values()) == 20
    assert sum(ev.values()) == 10


def test_largest_remainder_is_exact_and_deterministic():
    counts = largest_remainder(10, {"a": 1.0, "b": 1.0, "c": 1.0})
    assert counts == {"a": 4, "b": 3, "c": 3}  # tie-break by insertion order
    assert sum(largest_remainder(37, {"x": 3.1, "y": 2.2, "z": 0.7}).values()) == 37
    assert largest_remainder(0, {"a": 1.0}) == {"a": 0}


def test_refusal_suites_added_to_eval_only(mini_pack):
    plan = plan_counts(mini_pack, DataCfg(calibration_size=20, eval_size=10))
    benign = plan.allocation(BENIGN_SENSITIVE_DOMAIN)
    should = plan.allocation(SHOULD_REFUSE_DOMAIN)
    assert benign.suite and should.suite
    assert benign.spec.task_type == TaskType.REFUSAL_BENIGN
    assert should.spec.task_type == TaskType.SHOULD_REFUSE
    assert benign.cal_count == 0 and should.cal_count == 0
    assert benign.eval_count == benign_suite_size(10) == 10  # max(10, 5%) floor
    assert should.eval_count == SHOULD_REFUSE_COUNT == 15
    # suites are additive on top of eval_size
    assert plan.eval_total == 10 + 10 + 15


def test_benign_suite_scales_with_eval_size():
    assert benign_suite_size(300) == 15
    assert benign_suite_size(2000) == 100
    assert benign_suite_size(40) == 10  # floor


def test_suites_omitted_when_pack_disables_them(mini_pack_dict, make_pack_file):
    mini_pack_dict["include_refusal_suites"] = False
    pack = DomainPack.from_yaml(make_pack_file(mini_pack_dict))
    plan = plan_counts(pack, DataCfg(calibration_size=20, eval_size=10))
    names = [a.spec.name for a in plan.allocations]
    assert BENIGN_SENSITIVE_DOMAIN not in names
    assert SHOULD_REFUSE_DOMAIN not in names


def test_long_context_share(mini_pack):
    plan = plan_counts(
        mini_pack, DataCfg(calibration_size=20, eval_size=10, long_context_share=0.5)
    )
    lr = plan.allocation("long_review")
    assert lr.long_context_cal == 1  # ceil(0.5 * 2)
    assert lr.long_context_eval == 1  # ceil(0.5 * 1) -> min 1 for non-empty long domains
    assert plan.allocation("advisory").long_context_eval == 0  # not flagged long_context


def test_tool_call_domain_without_tools_fails_fast(mini_pack_dict, make_pack_file):
    bad = copy.deepcopy(mini_pack_dict)
    bad["domains"][3].pop("tools")
    pack = DomainPack.from_yaml(make_pack_file(bad))
    with pytest.raises(ValueError, match="tools"):
        plan_counts(pack, DataCfg())


def test_json_schema_domain_without_schema_fails_fast(mini_pack_dict, make_pack_file):
    bad = copy.deepcopy(mini_pack_dict)
    bad["domains"][1].pop("json_schema")
    pack = DomainPack.from_yaml(make_pack_file(bad))
    with pytest.raises(ValueError, match="json_schema"):
        plan_counts(pack, DataCfg())


def test_reserved_suite_names_rejected(mini_pack_dict, make_pack_file):
    bad = copy.deepcopy(mini_pack_dict)
    bad["domains"][2]["name"] = "should_refuse"
    pack = DomainPack.from_yaml(make_pack_file(bad))
    with pytest.raises(ValueError, match="reserved"):
        plan_counts(pack, DataCfg())
