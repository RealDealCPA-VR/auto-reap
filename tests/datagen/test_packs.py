"""The shipped example packs load, plan, and generate procedurally."""

from __future__ import annotations

import json

import jsonschema
import pytest

from reaplab.core.config import DataCfg, DomainPack
from reaplab.core.records import TaskType
from reaplab.datagen.planning import benign_suite_size, plan_counts
from reaplab.datagen.procedural import generate_procedural_items

PACK_FILES = ["cpa-firm.yaml", "coding-agent.yaml", "general-assistant.yaml"]


@pytest.fixture(params=PACK_FILES)
def pack(request, packs_dir) -> DomainPack:
    return DomainPack.from_yaml(packs_dir / request.param)


def test_pack_loads_and_plans(pack):
    data = DataCfg(calibration_size=100, eval_size=40)
    plan = plan_counts(pack, data)
    assert plan.calibration_total == 100
    assert plan.eval_total == 40 + benign_suite_size(40) + 15
    for alloc in plan.allocations:
        if alloc.spec.task_type == TaskType.TOOL_CALL:
            assert alloc.spec.tools, f"{alloc.spec.name} must declare tools"
        if alloc.spec.task_type == TaskType.JSON_SCHEMA:
            assert alloc.spec.json_schema, f"{alloc.spec.name} must declare a schema"


def test_pack_has_long_context_coverage(pack):
    assert any(d.long_context for d in pack.domains), (
        "every shipped pack must exercise 16k+ contexts (PRD FR-1.4)"
    )


def test_pack_generates_procedurally(pack):
    """Every domain in every shipped pack must be procedurally generatable,
    with scoreable fields per task type."""
    for spec in pack.domains:
        items = generate_procedural_items(spec, pack, 42, "eval", 3)
        assert len(items) == 3
        for item in items:
            assert item["prompt"].strip()
            if spec.task_type == TaskType.EXACT:
                assert item["gold"]
            elif spec.task_type == TaskType.JSON_SCHEMA:
                jsonschema.validate(json.loads(item["gold"]), spec.json_schema)
            elif spec.task_type == TaskType.TOOL_CALL:
                names = {t["function"]["name"] for t in spec.tools}
                assert item["expected_tool"] in names
            elif spec.task_type == TaskType.OPEN_ENDED:
                assert item["rubric"]


def test_new_packs_have_distinct_names(packs_dir):
    names = [DomainPack.from_yaml(packs_dir / f).name for f in PACK_FILES]
    assert len(set(names)) == len(names)
