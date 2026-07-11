from __future__ import annotations

import pytest

from reaplab.core.jsonl import append_jsonl, iter_jsonl, read_jsonl, write_jsonl
from reaplab.core.records import CalibrationRecord, EvalRecord, TaskType


def _cal(i: int) -> CalibrationRecord:
    return CalibrationRecord(id=f"cal-{i:06d}", domain="qbo_categorization", prompt=f"categorize tx {i}")


def test_roundtrip(tmp_path):
    path = tmp_path / "cal.jsonl"
    n = write_jsonl(path, [_cal(i) for i in range(5)])
    assert n == 5
    back = read_jsonl(path, CalibrationRecord)
    assert [r.id for r in back] == [f"cal-{i:06d}" for i in range(5)]
    assert back[0].schema_version == "1.0"


def test_append_and_iter(tmp_path):
    path = tmp_path / "results.jsonl"
    append_jsonl(path, _cal(1))
    append_jsonl(path, _cal(2))
    assert [r.id for r in iter_jsonl(path, CalibrationRecord)] == ["cal-000001", "cal-000002"]


def test_invalid_line_reports_location(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        _cal(1).model_dump_json() + "\n" + '{"id": "x"}' + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_jsonl(path, CalibrationRecord)


def test_eval_record_task_types(tmp_path):
    rec = EvalRecord(
        id="ev-000001",
        domain="fs_extraction",
        prompt="extract this",
        task_type=TaskType.JSON_SCHEMA,
        json_schema={"type": "object"},
    )
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, [rec])
    back = read_jsonl(path, EvalRecord)[0]
    assert back.task_type is TaskType.JSON_SCHEMA
    assert back.gold is None
