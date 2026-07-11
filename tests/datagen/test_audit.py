"""Audit sampling (PRD M1): stratification, minimum size, determinism, empty sets."""

from __future__ import annotations

from reaplab.core.records import EvalRecord, TaskType
from reaplab.datagen.audit import write_audit_sample


def _records(domain: str, n: int, start: int) -> list[EvalRecord]:
    return [
        EvalRecord(
            id=f"ev-{start + i:06d}",
            domain=domain,
            prompt=f"Prompt number {start + i} for {domain}: review the {domain} scenario case {i}.",
            task_type=TaskType.OPEN_ENDED,
            rubric="Score 0 to 1.",
        )
        for i in range(n)
    ]


def test_stratified_sample_hits_target_and_every_domain(tmp_path):
    records = _records("alpha", 20, 1) + _records("beta", 15, 100) + _records("gamma", 5, 200)
    path = write_audit_sample(records, tmp_path / "audit.md", seed=42)
    text = path.read_text(encoding="utf-8")
    assert text.count("### ev-") == 10  # max(10, round(0.05 * 40))
    for domain in ("alpha", "beta", "gamma"):
        assert f"## {domain}" in text, f"domain {domain} missing from the sample"


def test_small_sets_are_sampled_entirely(tmp_path):
    records = _records("alpha", 8, 1)
    path = write_audit_sample(records, tmp_path / "audit.md", seed=42)
    assert path.read_text(encoding="utf-8").count("### ev-") == 8


def test_deterministic_for_a_seed(tmp_path):
    records = _records("alpha", 30, 1) + _records("beta", 30, 100)
    a = write_audit_sample(records, tmp_path / "a.md", seed=42).read_bytes()
    b = write_audit_sample(records, tmp_path / "b.md", seed=42).read_bytes()
    assert a == b
    c = write_audit_sample(records, tmp_path / "c.md", seed=43).read_bytes()
    assert a != c


def test_empty_eval_set_still_writes_a_marked_file(tmp_path):
    path = write_audit_sample([], tmp_path / "audit.md", seed=42)
    text = path.read_text(encoding="utf-8")
    assert "No eval records" in text


def test_long_prompts_are_truncated_for_review(tmp_path):
    long_rec = EvalRecord(
        id="ev-000001",
        domain="alpha",
        prompt="word " * 2000,
        task_type=TaskType.OPEN_ENDED,
    )
    path = write_audit_sample([long_rec], tmp_path / "audit.md", seed=42)
    text = path.read_text(encoding="utf-8")
    assert "truncated for review" in text
    assert len(text) < 5000
