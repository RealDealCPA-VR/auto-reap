"""PRD §8: the base model must permit modification (pruning IS modification) and
local commercial use. reap-lab can't adjudicate a license — it must make sure the
user was told to look."""

from __future__ import annotations

import pytest

from reaplab.cli.doctor import _check_license
from reaplab.core.config import SweepSpec


def _spec(model_id: str) -> SweepSpec:
    return SweepSpec(model_id=model_id, domain_pack="pack.yaml")


@pytest.mark.parametrize(
    "model_id",
    ["Qwen/Qwen3-30B-A3B", "mistralai/Mixtral-8x7B-Instruct-v0.1", "zai-org/GLM-4.5-Air"],
)
def test_known_permissive_models_pass_with_a_verify_reminder(model_id):
    name, level, message = _check_license(_spec(model_id))
    assert name == "base model license"
    assert level == "OK"
    assert "Verify" in message  # never asserts the license is settled


def test_unknown_model_warns_that_pruning_is_modification():
    _, level, message = _check_license(_spec("some-lab/Mystery-MoE-42B"))
    assert level == "WARN"
    assert "MODIFICATION" in message
    assert "Mystery-MoE-42B" in message


def test_without_a_spec_it_still_reminds_the_user():
    _, level, message = _check_license(None)
    assert level == "WARN"
    assert "MODIFICATION" in message
