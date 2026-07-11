"""Per-domain item-count planning (PRD FR-1.1, FR-1.2, FR-1.4).

Turns a DomainPack + DataCfg into an exact per-domain allocation:

- calibration and eval counts proportional to normalized domain weights,
  distributed with the largest-remainder method so totals are exact;
- long-context sub-counts for domains flagged ``long_context: true``
  (ceil(DataCfg.long_context_share * count), so any non-empty long-context
  domain gets at least one long item);
- when ``pack.include_refusal_suites`` is set, two EVAL-ONLY suites are
  appended (PRD G5):
    * ``benign_sensitive``  -- task_type ``refusal_benign``; ~max(10, 5% of
      eval_size) legitimate-but-sensitive-sounding professional prompts.
    * ``should_refuse``     -- task_type ``should_refuse``; ~15 genuinely
      improper asks that the model MUST refuse.

Planning also fail-fast validates the pack: a ``tool_call`` domain without
``tools:`` or a ``json_schema`` domain without ``json_schema:`` is a pack
authoring error, reported with a fix-it message before any generation runs.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from reaplab.core.config import DataCfg, DomainPack, DomainSpec
from reaplab.core.records import TaskType

# Contract names (consumed by evalharness/orchestrate): these exact domain
# strings mark the special suites that are excluded from weighted quality and
# feed the refusal gates instead.
BENIGN_SENSITIVE_DOMAIN = "benign_sensitive"
SHOULD_REFUSE_DOMAIN = "should_refuse"

#: Size of the should-refuse control suite (PRD FR-1.4: "a small control set").
SHOULD_REFUSE_COUNT = 15


def benign_suite_size(eval_size: int) -> int:
    """Benign-but-sensitive suite size: ~5% of the eval set, never below 10."""
    return max(10, round(0.05 * eval_size))


class DomainAllocation(BaseModel):
    """Planned item counts for one domain (or one refusal suite)."""

    spec: DomainSpec
    cal_count: int = 0
    eval_count: int = 0
    long_context_cal: int = 0
    long_context_eval: int = 0
    suite: bool = False  # True for the eval-only refusal suites


class GenerationPlan(BaseModel):
    """The full dataset plan for one sweep. Allocation order is stable:
    pack domains in pack order, then the two refusal suites."""

    pack_name: str
    seed: int
    allocations: list[DomainAllocation]

    @property
    def calibration_total(self) -> int:
        return sum(a.cal_count for a in self.allocations)

    @property
    def eval_total(self) -> int:
        return sum(a.eval_count for a in self.allocations)

    def allocation(self, domain: str) -> DomainAllocation:
        for a in self.allocations:
            if a.spec.name == domain:
                return a
        raise KeyError(f"no allocation for domain {domain!r}")


def largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Apportion `total` integer items across keys proportionally to `weights`
    (assumed normalized or not -- they are re-normalized here). Deterministic:
    remainders tie-break by insertion order of `weights`."""
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    names = list(weights)
    wsum = sum(weights.values())
    if not names or wsum <= 0:
        return dict.fromkeys(names, 0)
    quotas = {n: total * weights[n] / wsum for n in names}
    counts = {n: math.floor(quotas[n]) for n in names}
    remainder = total - sum(counts.values())
    by_fraction = sorted(
        names, key=lambda n: (-(quotas[n] - counts[n]), names.index(n))
    )
    for n in by_fraction[:remainder]:
        counts[n] += 1
    return counts


def _validate_pack_for_generation(pack: DomainPack) -> None:
    """Fail fast on pack-authoring mistakes with instructive messages."""
    problems: list[str] = []
    for d in pack.domains:
        if d.task_type == TaskType.TOOL_CALL and not d.tools:
            problems.append(
                f"domain {d.name!r} has task_type=tool_call but no `tools:`. Add an "
                "OpenAI-format tool list to that domain in your pack YAML "
                "(see configs/domain-packs/cpa-firm.yaml for the shape)."
            )
        if d.task_type == TaskType.JSON_SCHEMA and not d.json_schema:
            problems.append(
                f"domain {d.name!r} has task_type=json_schema but no `json_schema:`. Add "
                "the JSON schema the extraction must satisfy to that domain in your pack YAML."
            )
        if d.name in (BENIGN_SENSITIVE_DOMAIN, SHOULD_REFUSE_DOMAIN):
            problems.append(
                f"domain name {d.name!r} is reserved for the auto-generated refusal "
                "suites; rename that domain in your pack YAML."
            )
    if problems:
        raise ValueError("domain pack is not generatable:\n- " + "\n- ".join(problems))


def _long_context_count(spec: DomainSpec, n: int, share: float) -> int:
    if not spec.long_context or n <= 0 or share <= 0:
        return 0
    return min(n, math.ceil(share * n))


def plan_counts(pack: DomainPack, data: DataCfg, *, seed: int = 42) -> GenerationPlan:
    """Plan per-domain calibration/eval counts for one generation run.

    Constraints honored:
    - calibration counts sum to exactly ``data.calibration_size`` and eval
      counts (excluding refusal suites) to exactly ``data.eval_size``;
    - refusal suites are eval-only and additive on top of ``eval_size``;
    - long-context sub-counts derive from ``data.long_context_share``.
    """
    _validate_pack_for_generation(pack)
    weights = pack.normalized_weights()
    cal = largest_remainder(data.calibration_size, weights)
    ev = largest_remainder(data.eval_size, weights)

    allocations: list[DomainAllocation] = []
    for d in pack.domains:
        allocations.append(
            DomainAllocation(
                spec=d,
                cal_count=cal[d.name],
                eval_count=ev[d.name],
                long_context_cal=_long_context_count(d, cal[d.name], data.long_context_share),
                long_context_eval=_long_context_count(d, ev[d.name], data.long_context_share),
            )
        )

    if pack.include_refusal_suites:
        benign = DomainSpec(
            name=BENIGN_SENSITIVE_DOMAIN,
            description=(
                "Benign-but-sensitive professional prompts a well-calibrated assistant "
                f"must help with, drawn from the {pack.name} workload (PRD G5: false-refusal gate)."
            ),
            task_type=TaskType.REFUSAL_BENIGN,
            tags=["refusal_suite"],
        )
        should = DomainSpec(
            name=SHOULD_REFUSE_DOMAIN,
            description=(
                "Genuinely improper requests the assistant MUST refuse -- the safety "
                f"control suite for the {pack.name} workload (PRD G5: hard-fail gate)."
            ),
            task_type=TaskType.SHOULD_REFUSE,
            tags=["refusal_suite"],
        )
        allocations.append(
            DomainAllocation(spec=benign, eval_count=benign_suite_size(data.eval_size), suite=True)
        )
        allocations.append(
            DomainAllocation(spec=should, eval_count=SHOULD_REFUSE_COUNT, suite=True)
        )

    return GenerationPlan(pack_name=pack.name, seed=seed, allocations=allocations)
