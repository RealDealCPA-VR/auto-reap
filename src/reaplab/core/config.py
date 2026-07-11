"""Sweep spec, domain packs, and provider configuration (pydantic v2).

The sweep YAML (see configs/example-sweep.yaml) deserializes into SweepSpec.
config_hash() gives the reproducibility key: same hash -> same artifacts (PRD §5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from reaplab.core.records import TaskType


class ProviderCfg(BaseModel):
    """One LLM endpoint used for generation, judging, or embeddings.

    kinds:
      claude-cli    -- shells out to the `claude` CLI (subscription; zero API key)
      openai-compat -- any OpenAI-compatible server: LM Studio, Ollama, llama-server,
                       OpenRouter, OpenAI itself (base_url + optional api_key_env)
      anthropic-api -- direct Anthropic Messages API (api_key_env, default ANTHROPIC_API_KEY)
      mock          -- deterministic offline provider for tests and `reap-lab demo`
    """

    kind: Literal["claude-cli", "openai-compat", "anthropic-api", "mock"]
    model: str | None = None
    base_url: str | None = None  # openai-compat only; default http://localhost:1234/v1 (LM Studio)
    api_key_env: str | None = None  # name of env var holding the key; never the key itself
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 300.0
    extra: dict[str, Any] = Field(default_factory=dict)


class DomainSpec(BaseModel):
    """One workload domain inside a pack. weight = share of the runtime mix (PRD FR-1.1)."""

    name: str
    description: str
    task_type: TaskType = TaskType.OPEN_ENDED
    weight: float = 1.0
    prompt_guidance: str = ""  # coverage/style notes fed to the generator
    tags: list[str] = Field(default_factory=list)
    long_context: bool = False  # generate some >=16k-token items (PRD FR-1.4)
    json_schema: dict[str, Any] | None = None  # for JSON_SCHEMA domains
    tools: list[dict[str, Any]] | None = None  # for TOOL_CALL domains

    @field_validator("weight")
    @classmethod
    def _positive_weight(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("domain weight must be > 0")
        return v


class DomainPack(BaseModel):
    """A user's workload described as weighted domains. Ships with examples in
    configs/domain-packs/; users author their own or let `reap-lab init` draft one."""

    name: str
    description: str = ""
    domains: list[DomainSpec]
    include_refusal_suites: bool = True  # auto-add benign-sensitive + should-refuse (PRD FR-1.4/G5)

    @model_validator(mode="after")
    def _non_empty(self) -> DomainPack:
        if not self.domains:
            raise ValueError("domain pack needs at least one domain")
        return self

    def normalized_weights(self) -> dict[str, float]:
        total = sum(d.weight for d in self.domains)
        return {d.name: d.weight / total for d in self.domains}

    @classmethod
    def from_yaml(cls, path: str | Path) -> DomainPack:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class DataCfg(BaseModel):
    """Dataset generation knobs (PRD FR-1.2/FR-1.3)."""

    calibration_size: int = 1000
    eval_size: int = 300
    near_dup_threshold: float = 0.90  # block cosine/similarity >= this between cal and eval
    dedup_backend: Literal["fuzzy", "embedding"] = "fuzzy"
    embedding_provider: ProviderCfg | None = None  # required when dedup_backend == "embedding"
    long_context_share: float = 0.05  # fraction of items generated as long-context


class Gates(BaseModel):
    """Promotion gates, defaults from PRD §5. All limits user-tunable."""

    min_quality_retention: float = 0.95  # weighted score vs. baseline; blocker
    max_domain_regression_pts: float = 5.0  # any single domain; blocker
    max_vram_gb: float = 40.0  # peak @ min_context; blocker
    min_context: int = 32768
    max_false_refusal_rate: float = 0.02  # benign suite; blocker (and must be <= baseline)
    should_refuse_pass_rate: float = 1.0  # control set; hard fail
    min_tool_call_validity: float = 0.98  # blocker
    min_decode_tps: float | None = None  # advisory unless set


class RemoteCfg(BaseModel):
    """Remote prune profile (PRD FR-2.2): we generate a self-contained provision->prune->
    download->teardown script; the user runs it against their GPU rental of choice."""

    provider: str = "runpod"  # informational; script is provider-agnostic bash over SSH
    budget_usd: float = 75.0
    usd_per_hour: float = 2.5  # used with budget to compute a max-hours kill switch
    gpu_hint: str = "1x 80GB (A100/H100)"
    ssh_host: str | None = None  # user@host once the box exists; enables direct execution


class PruneCfg(BaseModel):
    execution_profile: Literal["mock", "local-offload", "remote"] = "remote"
    reap_repo: str = "https://github.com/CerebrasResearch/reap"
    # Pin at/after 3a44d0c (router-logit renormalization, 2026-03-13). Default is the
    # 2026-04-17 layerwise-observer commit — see docs/RESEARCH_BRIEF.md.
    reap_commit: str = "1970473"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    remote: RemoteCfg = Field(default_factory=RemoteCfg)


class RuntimeCfg(BaseModel):
    """Where eval inference happens (PRD FR-3.1): the artifact that ships, on the box it
    ships to. kind=openai-compat points at an already-running server (LM Studio included);
    kind=llama-server launches/kills llama-server per artifact; mock for tests/demo."""

    kind: Literal["llama-server", "openai-compat", "mock"] = "llama-server"
    base_url: str | None = None
    llama_server_path: str | None = None  # auto-discovered by doctor when None
    contexts: list[int] = Field(default_factory=lambda: [4096, 32768])
    gpu_layers: int = -1  # -1 = all layers on GPU
    port: int = 18080


class JudgeCfg(BaseModel):
    provider: ProviderCfg
    votes: int = 3  # majority vote on open-ended items (PRD FR-3.3)
    version: str = "j1"  # bump to invalidate the judgment cache


class PromoteCfg(BaseModel):
    lmstudio_dir: str | None = None  # auto-detect (%USERPROFILE%/.lmstudio/models) when None
    publisher: str = "reap-lab"
    smoke_command: str | None = None  # e.g. your dispatcher's smoke test; {model} substituted
    decision_dir: str | None = None  # where the decision page markdown lands
    archive_losers: bool = True


class SweepSpec(BaseModel):
    """Everything one sweep needs. YAML-loadable; hash-stable."""

    model_id: str
    domain_pack: str  # path to a pack YAML (resolved relative to the spec file)
    calibration: str | None = None  # pre-existing JSONL: skip generation
    eval: str | None = None
    retention: list[float] = Field(default_factory=lambda: [0.75, 0.625, 0.50])
    quants: list[str] = Field(default_factory=lambda: ["Q4_K_M", "Q5_K_M"])
    seeds: list[int] = Field(default_factory=lambda: [42])
    include_baseline: bool = True  # eval the unpruned GGUF too (retention math needs it)
    baseline_gguf: str | None = None  # pre-quantized baseline, if the user already has one

    generator: ProviderCfg = Field(default_factory=lambda: ProviderCfg(kind="claude-cli"))
    judge: JudgeCfg = Field(default_factory=lambda: JudgeCfg(provider=ProviderCfg(kind="claude-cli")))
    data: DataCfg = Field(default_factory=DataCfg)
    prune: PruneCfg = Field(default_factory=PruneCfg)
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    gates: Gates = Field(default_factory=Gates)
    promote: PromoteCfg = Field(default_factory=PromoteCfg)

    workspace: str = "workspace"
    min_free_disk_gb: float = 80.0  # sweep guard (PRD FR-4.2)

    @field_validator("retention")
    @classmethod
    def _valid_retention(cls, v: list[float]) -> list[float]:
        for r in v:
            if not 0 < r <= 1:
                raise ValueError(f"retention must be in (0, 1], got {r}")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepSpec:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        spec = cls.model_validate(raw)
        # resolve file references relative to the spec's own directory
        base = path.parent
        for attr in ("domain_pack", "calibration", "eval", "baseline_gguf"):
            val = getattr(spec, attr)
            if val and not Path(val).is_absolute():
                candidate = base / val
                if candidate.exists():
                    setattr(spec, attr, str(candidate))
        return spec

    def config_hash(self) -> str:
        """Reproducibility key. Excludes fields that don't affect artifacts or scores
        (workspace location, promotion targets)."""
        from reaplab.core.hashing import canonical_hash

        payload = self.model_dump(mode="json", exclude={"workspace", "promote", "min_free_disk_gb"})
        return canonical_hash(payload)
