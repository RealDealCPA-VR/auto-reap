# reap-lab — Action Plan

**Source:** `PRD_Automated_REAP_Pipeline.md` (Draft v0.1, 2026-07-11)
**Reframe:** The PRD describes a firm-specific pipeline. This build generalizes it so **any user can point it at any MoE model + their own workload** and get a pruned, evaluated, ranked, promotable local model. The CPA-firm workload ships as one *domain pack* among several examples — not as hardcoded behavior.

---

## 1. PRD Analysis — what matters and what changes

### What the PRD gets right (kept as-is)
- **Evaluate the artifact that ships** (GGUF at target quant), never only the HF checkpoint — pruning × quantization losses compound (FR-3.1).
- **Calibration is prompts-only** — REAP observes router gates/activations on forward passes; no gold outputs needed. Cheap to generate, no output-reuse ToS questions (FR-1.2).
- **Promotion gates as hard blockers** (§5): ≥95% weighted retention, ≤5-pt per-domain regression, VRAM ceiling, ≥98% tool-call validity, should-refuse control at 100%.
- **Buy-vs-build check first**: eval a Cerebras pre-pruned public checkpoint before paying for custom prune runs (§9 note).
- **Config-hash discipline** end to end; resumable sweeps; failure isolation.

### What generalizing for "any user" requires (the deltas)
| PRD assumption | Generalized design |
|---|---|
| Fixed firm domains (QBO, IRS notices…) | **Domain packs**: YAML files describing a workload mix. Ships with `cpa-firm`, `coding-agent`, `general-assistant` examples. `reap-lab init` wizard drafts a custom pack from a plain-English description of the user's workload. |
| Claude Agent SDK as the frontier model | **Pluggable providers**: `claude-cli` (subscription, zero-key), `anthropic-api`, `openai-compat` (works with LM Studio, Ollama, llama-server, OpenRouter, OpenAI), `mock` (offline). Same interface for generation *and* judging. |
| vr-dispatch / hermes-brain integration | Generic hooks: promotion runs a configurable smoke-test command; decision page is written to a configurable directory. |
| RTX 6000 Ada 48 GB | VRAM ceiling, context target, and LM Studio directory are all config values with auto-detection (`nvidia-smi`, standard install paths). |
| Firm PDF redaction tool | **Dropped, not generalized.** There is no seed-material ingestion path at all: every calibration and eval item is *generated* — procedurally in mock mode, otherwise by the frontier provider from the domain pack's descriptions and `prompt_guidance`. No user document, PDF, log or transcript ever enters the pipeline, so there is nothing to redact and no PII-redaction hook exists. (`spec.calibration` / `spec.eval` can point at pre-existing JSONL *datasets* you built yourself — those are your files, validated and re-emitted as-is, never scraped from source material.) |

### Key feasibility notes
- Actual REAP pruning of a 30B MoE needs ~60 GB+ (bf16) → the prune step runs on a **remote 80 GB GPU** (generated, budget-capped provisioning script: provision → prune → package → download; destroying the instance stays the user's explicit step, since a guest-side shutdown bills on and kills the shell you need to fetch the tarball) or `local-offload` if the user has the RAM+patience — Linux/WSL only, since reap's vllm/torch pins have no Windows wheels. The lab machine never needs to hold the bf16 model.
- Everything else — data generation, GGUF conversion of a downloaded pruned checkpoint, evaluation, reporting, promotion — runs locally.
- The pipeline is **fully testable offline**: a `mock` provider, a mock pruner, and a mock model runner let `reap-lab demo` execute the entire flow (data → "prune" → "convert" → eval → report → promote) in seconds with zero GPU/network. This is the primary validation vehicle in this build; real GPU runs are the user's M2+.

---

## 2. Stack (decided)

| Concern | Choice | Why |
|---|---|---|
| Language / env | Python ≥3.11, **uv-locked** single package `reap-lab` (import `reaplab`) | PRD NFR; matches local tooling; one `uv run reap-lab …` for everything |
| CLI | **Typer + Rich** | Wizard prompts, progress, pretty tables; near-zero boilerplate |
| Config & schemas | **pydantic v2** + PyYAML | Validated sweep specs / domain packs / JSONL records with real error messages |
| Job state | **SQLite (stdlib)** + append-only JSONL results | Resumable sweeps, no server, matches PRD FR-4.1 |
| HTTP | httpx | Sync client for OpenAI-compat servers; easy to mock (respx) |
| Near-dup filter | **rapidfuzz** default; optional embedding backend via any OpenAI-compat `/v1/embeddings` (e.g. LM Studio) | Zero heavy deps by default; PRD's cosine ≥0.90 embedding filter available when a local embedder exists |
| JSON scoring | jsonschema | Structured-task and tool-call validation |
| Perf capture | psutil + `nvidia-smi` subprocess + llama-server timings | Prefill/decode tok/s, peak VRAM, load time |
| Pruning | Subprocess wrapper around pinned **CerebrasResearch/reap** in its own env (remote profile default); never imported into this package | Research-grade dep stays isolated; version pin in config |
| GGUF | llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` wrappers | Deployed artifact is GGUF (LM Studio) |
| Eval runtime | Any **OpenAI-compatible server** (llama-server primary; LM Studio works too) | Evaluate what ships; user probably already runs one |
| Tests | pytest (+respx), everything green **offline** | CI-able; no GPU in the loop |

Heavy/optional deps (`huggingface_hub`) live in an optional extra so the base install stays light.

## 3. Architecture

```
src/reaplab/
  core/          # contracts everyone builds against (WRITTEN FIRST, inline)
    config.py    # SweepSpec, DomainPack, ProviderCfg, Gates, RemoteCfg (pydantic)
    records.py   # CalibrationRecord, EvalRecord, ItemResult, ArtifactManifest, PerfMetrics
    providers/   # LLMProvider base + claude_cli / openai_compat / anthropic_api / mock
    hashing.py   # canonical config_hash, streamed artifact_hash
    state.py     # SQLite job state: claim/complete/fail/manual/resume
    jsonl.py     # schema-validated JSONL read/write/append
    paths.py     # workspace layout: runs/<hash>/{data,manifests,logs}, artifacts/<hash>/,
                 #   prune/ (remote scripts + tarballs), reports/, cache/judge/<hash>/, archive/
  datagen/       # C1: pack-driven generation, dedup/leakage filter, refusal suites, audit sampling
  prune/         # C2: reap runner (local-offload/remote/mock), remote provisioning script-gen, GGUF convert+quant
  evalharness/   # C3: openai-compat runner, scorers (exact/json_schema/tool_call/refusal/judge), perf, judge cache
  orchestrate/   # C4: sweep engine, guards (disk/budget), report (ranked+Pareto+gates), promote
  cli/           # typer app: init doctor generate audit prune convert eval sweep report promote status demo
configs/         # example-sweep.yaml + domain-packs/{cpa-firm,coding-agent,general-assistant}.yaml
tests/           # mirrors src; e2e/ runs the full mock demo
docs/            # QUICKSTART, ARCHITECTURE, DOMAIN_PACKS, REMOTE_GPU, this plan
```

**Flow (`reap-lab sweep sweep.yaml`):** load spec → hash config → generate/load datasets (C1) → for each retention: prune (C2) → for each quant: convert (C2) → eval + perf (C3) → score/aggregate → report + winner (C4) → optional promote. Every stage records state in SQLite; re-running the same spec skips completed stages (resume). One bad config marks `failed` and the sweep continues.

## 4. Comprehensive Todo List

### Phase 0 — Scaffold & contracts (inline, before the swarm)
- [ ] T0.1 git init, .gitignore, pyproject (all deps pre-declared), uv lock
- [ ] T0.2 `core/` contracts: config, records, providers, hashing, state, jsonl, paths + core tests
- [ ] T0.3 Example configs: sweep YAML + 3 domain packs
- [ ] T0.4 Research brief: REAP repo interface, llama.cpp conversion, LM Studio conventions, remote GPU options (background agent)

### Phase 1 — Component build (parallel swarm, disjoint ownership)
- [ ] T1.1 **C1 datagen**: pack loader, prompt generation via provider (batched, mix-proportional), eval-set generation with gold/rubrics, near-dup + leakage filter, benign-sensitive + should-refuse suites, long-context items, audit sampling, JSONL emit — + tests
- [ ] T1.2 **C2 prune**: reap config → command generation, execution profiles (`mock`, `local-offload`, `remote` script-gen with budget cap), artifact download/verify, per-run manifest, GGUF convert + quant grid wrappers — + tests
- [ ] T1.3 **C3 evalharness**: openai-compat runner (llama-server/LM Studio), scorer registry (exact/normalized, json-schema, tool-call validity, refusal classifier, pairwise LLM judge n=3 majority with cache keyed `(item_id, artifact_hash, judge_version)`), perf capture at 4k/32k, determinism (temp 0, seeds) — + tests
- [ ] T1.4 **C4 orchestrate**: sweep engine over retention×quant grid, resume from SQLite, failure isolation, disk-space guard, weighted scoring + per-domain breakdown, gates from §5, markdown report (ranked table, Pareto, regression diff, anomalies), promote (LM Studio dir, decision page, archive, smoke hook) — + tests
- [ ] T1.5 **Docs draft**: QUICKSTART, ARCHITECTURE, DOMAIN_PACKS, REMOTE_GPU

### Phase 2 — Integration (single agent + me)
- [ ] T2.1 CLI: all commands wired to components; `init` wizard (workload description → drafted domain pack via provider); `doctor` (checks llama.cpp, providers, VRAM, disk, LM Studio dir)
- [ ] T2.2 `demo` mode: full pipeline end-to-end with mock provider/pruner/runner; produces a real report
- [ ] T2.3 Full test suite green; e2e demo test

### Phase 3 — Validation swarm
- [ ] T3.1 Full pytest run + `demo` smoke on the real CLI
- [ ] T3.2 Multi-lens adversarial review (correctness, PRD-compliance, UX-for-new-user, Windows-specifics, security/data-governance) → verify → fix
- [ ] T3.3 Re-run suite + demo after fixes; PRD requirements traceability check (every FR mapped to code+test or explicitly deferred)

### Phase 4 — Ship prep
- [ ] T4.1 Banger README (quickstart in <5 commands, architecture diagram, gates table, screenshots of report output)
- [ ] T4.2 Logical commits on master; **no push** until user provides the GitHub location

### Explicitly deferred (per PRD non-goals)
LoRA/SFT healing, serving changes, UI beyond CLI+markdown, algorithm modifications, automated retrain triggers.

## 5. Validation criteria for this build
1. `uv run pytest` fully green, offline, on Windows.
2. `uv run reap-lab demo` executes data→prune→convert→eval→report→promote with mocks and emits a ranked report with gates.
3. `reap-lab doctor` and `--help` for every command work.
4. Review swarm findings triaged to zero confirmed-open.
5. Traceability: each PRD FR → implementation + test (or documented deferral).
