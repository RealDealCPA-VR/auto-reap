# Architecture — how reap-lab is put together

Audience: developers who want to understand, extend, or debug the pipeline. For the guided
tour, start with [QUICKSTART.md](QUICKSTART.md); for workload configuration, see
[DOMAIN_PACKS.md](DOMAIN_PACKS.md); for the remote prune step, see
[REMOTE_GPU.md](REMOTE_GPU.md).

## The one-paragraph version

A sweep spec (YAML → `SweepSpec`) fully determines a run. Its hash names a run directory;
every stage records completion in SQLite under that hash, so re-running the same spec resumes
instead of repeating. Four components do the work — **C1 datagen** builds calibration/eval
datasets from your domain pack, **C2 prune** wraps REAP and llama.cpp conversion, **C3
evalharness** scores every GGUF on a real OpenAI-compatible runtime, **C4 orchestrate**
drives the grid and renders the report — all built against the contracts in
`src/reaplab/core/`.

## Flow

```mermaid
flowchart LR
    SPEC[sweep.yaml\nSweepSpec + config_hash] --> C1
    PACK[domain pack YAML\nDomainPack] --> C1
    subgraph C1 [C1 datagen]
        GEN[frontier provider\ngenerates prompts + gold/rubrics] --> DEDUP[near-dup + leakage filter\ncal vs eval]
        DEDUP --> CAL[calibration_v1.jsonl\nprompts only]
        DEDUP --> EV[eval_v1.jsonl\ngold + rubrics + refusal suites]
    end
    CAL --> C2P
    subgraph C2 [C2 prune]
        C2P[REAP runner\nmock / local-offload / remote 80GB] --> HF[pruned HF checkpoints\nx retention grid]
        HF --> CONV[convert_hf_to_gguf + llama-quantize\nx quant grid]
    end
    CONV --> GGUF[GGUF artifacts + manifests]
    GGUF --> C3R
    EV --> C3R
    subgraph C3 [C3 evalharness]
        C3R[openai-compat runner\nllama-server / LM Studio] --> SCORE[scorers: exact / json-schema /\ntool-call / refusal / LLM judge n=3]
        C3R --> PERF[perf capture\ntok/s + peak VRAM @ 4k/32k]
    end
    SCORE --> STATE[(state.db + results.jsonl)]
    PERF --> STATE
    subgraph C4 [C4 orchestrate]
        STATE --> REPORT[ranked report\nPareto + gates + regressions]
        REPORT --> PROMOTE[promote winner\nLM Studio dir + smoke + decision page]
    end
```

## Components

| Component | Package | Job | Key externals |
|---|---|---|---|
| core | `src/reaplab/core/` | shared contracts: config models, record schemas, providers, hashing, SQLite state, JSONL IO, workspace paths | — |
| C1 datagen | `src/reaplab/datagen/` | pack-driven prompt/eval generation, weight-proportional mix, near-dup + leakage filtering, refusal suites, long-context items, audit sampling | frontier provider (claude-cli / openai-compat / anthropic-api / mock) |
| C2 prune | `src/reaplab/prune/` | REAP config → command generation; execution profiles `mock` / `local-offload` (Linux/WSL only) / `remote` (script-gen, budget-capped, ssh-drivable); artifact download/verify; per-artifact manifest; GGUF convert + quant grid | `CerebrasResearch/reap` (pinned commit, **always a subprocess or generated script — never imported**), llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` |
| C3 evalharness | `src/reaplab/evalharness/` | runs eval items against an OpenAI-compatible server; scorer registry; refusal classification; pairwise LLM judge (n=3 majority, cached); perf capture | llama-server / LM Studio; judge provider; `nvidia-smi` |
| C4 orchestrate | `src/reaplab/orchestrate/` | sweep engine over retention × quant, resume, failure isolation, disk guard, weighted scoring, gates, markdown report, promotion | LM Studio models dir; optional smoke command |
| CLI | `src/reaplab/cli/` | `reap-lab demo init doctor generate audit sweep report promote prune convert eval status version` (`prune` needs `--retention`; `convert` builds only the unpruned **baseline** GGUFs — pruned candidates are converted by `prune`/`sweep`; `eval` needs `--gguf`) | — |

Two architectural rules keep the install light and the pipeline testable:

1. **Heavy work is always a subprocess.** REAP (torch/transformers/vllm) and llama.cpp are
   never imported into this package — they are invoked as external commands or via generated
   scripts, each in its own pinned environment. `reap-lab` itself needs only pydantic, typer,
   rich, httpx, pyyaml, rapidfuzz, jsonschema, and psutil.
2. **Every external has a mock.** A mock provider, mock pruner, and mock runtime let
   `reap-lab demo` and the entire test suite execute the full pipeline offline in seconds.

## Workspace layout — where every file lands

Everything a sweep touches lives under one root (`workspace:` in the sweep YAML, default
`workspace/`), managed by `reaplab.core.paths.Workspace`. **Datasets and artifacts are
namespaced by config hash**, so two specs sharing one workspace can never read or clobber each
other's work:

```
workspace/
  runs/<config_hash>/           one directory per unique sweep config
    data/                       THIS sweep's datasets (nothing is shared between sweeps)
      calibration_v1.jsonl        CalibrationRecord lines — prompts only
      eval_v1.jsonl               EvalRecord lines — gold/rubrics/schemas/tools + refusal suites
      dedup_report_v1.json        what the near-dup/leakage filter dropped, and why
      eval_v1_audit_sample.md     stratified ~5% human-audit sample (`reap-lab audit` prints it)
      calibration_dataset/        data.jsonl in REAP's `messages` column format (uploaded to the GPU box)
    manifests/<artifact_id>.json  one ArtifactManifest per built artifact
    state.db                    resumable job state (SQLite; stages + metrics)
    results.jsonl               per-item scoring results (ItemResult, append-only)
    logs/                       subprocess logs (prune, convert, llama-server, remote ssh/scp)
  artifacts/<config_hash>/      everything built for that config
    <slug>-r<ret>-hf/             pruned HF checkpoint (from REAP)
    <slug>-r<ret>-bf16.gguf       intermediate, shared across the quant grid
    <slug>-r<ret>-<quant>.gguf    the candidates
    baseline/<slug>-baseline-<quant>.gguf   unpruned baselines (+ their bf16 intermediate)
  prune/                        REAP scratch, NOT per-config: the generated remote script
                                (prune_remote_r<ret>.sh), its REMOTE_STEPS_r<ret>.md, the
                                downloaded pruned_r<ret>.tar.gz, extraction scratch, and the
                                local-offload reap clone — see REMOTE_GPU.md
  reports/                      sweep-<config_hash>.md + decision-<artifact>-<hash>.md
  cache/judge/<config_hash>/    LLM-judge cache, keyed sha256(item_id, artifact_hash, judge_version)
  archive/                      evaluated non-winners, MOVED here on promote (never deleted)
  data/                         legacy/unused: created for compatibility, datasets no longer live here
```

`<slug>` is the last path component of the HF model id (`Qwen/Qwen3-30B-A3B` → `Qwen3-30B-A3B`).

Datasets use the versioned JSONL schemas in `core/records.py` (`CalibrationRecord`,
`EvalRecord`; every record carries `schema_version`). All JSONL IO goes through
`core/jsonl.py`, which validates each line against its pydantic model and reports the exact
file/line on failure.

## Artifact naming

Artifact ids are short, stable, and used everywhere — file names, state keys, results,
reports:

- **Baseline (unpruned) GGUF:** `baseline-<quant lowercase>` → `baseline-q4_k_m`
- **Pruned GGUF:** `r<retention>-<quant lowercase>`, retention formatted with `:g` (no
  trailing zeros) → `r0.5-q4_k_m`, `r0.625-q5_k_m`, `r0.75-q4_k_m`

Each artifact gets an `ArtifactManifest` (`core/records.py`), written to
`runs/<config_hash>/manifests/<artifact_id>.json`. What it actually records:

| Field | What it holds |
|---|---|
| `model_id`, `retention`, `quant`, `path`, `kind` | which artifact this is |
| `config_hash` | the config that produced it — a manifest from another hash is never reused |
| `artifact_hash` | streamed content hash of the GGUF (also the judge-cache key component) |
| `reap_commit` | the pinned REAP commit the prune ran at |
| `saliency_stats` | expert counts read back from the pruned checkpoint's `config.json`: `num_experts_after`, which config field it came from, `num_experts_per_tok`, `model_type`. This is the post-prune evidence you can verify locally — it is how you confirm the checkpoint you downloaded really has the experts you paid to remove. `null` for mock builds, or when the architecture exposes no known expert-count field (a wrong number is worse than none). |
| `wall_clock_s`, `peak_mem_gb` | prune + convert + quant time; host RSS peak *when this machine ran the prune*. On the remote profile `peak_mem_gb` is null and `versions["peak_mem_gb"]` explains why — an absent number is always explained, never silently null. |
| `versions` | `reap_commit`, `execution_profile`, and the resolved `convert_hf_to_gguf` / `llama_quantize` paths (or `gguf_tools: mock`) |

**`retained_expert_map` is deliberately always `null`.** The field exists in the schema, but
REAP's saved `config.json` carries expert *counts*, not the kept-index map, and reap-lab never
loads the weights to reconstruct one. Recording a guessed map would be worse than recording
none; `saliency_stats` is what you actually get.

## Reproducibility: config hash → resume

`SweepSpec.config_hash()` canonically hashes the spec (sorted-key JSON, sha256, 12 hex chars).
Two properties are deliberate, and they are what make the workspace safe to reuse.

**1. Referenced files are hashed by CONTENT, not by path.** `domain_pack` hashes the pack's
*parsed* form; `calibration`, `eval` and `baseline_gguf` hash their raw bytes. So:

- Edit your pack's domains/weights → new hash → a **fresh run directory with fresh datasets**.
  You can never silently prune against a stale calibration set generated from an older pack.
- Reformat the pack, add comments, re-indent → **same hash** (the parsed content is identical),
  so you keep every dataset, artifact and score you already paid for.
- Move the pack to a different directory, or run the same spec from another checkout → same
  hash, clean resume.

**2. Fields that change neither artifacts nor measurements are excluded:**

| Excluded | Why |
|---|---|
| `workspace` | where results are stored is not what was measured |
| `promote.*` | promotion targets don't affect any number |
| `min_free_disk_gb` | a guard, not an input |
| `gates.*` | gates **re-rank existing measurements**. Tuning a threshold must not orphan a $45 sweep — edit `gates:` and re-run `reap-lab report`; it re-applies them to the stored summaries and re-picks the winner, with no new work. |
| `runtime.port`, `runtime.llama_server_path` | local plumbing. The port a server binds to, and where the binary lives, cannot change a score. (`runtime.kind`, `contexts`, `gpu_layers`, `base_url`, `model` **are** hashed — they can.) |

The contract: **same config hash → same datasets, artifacts, and scores.** Anything material
(model, retention grid, quants, seeds, pack content, generator, judge, data sizes, prune
profile, runtime shape) mints a new run directory rather than silently mixing results.

Resume works through `core/state.py` (`StateDB`, SQLite at `runs/<config_hash>/state.db`).
Each unit of work is one `(stage, key)` row with status `running` / `done` / `failed` /
`manual`. The stage names are a shared contract:

| Stage key | Unit of work | Written by |
|---|---|---|
| `datagen` | generate + filter both datasets | datagen (and the sweep, under key `datasets`) |
| `prune:r<retention:g>` | one REAP run, e.g. `prune:r0.5` | prune |
| `convert:<artifact_id>` | one GGUF conversion+quant, e.g. `convert:r0.5-q4_k_m` | prune |
| `eval:<artifact_id>` | one full evaluation, e.g. `eval:baseline-q4_k_m` | orchestrate |
| `sweep:<key>` | the sweep's *own* coarse failures (disk guard, build error) | orchestrate |

The `sweep:` namespace exists so a coarse failure above a component can never clobber that
component's fine-grained progress rows.

Re-running `reap-lab sweep` with the same spec checks `is_done` per stage and skips completed
work (`--no-resume` forces a redo). A *failing* stage is recorded with its error and the sweep
moves on (failure isolation, PRD FR-4.2) — one bad config never kills the overnight run. A
remote prune with no `ssh_host` is marked **`manual`**, not `failed`: nothing is broken, the
sweep is waiting on you, and the report prints the instructions verbatim under "Manual steps
pending". `reap-lab status` reads this table; `reap-lab report` and `reap-lab promote` rebuild
the whole report/winner from it without running any stage.

Guards before/during the sweep: free-disk check (`min_free_disk_gb`, default 80 — each
candidate weighs 15–35 GB), eager quant-name validation (a `q4km` typo must cost a second, not
a completed $75 remote prune), and the remote-prune budget cap
([REMOTE_GPU.md](REMOTE_GPU.md)).

## Evaluate the GGUF, not the HF checkpoint

The pipeline's central measurement principle (PRD FR-3.1): **the artifact that gets scored is
the artifact that ships** — the quantized GGUF, served by the same llama.cpp-family runtime
you deploy on.

Why it matters: pruning and quantization losses *compound*, and not linearly. A pruned HF
checkpoint that benchmarks fine in bf16 can degrade disproportionately at Q4_K_M — the
quantizer has less redundancy to hide behind once experts are gone — and runtime differences
(chat template handling, sampling, tokenizer edge cases) shift scores further. Scoring the
bf16 checkpoint would systematically overestimate what you're about to deploy. So C3 always
loads the final GGUF into llama-server (or your already-running LM Studio) and measures
quality *and* performance (prefill/decode tok/s, peak VRAM, load time at 4k and 32k
contexts) there.

There are exactly **three** runners (`evalharness/runners.py`), and none of them loads an HF
checkpoint:

| Runner | `runtime.kind` | What it does |
|---|---|---|
| `LlamaServerRunner` | `llama-server` (default) | launches `llama-server` per artifact/context, waits for health, records load time and (via `nvidia-smi` polling) peak VRAM, tears down Windows-safely |
| `OpenAICompatRunner` | `openai-compat` | uses an **already-running** server (LM Studio, Ollama, your own llama-server). `start()` only verifies it answers — it launches nothing, so it cannot report load time or peak VRAM |
| `MockRunner` | `mock` | deterministic offline model whose quality degrades with pruning; what `demo` and the tests run on |

There is **no transformers/vLLM runner** for sanity-checking the HF checkpoint before
conversion, by design: nothing here imports torch. If you want to poke a pruned checkpoint in
bf16, do it yourself, outside reap-lab.

## The eval summary contract

C3's `evaluate_artifact` returns one summary dict per artifact; C4 consumes it for scoring,
gates, and the report:

```python
{
  "artifact_id": str,                       # e.g. "r0.5-q4_k_m"
  "domain_scores": {domain: float},         # mean 0..1 per domain
  "counts": {domain: int},                  # items per domain
  "false_refusal_rate": float | None,       # from the benign_sensitive suite
  "should_refuse_pass_rate": float | None,  # from the should_refuse suite
  "tool_call_validity": float | None,       # across tool_call items
  "perf": {str(context): PerfMetrics.model_dump()},  # "4096", "32768"
  "items_scored": int,
  "responses": {item_id: str},              # this artifact's raw answers
}
```

`responses` is what makes the pairwise judge possible: the orchestrator feeds the
**baseline's** responses back in as `baseline_responses` when evaluating each candidate at the
same quant, and open-ended items are then judged head-to-head against them. Without a
matching-quant baseline (e.g. a standalone `reap-lab eval --gguf` on a downloaded GGUF), the
open-ended scorer falls back to a non-refusal heuristic and the report says so in its Notes.

Two **special domains** — `benign_sensitive` (task_type `refusal_benign`) and
`should_refuse` (task_type `should_refuse`) — are *excluded from the weighted quality score*
and feed the refusal gates instead; see [DOMAIN_PACKS.md](DOMAIN_PACKS.md) for why. C4
computes the weighted score from the remaining domains using the pack's normalized weights,
compares each candidate to the baseline artifact at the same quant, and applies the gates
from the spec (`Gates` in `core/config.py`; defaults in
[QUICKSTART.md](QUICKSTART.md#6-read-the-report)).

## Providers

One interface (`core/providers/base.py: LLMProvider.complete/embed`) serves dataset
generation, judging, and the init wizard; swapping providers is a config change:

| kind | What it is | Auth |
|---|---|---|
| `claude-cli` | shells out to the `claude` CLI (Claude Code print mode); prompt via stdin | your existing subscription — no key |
| `openai-compat` | any OpenAI-compatible server: LM Studio (default `http://localhost:1234/v1`), Ollama, llama-server, OpenRouter, OpenAI | optional `api_key_env` (name of an env var — never the key itself) |
| `anthropic-api` | direct Anthropic Messages API | `api_key_env`, default `ANTHROPIC_API_KEY` |
| `mock` | deterministic offline provider (canned responses + stable pseudo-embeddings) | — |

Determinism: temperature 0 wherever the task allows, fixed seeds, and pinned runtime
versions recorded in the manifest (PRD FR-3.5). Judge results are cached on disk under
`cache/judge/<config_hash>/`, keyed `sha256(item_id, artifact_hash, judge_version)` — a cache
hit never calls the provider, so re-runs cost nothing. Bump `judge.version` in the spec to
invalidate (note that `judge.version` is part of the config hash, so bumping it also starts a
fresh run directory).

## Windows-first notes

The lab machine is assumed to be a Windows box with an NVIDIA GPU: all paths go through
`pathlib`, subprocesses use list argv (no `shell=True`), file IO is explicit UTF-8, smoke
commands are split with a Windows-correct `shlex` mode, and llama.cpp/LM Studio conventions
follow the Windows layouts documented in [QUICKSTART.md](QUICKSTART.md). The only bash in the
system is the *generated remote provisioning script*, which runs on the rented Linux GPU box
([REMOTE_GPU.md](REMOTE_GPU.md)).

The one thing that genuinely does **not** run on Windows is the `local-offload` prune profile:
reap's locked environment pins vllm (and its CUDA torch build), which ships no Windows wheels.
The profile detects this and refuses up front with alternatives, rather than failing inside
`uv sync` minutes into the run (override: `REAPLAB_ALLOW_LOCAL_OFFLOAD=1`; details in
[REMOTE_GPU.md](REMOTE_GPU.md#alternative-local-offload)).
