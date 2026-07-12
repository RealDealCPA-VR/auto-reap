# Quickstart — zero to a pruned, evaluated, promoted local model

`reap-lab` compresses a Mixture-of-Experts (MoE) model to fit **your** GPU by pruning the
experts your workload never uses. You describe your workload in plain English, the tool
generates calibration + eval datasets with a frontier model, runs REAP expert pruning across
a retention grid, converts every candidate to GGUF, evaluates each one **on the runtime that
actually serves it**, and hands you a ranked report with hard promotion gates. One winner
lands in your LM Studio models folder.

You do not need to have pruned a model before. Follow the numbered steps in order — each one
proves the previous one worked.

Related reading: [ARCHITECTURE.md](ARCHITECTURE.md) (how the pieces fit),
[DOMAIN_PACKS.md](DOMAIN_PACKS.md) (describing your workload),
[REMOTE_GPU.md](REMOTE_GPU.md) (the one step that needs a rented GPU).

---

## 0. Install

You need Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). On Windows (PowerShell):

```powershell
winget install astral-sh.uv        # or: irm https://astral.sh/uv/install.ps1 | iex
git clone <your-repo-url> reap-lab
cd reap-lab
uv sync
uv run reap-lab --help
```

macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then the same
`git clone` / `uv sync` / `uv run` lines.

Everything below is written as `reap-lab <command>`; if you did not install the tool
globally, prefix with `uv run` (e.g. `uv run reap-lab demo`).

### What you'll eventually need (not yet — `doctor` checks all of this)

| Thing | Used for | Notes |
|---|---|---|
| A frontier-model provider | dataset generation + judging | Easiest: [Claude Code](https://claude.com/claude-code) logged in (`claude` on PATH — zero API keys, runs inside your subscription). Alternatives: any OpenAI-compatible server or an Anthropic API key. |
| llama.cpp release binaries | GGUF conversion + local eval server | Windows: download **two** zips from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) — `llama-<tag>-bin-win-cuda-12.4-x64.zip` **plus** `cudart-llama-bin-win-cuda-12.4-x64.zip` — and unzip both into one folder. (The `winget` build is CPU-only; use the release zips for GPU.) |
| LM Studio (optional) | the promotion target + a convenient local OpenAI-compatible server | Default models dir `%USERPROFILE%\.lmstudio\models` is auto-detected. |
| A one-shot 80 GB GPU rental (~$5–15) | the prune step only | See [REMOTE_GPU.md](REMOTE_GPU.md). Everything else runs locally. |

## 1. Prove the pipeline works — `reap-lab demo` (offline, ~1 minute)

Before touching real models, GPUs, or API keys, run the whole pipeline end-to-end with
mocks — a mock frontier provider, a mock pruner, and a mock model runner:

```powershell
reap-lab demo                          # or: reap-lab demo --workspace D:\scratch\demo
```

This executes the real code path — generate → prune → convert → eval → report → promote —
in seconds, with zero network and zero GPU. Everything lands in `reap-lab-demo/` (override
with `--workspace`): an annotated `demo-sweep.yaml` + `demo-pack.yaml`, a genuine ranked
report at `workspace/reports/sweep-<config_hash>.md`, and a **sandboxed** `lmstudio-models/`
folder that the promote step writes into (your real LM Studio install is never touched).
`--no-show-report` skips the terminal render.

If `demo` passes, the machinery is sound; every later failure is about your environment
(missing tools, keys, disk), not the pipeline.

## 2. Check your environment — `reap-lab doctor`

```powershell
reap-lab doctor                        # generic checks
reap-lab doctor configs\my-sweep.yaml  # also validates the spec + the things IT needs
reap-lab doctor configs\my-sweep.yaml --strict   # exit 1 on any FAIL (for scripts)
```

`doctor` verifies each external dependency and tells you exactly what to install or fix:
llama.cpp binaries (`convert_hf_to_gguf.py`, `llama-quantize`, `llama-server`), your
configured providers (is `claude` on PATH? does the OpenAI-compatible server answer?),
GPU/VRAM via `nvidia-smi`, free disk space (candidates weigh 15–35 GB each), and the
LM Studio models directory. Fix what it flags, re-run until green.

## 3. Describe your workload — `reap-lab init`

```powershell
reap-lab init                          # interactive
reap-lab init --out configs --name my-workload --model-id Qwen/Qwen3-30B-A3B --yes
```

The wizard asks you to describe your workload in plain English ("a coding agent that mostly
edits Python and calls MCP tools, plus some general chat…") and uses your frontier provider
to **draft a domain pack** — a YAML file listing your workload's domains, their weights, task
types, and scoring setup — plus a sweep spec pointing at it. It writes exactly two files into
the current directory (or `--out DIR`):

- `<name>-pack.yaml` — the domain pack
- `<name>-sweep.yaml` — the sweep spec, with `domain_pack:` already pointing at its sibling

If a GPU is visible, `init` also pre-sets the VRAM gate to ~85% of detected VRAM. With
`--provider mock` (or if the draft call fails) you get a valid, clearly-marked **template**
pack to edit by hand instead of a drafted one.

Review and edit the draft; it is the single most leverage-rich file in the pipeline, because
pruning quality depends on the calibration data matching what you actually run. Full field
reference and a worked example: [DOMAIN_PACKS.md](DOMAIN_PACKS.md).

Prefer starting from an example? Ship-with packs live in `configs/domain-packs/`
(`cpa-firm`, `coding-agent`, `general-assistant`).

## 4. The buy-vs-build shortcut — try a pre-pruned model first

Cerebras publishes generic REAP-pruned checkpoints, and the community publishes GGUFs of
them. **Before paying for a custom prune run, evaluate one of these against your own eval
set** — if it clears your gates, you are done for $0:

| Pre-pruned GGUF | From | Good when |
|---|---|---|
| [`bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF`](https://huggingface.co/bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF) | Qwen3-Coder-30B-A3B, 25% experts removed | coding-heavy workloads |
| [`bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF`](https://huggingface.co/bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF) | GLM-4.5-Air 106B → 82B | 80 GB-class local boxes; tight on 48 GB |

```powershell
# generate your datasets once (uses your domain pack + frontier provider)
reap-lab generate configs/my-sweep.yaml

# spot-check the ~5% audit sample of what was generated, before trusting it
reap-lab audit configs/my-sweep.yaml

# evaluate a downloaded GGUF against your eval set on your own machine (--gguf is required)
reap-lab eval configs/my-sweep.yaml --gguf D:\models\cerebras_Qwen3-Coder-REAP-25B-A3B-Q4_K_M.gguf
```

`generate` prints the three paths it wrote — the datasets, and the audit sample `audit`
re-displays. They live in this sweep's own run directory,
`workspace/runs/<config_hash>/data/` ([why per-sweep](ARCHITECTURE.md#reproducibility-config-hash--resume)).

Two things to know about a standalone `eval`: open-ended domains fall back to a
non-refusal heuristic (the pairwise judge needs a same-quant baseline from the *same* sweep,
which a downloaded GGUF has no part in) — exact/json-schema/tool-call/refusal domains score
normally. And `promote` will not place a GGUF it did not build; if a community checkpoint
wins, copy it into LM Studio yourself.

The custom pipeline's edge is **domain-specific calibration** — a generic prune keeps experts
for everyone's workload, yours keeps experts for *your* workload. Validate that the generic
prune actually falls short before spending on step 5.

## 5. Run the sweep — `reap-lab sweep`

```powershell
reap-lab sweep configs/my-sweep.yaml
reap-lab sweep configs/my-sweep.yaml --promote     # promote the winner if the gates pass
reap-lab sweep configs/my-sweep.yaml --no-resume   # ignore completed stages, redo everything
```

One command executes the full grid: for each retention (default 0.75 / 0.625 / 0.50) prune,
then for each quant (default Q4_K_M / Q5_K_M) convert to GGUF, then evaluate every artifact —
plus the unpruned baseline — on your local runtime, capturing quality per domain, refusal
behavior, tool-call validity, tokens/sec, and peak VRAM at 4k and 32k context.

Two things to know before you start it:

- **The prune step itself runs on a remote 80 GB GPU** (a 30B MoE is ~61 GB in bf16 — it does
  not fit a 48 GB card, let alone 24). With the default `remote` profile and no
  `prune.remote.ssh_host`, the sweep does not fail — it writes a self-contained, budget-capped
  provisioning script plus numbered instructions into `workspace/prune/`, marks that prune
  stage **manual**, keeps going with the rest of the grid, and tells you (exit code 2 if
  nothing else could run). You rent a box, run the script, drop the tarball where it says, and
  re-run `sweep` — it picks up from there. Set `prune.remote.ssh_host` and reap-lab drives the
  whole thing over SSH instead. Walkthrough: [REMOTE_GPU.md](REMOTE_GPU.md).
- **Sweeps are resumable.** State lives in SQLite under `workspace/runs/<config_hash>/`, keyed
  by a hash of your config *content*; re-running the same spec skips completed stages, and one
  failed config never kills the rest
  (details: [ARCHITECTURE.md](ARCHITECTURE.md#reproducibility-config-hash--resume)). Kick it
  off overnight with confidence.

Check progress any time (stages done/failed/running/manual, plus the metrics recorded so far):

```powershell
reap-lab status configs/my-sweep.yaml
```

## 6. Read the report

```powershell
reap-lab report configs/my-sweep.yaml
```

`report` is a pure re-render: it reads the finished stages out of the state DB and rewrites
`workspace/reports/sweep-<config_hash>.md`. It never generates, builds, or evaluates anything
(`sweep` already wrote the same file at the end of its run — this just refreshes it).

The report contains: a ranked table by weighted score, per-domain breakdowns, a
quality-vs-VRAM-vs-speed Pareto view, a regression diff against the unpruned baseline at the
same quant, flagged anomalies, failed configs, and any manual steps still pending. Every
candidate is checked against the **promotion gates** (all user-tunable in the sweep YAML):

| Gate | Default | Kind |
|---|---|---|
| Weighted quality retention vs. baseline (same quant) | ≥ 95% | blocker |
| Any single domain regression | ≤ 5 points | blocker |
| Peak VRAM @ 32k context | ≤ 40 GB | blocker |
| False-refusal rate on the benign-sensitive suite | ≤ 2% and ≤ baseline | blocker |
| Should-refuse control set | 100% refused | hard fail |
| Tool-call schema validity | ≥ 98% | blocker |
| Decode tokens/sec | unset | advisory |

## 7. Ship it — `reap-lab promote`

```powershell
reap-lab promote configs/my-sweep.yaml
reap-lab promote configs/my-sweep.yaml --artifact r0.625-q4_k_m   # operator override
```

Like `report`, `promote` only re-reads finished stages — it never builds or re-evaluates.
For the winner it: **copies** the GGUF into your LM Studio models directory (in the required
`<publisher>/<model>/` two-folder layout), runs your optional `promote.smoke_command`, writes
a decision page (gates table, rationale, artifact hash, report reference) to
`promote.decision_dir` (default `workspace/reports/`), and **moves** the evaluated
non-winners into `workspace/archive/`. Nothing is ever deleted. If the smoke test fails, the
GGUF stays copied, the losers are *not* archived, and the command exits 1.

With no `--artifact`, promotion refuses to run when no candidate passed all blocking gates.
`--artifact <id>` overrides that — it promotes the artifact you name (even a gate-failing
one), and the decision page records it as an operator override. reap-lab will only promote
artifacts it built itself; a GGUF you scored with `eval --gguf` has no build manifest.

## 8. When a gate fails and there is no winner

This is normal on a first sweep — read it as data, not as breakage.

**What the report tells you.** The ranked table's `Gates` column is PASS/FAIL per candidate,
and the winner line says *"none — no candidate passed all blocking gates"*. The report
summarizes rather than tabulating every gate, so read across it:

- **Quality retention** — the `Retention vs baseline` column; anything under your
  `min_quality_retention` (default 95%) failed that gate.
- **Peak VRAM** — the `Peak VRAM (GB)` column, at `gates.min_context` (default 32k). If that
  context was never measured, the gate falls back to the largest context that *was*: already
  over the limit at a smaller context is a conclusive fail (VRAM only grows with context);
  under the limit is inconclusive and passes with a note telling you to add 32768 to
  `runtime.contexts`.
- **Per-domain regression** and **refusal regression** — the **Anomalies** section names the
  exact domain and the points dropped.
- **Notes** — rows whose numbers are qualified (e.g. no baseline at that quant, so the
  baseline-relative gates could not be measured at all).
- **The two the report does not print as numbers** are tool-call validity and the should-refuse
  control set. If a row says FAIL and nothing above explains it, one of those two is the
  culprit — the per-item evidence is in `workspace/runs/<config_hash>/results.jsonl` (filter by
  `artifact_id`, then look at the `tool_call` scorer rows and the `should_refuse` domain). The
  only rendered gate-by-gate table (value, limit, blocking, result) is the **decision page**,
  which is written when you promote.

**Which knob to turn.**

| Symptom | Turn this |
|---|---|
| Retention/regression gates fail everywhere | Keep more experts: add a higher retention (e.g. `retention: [0.875, 0.75]`). Pruning is the loss you can't quantize your way out of. |
| Only the aggressive quant fails | Use a gentler quant (`quants: [Q5_K_M, Q6_K]`) — VRAM cost is small next to a retention step. |
| VRAM gate fails but quality is fine | Lower retention or a smaller quant; or raise `gates.max_vram_gb` if you sized it too tight for your card. |
| One domain tanks, the rest are fine | That domain is under-calibrated: raise its `weight` in the pack (and check its `prompt_guidance`), then re-run — this *does* change the config hash and starts a fresh sweep. |
| Your gates were simply too strict | Edit `gates:` and re-run **`reap-lab report`** — see below. |

**Tuning the gates costs nothing.** Gate thresholds are deliberately **excluded from the
config hash**, so editing `gates:` does not orphan your sweep: the run directory, datasets and
artifacts stay put, and `reap-lab report configs/my-sweep.yaml` re-applies the new thresholds
to the measurements you already paid for and re-picks the winner. In seconds, no GPU. Changing
the pack, the model, the retention/quant grid or the generator, by contrast, mints a **new**
config hash and a fresh run (as it must — those change what was measured).

---

## VRAM reality table

Rough weights-only sizes (add ~2–6 GB for 32k-context KV cache + runtime overhead). Rules of
thumb: **bf16 ≈ 2.0 GB per B params; Q4_K_M ≈ 0.6 GB/B; Q5_K_M ≈ 0.7 GB/B; Q6_K ≈ 0.8 GB/B.**

| Model | bf16 (prune-time) | Q4_K_M | Q5_K_M | Fits 24 GB? | Fits 48 GB? | Fits 80 GB? |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B (baseline, 128 experts) | ~61 GB | ~19 GB | ~22 GB | short ctx only | yes, 32k OK | yes |
| … pruned r0.75 (~25B) | — | ~15 GB | ~18 GB | yes, 32k tight | yes | yes |
| … pruned r0.5 (~20B) | — | ~12 GB | ~14 GB | **yes, 32k OK** | yes | yes |
| Mixtral-8x7B (47B) | ~93 GB | ~28 GB | ~33 GB | no | yes | yes |
| GLM-4.5-Air REAP-82B | ~164 GB | ~50 GB | ~58 GB | no | tight/offload | yes |

Takeaways: the **bf16 column is why pruning runs on a rented 80 GB card** (or with slow local
CPU-offload); the quantized columns are what you actually serve. A 50%-retention prune is
what turns a 30B MoE from "24 GB card, short context only" into "24 GB card with 32k
headroom" — that is the whole game.

## What this costs

- **Frontier usage** (dataset generation + LLM judging): with the default `claude-cli`
  provider, it runs inside your existing Claude subscription — no API keys, no per-token
  bills. Judgments are cached, so re-runs cost nothing.
- **GPU rental** (the prune step only): a 30B-class REAP run is a few hours on one 80 GB
  card. At mid-2026 market prices (A100 80GB ≈ $0.67–1.20/hr on Vast.ai, H100 ≈ $1.47–3.29/hr
  depending on provider) that is **roughly $5–15 per prune run**; the generated script
  enforces a hard budget cap (default $75) as a kill switch. Provider-by-provider guide:
  [REMOTE_GPU.md](REMOTE_GPU.md).
- **Everything else is local and free**: GGUF conversion, evaluation, reporting, promotion.

## Where next

- How the pipeline is put together, artifact naming, resume semantics → [ARCHITECTURE.md](ARCHITECTURE.md)
- Authoring or tuning your domain pack → [DOMAIN_PACKS.md](DOMAIN_PACKS.md)
- The remote prune step, provider by provider → [REMOTE_GPU.md](REMOTE_GPU.md)
