# Remote GPU — the one step that doesn't run on your machine

Everything in reap-lab runs locally **except the REAP prune itself**. This doc explains why,
walks through the generated provisioning script, and gives per-provider instructions for
renting a single 80 GB GPU for a few hours (~$5–15 per prune run, hard-capped by a budget
kill switch).

Related: [QUICKSTART.md](QUICKSTART.md) (where this fits),
[ARCHITECTURE.md](ARCHITECTURE.md) (execution profiles, artifact flow),
[DOMAIN_PACKS.md](DOMAIN_PACKS.md) (the calibration data that travels to the box).

## Why the prune needs ~80 GB

REAP loads the **full unpruned model in bf16**, runs your calibration prompts through it to
observe router gates and expert activations, then prunes and saves. bf16 is 2 bytes per
parameter, plus activation/observer overhead on top:

| Model | Params | bf16 weights | Fits where |
|---|---|---|---|
| Qwen3-30B-A3B / Qwen3-Coder-30B-A3B | ~30.5B | **~61 GB** | one 80 GB card (A100/H100) |
| Mixtral-8x7B | ~47B | **~93 GB** | one 80 GB card only with offload; prefer 2×80 GB or a 141 GB H200 |
| GLM-4.5-Air (106B-A12B) | ~106B | **~212 GB** | multi-GPU node, or single-GPU via reap's layerwise observer (slow) |

Your 24/48 GB local card cannot hold the bf16 model, and pruning **must** see the full model
— you can't prune a quantized GGUF. Hence the three execution profiles in the sweep spec
(`prune.execution_profile`):

- **`remote`** (default) — rent one 80 GB GPU, run a generated script, download the pruned
  checkpoint, destroy the box. This doc.
- **`local-offload`** — run REAP locally with weights spilled to system RAM. Free, slow,
  fully on-prem. See [the last section](#alternative-local-offload).
- **`mock`** — no GPU at all; used by `reap-lab demo` and tests.

The pinned reap commit (`1970473`, 2026-04-17) includes the **layerwise calibration
observer**, which processes the model layer-by-layer and makes big models feasible on a
single GPU at the cost of wall-clock. For 30B-class models on an 80 GB card you don't need
it; for GLM-4.5-Air-class models it's the difference between "single rented GPU" and
"multi-GPU node".

## What actually travels to the rented box

Three things, and only three things:

1. The generated provisioning script (bash).
2. Your **calibration dataset folder** — `runs/<config_hash>/data/calibration_dataset/`,
   holding one `data.jsonl` whose every row is `{"messages": [{"role": "user", "content":
   "<prompt>"}]}`. reap-lab builds it locally from `calibration_v1.jsonl`, because REAP has no
   jsonl flag — it only accepts an HF `load_dataset()` target. Synthetic, prompts-only: no
   gold answers, no documents, **no client data** (see Security below).
3. Your `HF_TOKEN` (only if the base model is gated), **piped in on stdin** at run time —
   never written into the script, never on a command line (see Security below).

One thing travels back: the pruned HF checkpoint (safetensors + tokenizer + patched
`config.json`), tarred, typically 15–60 GB depending on model and retention. GGUF conversion
and all evaluation happen back on your machine — the rental exists for the prune alone.

## The generated script, step by step

```powershell
reap-lab prune configs/my-sweep.yaml --retention 0.5
```

(`reap-lab sweep` does the same thing for every retention in the grid — this is just the
one-at-a-time form.) With `execution_profile: remote`, three files are involved, and they all
live in **`workspace/prune/`** — not the run directory:

| Path | What it is |
|---|---|
| `workspace\prune\prune_remote_r0.5.sh` | the generated, self-contained bash script (one per retention) |
| `workspace\prune\REMOTE_STEPS_r0.5.md` | numbered manual instructions, written when `prune.remote.ssh_host` is unset |
| `workspace\prune\pruned_r0.5.tar.gz` | where reap-lab **expects to find** the tarball you download |

If `prune.remote.ssh_host` (`user@host`) is set, reap-lab drives the whole flow itself
(`ssh`/`scp` subprocesses, logged to `runs/<config_hash>/logs/remote-r0.5.log`). If it is
unset, the prune stage is marked **`manual`** — not failed — the instructions file is written,
and the sweep continues with the rest of the grid.

> **Gotcha:** the `workspace/prune/` scratch dir is keyed by *retention only*, not by config
> hash. If you change your pack and re-run at the same retention, an old
> `pruned_r0.5.tar.gz` still sitting there will be picked up and extracted as if it belonged to
> the new config. Delete or rename stale tarballs when you change the pack.

**First, before any of its six numbered steps, the script arms the budget kill switch.** It
computes `seconds = budget_usd / usd_per_hour × 3600` from your spec (defaults: $75 / $2.50/hr
→ 30 h, far above a normal run) and **re-executes itself under `timeout`**, so the cap covers
the *entire* run — clone, `uv sync`, the 61 GB model download, the prune and the packaging —
not just the prune. A stalled download bills exactly like a wedged prune, so it is guarded
exactly the same. On expiry it prints a loud banner and exits 124. Set `usd_per_hour` to what
you are actually paying, so the cap translates to real dollars. It does **not** power the box
down (see below).

Then it prints `== [1/6]` … `== [6/6]` as it goes:

1. **Clone reap at the pinned commit** — `git clone --recursive` (vendored submodules) +
   `git checkout` of `prune.reap_commit` (default `1970473`; never go earlier than `3a44d0c`,
   the router-logit renormalization fix that measurably improves pruned accuracy).
2. **Build the environment** — installs `uv` if missing, then runs the repo's own
   `scripts/build.sh` (python ≥ 3.12, torch 2.7.1, transformers 4.55.0).
3. **Pre-download the model** — `uv run hf download <model_id>` (falling back to
   `huggingface-cli download`) into the HF cache. This is mandatory: reap loads with
   `local_files_only` semantics, so a prune with an empty cache just fails. `HF_TOKEN` is read
   from the environment for gated models.
4. **Verify the calibration dataset** — the script does *not* build it. It expects the folder
   you uploaded at `$WORK/dataset/` to contain `data.jsonl`, and exits 2 with a clear message
   if it does not.
5. **Run the prune,** from the reap repo root, invoking the file directly (not `python -m`):

   ```bash
   uv run python src/reap/prune.py \
       --model-name <model_id> \
       --dataset-name "$DATASET" \
       --compression-ratio 0.5 \
       --prune-method reap \
       --seed 42 \
       --distance_measure cosine \
       --record_pruning_metrics_only false
   ```

   Note the conversion: **`--compression-ratio` is the fraction of experts REMOVED**, so the
   script passes `1 - retention` (retention 0.75 → ratio 0.25). `--record_pruning_metrics_only
   false` is explicit: we want the checkpoint saved, not just the saliency metrics. dtype and
   device_map are not flags — reap hardcodes `torch_dtype="auto", device_map="auto"`.
6. **Package the output** — finds the `pruned_models` directory REAP wrote and tars it to
   `$WORK/pruned_r0.5.tar.gz`, printing `DONE: <path>`.

Then the script stops. It deliberately does **not** shut the box down: on most rentals a
guest-side shutdown keeps billing the still-allocated instance while destroying the shell you
need to fetch the result. **Billing stops when you destroy the instance** — that is your
explicit step, below.

`$WORK` is `$HOME/reap-work` (override with the `REAP_WORK` env var). So on the box:
the script sits at `~/reap-work/prune_remote_r0.5.sh`, the dataset at `~/reap-work/dataset/`,
and the result at `~/reap-work/pruned_r0.5.tar.gz`.

## Resuming after a manual remote run

This is the part people get wrong. After the remote prune finishes:

1. **Download the tarball to exactly the path reap-lab named** (it is printed in the manual
   step message and in `REMOTE_STEPS_r0.5.md`):

   ```powershell
   scp user@HOST:reap-work/pruned_r0.5.tar.gz .\workspace\prune\pruned_r0.5.tar.gz
   ```

2. **Destroy the instance** (billing stops here).

3. **Re-run the sweep:**

   ```powershell
   reap-lab sweep configs/my-sweep.yaml
   ```

   That is the whole resume story. The remote profile short-circuits on two things, in order:
   the extracted checkpoint (`artifacts/<config_hash>/<slug>-r0.5-hf/config.json`) — already
   done, nothing happens; failing that, the tarball at `workspace\prune\pruned_r0.5.tar.gz` —
   it extracts it, locates the checkpoint inside (a directory with `config.json` **and** at
   least one `*.safetensors`, preferring anything under `pruned_models/`), moves it into the
   artifacts dir, records the manifest, and carries straight on to GGUF conversion, quantization
   and evaluation. Only if *neither* exists does it re-emit the manual-step instructions.

   `reap-lab prune configs/my-sweep.yaml --retention 0.5` does the same for that one retention.

**Commands that will *not* do this,** despite what you might guess:

- `reap-lab convert <spec>` builds **only the unpruned baseline** GGUFs for the quant grid.
  It never touches your pruned checkpoint.
- `reap-lab eval <spec>` requires `--gguf <file>` and scores exactly that one file. It builds
  nothing.

Both are useful, neither resumes a prune. Re-run `sweep` (or `prune`).

## Provider guides

All prices are mid-2026 ballparks — check current rates. Any provider that gives you one
80 GB CUDA GPU with ~200 GB disk and SSH works; the script is provider-agnostic.

The fully automated path is the same everywhere: once the box is up, put its `user@host` in
`prune.remote.ssh_host` and run `reap-lab sweep` — reap-lab uploads the script and the
dataset, runs the prune (piping `HF_TOKEN` over stdin if it is set in your environment),
downloads the tarball to `workspace\prune\`, and logs every step. The manual lines below are
for when you would rather drive it yourself.

### Vast.ai — cheapest for scripted one-shots

```powershell
pip install vastai
vastai set api-key <YOUR_KEY>

# find one 80GB GPU with enough disk, verified host, best $/hr first
vastai search offers "gpu_ram>=80 num_gpus=1 disk_space>=200 verified=true rentable=true" -o "dph+"

# rent it (ID from the search output)
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.7.1-cuda12.4-cudnn9-runtime `
  --disk 200 --ssh --direct
vastai show instances                       # wait for 'running', note ssh host/port

# upload the script + the calibration dataset folder (paths reap-lab printed)
ssh -p <PORT> root@<HOST> "mkdir -p reap-work"
scp -P <PORT> .\workspace\prune\prune_remote_r0.5.sh root@<HOST>:reap-work/prune_remote_r0.5.sh
scp -P <PORT> -r .\workspace\runs\<config_hash>\data\calibration_dataset root@<HOST>:reap-work/dataset

# run it (this streams the 6 steps; tee if you want a log on the box)
ssh -p <PORT> root@<HOST> "bash reap-work/prune_remote_r0.5.sh 2>&1 | tee reap-work/prune.log"

# when it prints DONE, pull the tarball to EXACTLY this path, then destroy
scp -P <PORT> root@<HOST>:reap-work/pruned_r0.5.tar.gz .\workspace\prune\pruned_r0.5.tar.gz
vastai destroy instance <INSTANCE_ID>
```

Pricing: A100 80GB ≈ **$0.67–1.20/hr**, H100 80GB ≈ **$1.47–1.87/hr**. Marketplace hosts
vary in reliability — filter `verified=true`, and prefer `--direct` (direct SSH) offers.

Note: Vast's `--onstart` hook is a poor fit here — the script needs the calibration dataset
folder to already be on the box (it exits 2 without it), so upload first, then run.

### RunPod — most predictable

Per-second billing, clean API/CLI (`runpodctl`), consistent images.

```powershell
# create a pod from the web console or CLI: 1x H100/A100 80GB, 200GB volume,
# image pytorch/pytorch:2.7.1-cuda12.4-cudnn9-runtime, SSH enabled
runpodctl create pod --gpuType "NVIDIA H100 80GB HBM3" --imageName "pytorch/pytorch:2.7.1-cuda12.4-cudnn9-runtime" --volumeSize 200

# same three steps as above: scp the script + dataset, ssh to run it, then:
scp <pod-ssh>:reap-work/pruned_r0.5.tar.gz .\workspace\prune\pruned_r0.5.tar.gz
runpodctl remove pod <POD_ID>              # billing stops here
```

Pricing: H100 ≈ **$2.89–3.29/hr** (Secure Cloud), **$1.80–2.40/hr** (Community Cloud).

### Lambda — simplest API, free egress

Fixed VM images, thin single-GPU availability, straightforward REST API; free egress is nice
for the multi-GB checkpoint download. H100 ≈ **$3.29/hr**. Launch a `gpu_1x_h100` instance
from console/API, SSH in with your registered key, upload the script + dataset folder, run the
script, `scp` the tarball back to `workspace\prune\`, terminate the instance.

### Budget math

A 30B-class run = model download (~61 GB) + calibration forward passes + prune + package
≈ **2–5 hours** end to end. At $1–3/hr that's **$5–15 per prune run**; a 3-retention sweep
is three runs (each retention is an independent prune). The default $75 cap covers the worst
realistic case with margin. GLM-4.5-Air-class models cost more (bigger download, layerwise
observer wall-clock) — budget accordingly or start from Cerebras's pre-pruned checkpoint
([QUICKSTART.md](QUICKSTART.md#4-the-buy-vs-build-shortcut--try-a-pre-pruned-model-first)).

## Security notes

- **No client data ever reaches the box — because none ever enters the pipeline.** reap-lab
  has no seed-material ingestion path: every calibration and eval item is *generated*, either
  procedurally (mock) or by your frontier provider from your pack's descriptions
  ([DOMAIN_PACKS.md](DOMAIN_PACKS.md)). The only thing uploaded is the calibration dataset
  folder — synthetic prompts, no gold answers, no source documents. You can (and should) read
  `data.jsonl` before uploading; `reap-lab audit` prints a sample of the matching eval set.
- **`HF_TOKEN` handling.** Needed only for gated base models, and it never appears on a
  command line. In SSH mode reap-lab pipes the token over **stdin** and the remote shell reads
  it (`IFS= read -r HF_TOKEN; export HF_TOKEN; bash …`), which keeps it out of your argv, the
  on-disk log, any raised error, and — the one people forget — the *remote* box's process
  table, where `ps` shows every user the full ssh command line. Doing it manually, use the same
  trick:

  ```powershell
  # PowerShell, gated model, manual run
  $env:HF_TOKEN | ssh user@HOST "IFS= read -r HF_TOKEN; export HF_TOKEN; bash reap-work/prune_remote_r0.5.sh"
  ```

  Any `HF_TOKEN=...` that does slip into a rendered command is redacted to `HF_TOKEN=***`
  before it is logged. Use a **fine-grained, read-only** token, not your account-wide write
  token.
- **Ephemeral by design.** Rent → prune → download → **destroy**. Nothing persists
  server-side; treat any provider volume as untrusted and destroy it with the instance.
- **SSH keys, not passwords.** All three providers support key auth; use it, and prefer
  direct SSH over web terminals for the download step.
- The remote box needs outbound HTTPS (Hugging Face, GitHub, PyPI). It needs **no** inbound
  ports beyond SSH.

## Alternative: local-offload

`prune.execution_profile: local-offload` runs REAP on your own machine, letting
`device_map="auto"` spill expert weights into system RAM (and disk, if you let it). It clones
reap into `workspace/prune/reap`, builds it with `uv sync`, runs the same prune command the
remote script runs, and moves the resulting checkpoint into `artifacts/<config_hash>/`. It also
samples the process tree's resident memory and records the peak in the manifest.

> **It refuses to run on Windows.** REAP's locked environment pins vllm (and the CUDA torch
> build it needs), which publishes no Windows wheels — `uv sync` would fail during resolution,
> minutes into the run. Rather than let you discover that the slow way, the profile raises up
> front and names your options: switch to `remote`, switch to `mock`, or run reap-lab from
> Linux/WSL with the GPU passed through. If you know better (a WSL setup that looks like `nt`
> to Python, or upstream shipped wheels), override with **`REAPLAB_ALLOW_LOCAL_OFFLOAD=1`**:
>
> ```powershell
> $env:REAPLAB_ALLOW_LOCAL_OFFLOAD = "1"
> ```
>
> It also checks up front that `git` and `uv` are on PATH and that the base model is already in
> your Hugging Face cache (REAP loads with `local_files_only`), each with the exact command to
> fix it.

| | remote (default) | local-offload |
|---|---|---|
| Cost | ~$5–15/run | $0 |
| Wall clock (30B-class) | 2–5 h | many hours to days — CPU↔GPU shuttling dominates |
| OS | any (bash runs on the rented Linux box) | **Linux/WSL only** (see above) |
| Requirements | provider account, ~$ | ≥ 64–96 GB system RAM (bf16 model + overhead beyond your VRAM), fast NVMe helps, model pre-downloaded |
| Data locality | synthetic calibration leaves the premises | **everything stays on-prem** |
| Babysitting | kill switch has your back | no budget cap (there is no bill to cap); your box is busy/unusable meanwhile |

Choose local-offload when policy forbids any external compute, when your time is cheaper
than $10, or when you already own a 96–128 GB-RAM Linux workstation and can leave it running
overnight. Choose remote for everything else — the whole point of the pipeline is that the
expensive step is scripted, capped, and disposable.

One machine-sizing note: local-offload competes with your normal work — the run pins the GPU
and most of your RAM. Run it overnight, and don't point the eval step at the same GPU until
the prune finishes.
