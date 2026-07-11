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
2. Your **calibration prompts** (`calibration.jsonl`) — synthetic, prompts-only, generated
   by C1. No gold answers, no documents, **no client data** (see Security below).
3. Your `HF_TOKEN` (only if the base model is gated), passed as an environment variable at
   run time — never written into the script.

One thing travels back: the pruned HF checkpoint (safetensors + tokenizer + patched
`config.json`), typically 15–60 GB depending on model and retention. GGUF conversion and all
evaluation happen back on your machine — the rental exists for the prune alone.

## The generated script, step by step

```powershell
reap-lab prune configs/my-sweep.yaml --retention 0.5
```

With `execution_profile: remote`, this writes a self-contained bash script (path is printed;
it lands under the run directory) and tells you how to run it. If you've set
`prune.remote.ssh_host` (`user@host`) it can execute directly over SSH; otherwise you paste
it into the provider's onstart/startup hook. What the script does, in order:

1. **Arm the budget kill switch.** The script computes
   `max_hours = budget_usd / usd_per_hour` from your spec (defaults: $75 / $2.5/hr → 30 h,
   far above a normal run) and starts a watchdog that kills the job and powers the box down
   when the deadline hits. A hung download or a wedged prune can cost you at most the cap,
   even if you're asleep. Set `usd_per_hour` to what you're actually paying so the cap
   translates to real dollars.
2. **Build the environment.** Installs `uv`, clones `CerebrasResearch/reap` **recursively**
   (vendored submodules) at the pinned commit from your spec
   (`prune.reap_commit`, default `1970473` — never earlier than `3a44d0c`, the router-logit
   renormalization fix that measurably improves pruned accuracy), then runs the repo's
   `scripts/build.sh` (python ≥ 3.12, torch 2.7.1, transformers 4.55.0).
3. **Pre-download the model.** `hf download <model_id>` into the HF cache. This is
   mandatory — reap loads with `local_files_only` semantics; invoking the prune without the
   cache populated fails. `HF_TOKEN` is read from the environment for gated models.
4. **Stage the calibration data.** reap has no jsonl flag — it only accepts
   `load_dataset()`-compatible specs. The script converts your uploaded
   `calibration.jsonl` into a dataset folder with a `messages` column
   (role/content) and passes the folder path as `--dataset-name`.
5. **Run the prune.** Invokes the file directly (not `python -m`):

   ```bash
   python src/reap/prune.py \
       --model-name <model_id> \
       --dataset-name /workspace/calibration_ds \
       --compression-ratio 0.5 \
       --prune-method reap \
       --seed 42 --distance_measure cosine
   ```

   Note the conversion: **`--compression-ratio` is the fraction of experts REMOVED**, so
   the script passes `1 - retention` (retention 0.75 → ratio 0.25). dtype and device_map are
   not flags — reap hardcodes `torch_dtype="auto", device_map="auto"`.
6. **Locate and package the output.** The checkpoint is saved under
   `results_dir/pruned_models/{method}-{seed}-{ratio}/`; the script globs for it, verifies
   `config.json` reflects the reduced expert count, and tars it.
7. **Download.** You pull the tarball with `scp`/`rsync` (provider sections below show the
   exact lines). reap-lab unpacks it into `workspace/artifacts/`, verifies it (directory
   content hash, expert count), and writes the `ArtifactManifest` — config hash, retained
   expert map, wall clock, versions — so the artifact is traceable forever.
8. **Teardown.** Destroy the instance (one CLI line, per provider below). **Billing stops
   only when the instance is destroyed/deleted, not when the script finishes** — the kill
   switch stops runaway *compute*, but you should destroy promptly after downloading.

After download, back on your machine:

```powershell
reap-lab convert configs/my-sweep.yaml     # HF checkpoint -> GGUF across the quant grid
reap-lab eval configs/my-sweep.yaml        # score the GGUFs locally
```

(`reap-lab sweep` orchestrates all of this per retention value and resumes cleanly if you
run it again after the download completes — see
[ARCHITECTURE.md](ARCHITECTURE.md#reproducibility-config-hash--resume).)

## Provider guides

All prices are mid-2026 ballparks — check current rates. Any provider that gives you one
80 GB CUDA GPU with ~200 GB disk and SSH works; the script is provider-agnostic.

### Vast.ai — cheapest for scripted one-shots

```powershell
pip install vastai
vastai set api-key <YOUR_KEY>

# find one 80GB GPU with enough disk, verified host, best $/hr first
vastai search offers "gpu_ram>=80 num_gpus=1 disk_space>=200 verified=true rentable=true" -o "dph+"

# rent it (ID from the search output); onstart runs the generated script
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.7.1-cuda12.4-cudnn9-runtime `
  --disk 200 --onstart .\workspace\runs\<hash>\remote_prune.sh --ssh --direct

vastai show instances                       # wait for 'running', note ssh host/port
ssh -p <PORT> root@<HOST> "tail -f /workspace/prune.log"   # watch progress

# when done, pull the artifact then destroy
scp -P <PORT> root@<HOST>:/workspace/pruned-r0.5.tar.gz .\workspace\artifacts\
vastai destroy instance <INSTANCE_ID>
```

Pricing: A100 80GB ≈ **$0.67–1.20/hr**, H100 80GB ≈ **$1.47–1.87/hr**. Marketplace hosts
vary in reliability — filter `verified=true`, and prefer `--direct` (direct SSH) offers.

### RunPod — most predictable

Per-second billing, clean API/CLI (`runpodctl`), consistent images.

```powershell
# create a pod from the web console or CLI: 1x H100/A100 80GB, 200GB volume,
# image pytorch/pytorch:2.7.1-cuda12.4-cudnn9-runtime, SSH enabled
runpodctl create pod --gpuType "NVIDIA H100 80GB HBM3" --imageName "pytorch/pytorch:2.7.1-cuda12.4-cudnn9-runtime" --volumeSize 200

# ssh in (connection details from console / runpodctl), run the script, then:
scp <pod-ssh>:/workspace/pruned-r0.5.tar.gz .\workspace\artifacts\
runpodctl remove pod <POD_ID>              # billing stops here
```

Pricing: H100 ≈ **$2.89–3.29/hr** (Secure Cloud), **$1.80–2.40/hr** (Community Cloud).

### Lambda — simplest API, free egress

Fixed VM images, thin single-GPU availability, straightforward REST API; free egress is nice
for the multi-GB checkpoint download. H100 ≈ **$3.29/hr**. Launch a `gpu_1x_h100` instance
from console/API, SSH in with your registered key, run the script, `scp` the tarball,
terminate the instance.

### Budget math

A 30B-class run = model download (~61 GB) + calibration forward passes + prune + package
≈ **2–5 hours** end to end. At $1–3/hr that's **$5–15 per prune run**; a 3-retention sweep
is three runs (each retention is an independent prune). The default $75 cap covers the worst
realistic case with margin. GLM-4.5-Air-class models cost more (bigger download, layerwise
observer wall-clock) — budget accordingly or start from Cerebras's pre-pruned checkpoint
([QUICKSTART.md](QUICKSTART.md#4-the-buy-vs-build-shortcut--try-a-pre-pruned-model-first)).

## Security notes

- **No client data ever reaches the box.** The only dataset that travels is
  `calibration.jsonl`, which C1 generates as fully synthetic prompts
  ([DOMAIN_PACKS.md](DOMAIN_PACKS.md)); there are no gold answers and no source documents in
  it. If you seeded generation from firm documents, that seeding ran through your redaction
  step *locally, before any cloud call* — the remote box only ever sees synthetic output.
  You can (and should) read the jsonl before uploading; `reap-lab audit` samples it for you.
- **`HF_TOKEN` handling.** Needed only for gated base models. The generated script reads it
  from the environment (`ssh host "HF_TOKEN=... bash script.sh"` or the provider's secret
  env mechanism) — it is never written into the script file, never logged, and dies with the
  instance. Use a **fine-grained, read-only** token, not your account-wide write token.
- **Ephemeral by design.** Rent → prune → download → **destroy**. Nothing persists
  server-side; treat any provider volume as untrusted and destroy it with the instance.
- **SSH keys, not passwords.** All three providers support key auth; use it, and prefer
  direct SSH over web terminals for the download step.
- The remote box needs outbound HTTPS (Hugging Face, GitHub, PyPI). It needs **no** inbound
  ports beyond SSH.

## Alternative: local-offload

`prune.execution_profile: local-offload` runs REAP on your own machine, letting
`device_map="auto"` spill expert weights into system RAM (and disk, if you let it).

| | remote (default) | local-offload |
|---|---|---|
| Cost | ~$5–15/run | $0 |
| Wall clock (30B-class) | 2–5 h | many hours to days — CPU↔GPU shuttling dominates |
| Requirements | provider account, ~$ | ≥ 64–96 GB system RAM (bf16 model + overhead beyond your VRAM), fast NVMe helps |
| Data locality | synthetic calibration leaves the premises | **everything stays on-prem** |
| Babysitting | kill switch has your back | your box is busy/unusable meanwhile |

Choose local-offload when policy forbids any external compute, when your time is cheaper
than $10, or when you already own a 96–128 GB-RAM workstation and can leave it running
overnight. Choose remote for everything else — the whole point of the pipeline is that the
expensive step is scripted, capped, and disposable.

One machine-sizing note: local-offload competes with your normal work — the run pins the GPU
and most of your RAM. Run it overnight, and don't point the eval step at the same GPU until
the prune finishes.
