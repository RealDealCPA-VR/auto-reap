# Research Brief: REAP → GGUF → LM Studio integration facts

Grounded 2026-07-11 via web research. Builder agents: treat this as the source of truth
for external interfaces; note UNCONFIRMED items and code defensively around them.

## 1. CerebrasResearch/reap

- Repo: https://github.com/CerebrasResearch/reap — Apache-2.0.
- Layout: `src/reap/` (`args.py`, `prune.py`, `merge.py`, `observer.py`, `model_util.py`, `data.py`), `experiments/pruning-cli.sh`, `scripts/build.sh`, `config/`, `tests/`.
- Entry point (from `experiments/pruning-cli.sh`) — invokes the file, not `python -m`:

```bash
python src/reap/prune.py \
    --model-name <hf_id> --dataset-name <dataset_spec> \
    --compression-ratio <ratio> --prune-method reap \
    --seed <seed> --distance_measure cosine \
    --record_pruning_metrics_only true \
    --batch_size <n> --batches_per_category <n> ...
```

- **`compression_ratio` = fraction of experts REMOVED** (0.25 = drop 25% = keep 75%).
  Our sweep specs use *retention*; convert: `compression_ratio = 1 - retention`.
  Alternative: `num_clusters` = absolute number of experts kept.
- Args parsed via `HfArgumentParser` over dataclasses: `ReapArgs, DatasetArgs, ObserverArgs, ModelArgs, EvalArgs, PruneArgs, ClusterArgs`.
- `ObserverArgs.renormalize_router_weights: bool = True` ("renormalize topk router weights to sum to 1"). Introduced-as-default by commit **`3a44d0ceb8922c01c71987ca1513e35ce4c7f737`** ("Renorm router logits by default (#13)", 2026-03-13). Later: `2b114e7` composite calibration datasets (#16), latest known `1970473` layerwise calibration observer (#17, 2026-04-17 — memory-efficient, big models on one GPU). **Pin at/after `3a44d0c`; default pin `1970473`.**
- dtype/device_map are NOT CLI flags — hardcoded `device_map="auto", torch_dtype="auto", trust_remote_code=True, local_files_only=True`. **Model must be pre-downloaded to the HF cache** (`hf download <model_id>` first). (`local_files_only` conditional: verify at runtime; code defensively.)
- Calibration data: **HF `load_dataset()` only — no jsonl flag.** Chat datasets need a `"messages"` column (role/content); plain-LM datasets need `"text"`. Composite spec: `name[subset](split):num_batches`, comma-separated. A local folder path containing a jsonl with a `messages` column generally loads via `load_dataset(folder)`; our remote script converts `calibration.jsonl` → messages-format dataset folder and passes its path; fallback = push a private HF dataset.
- Output: `results_dir/pruned_models/{method}-{seed}-{ratio}` via `save_pretrained` (safetensors + tokenizer + patched `config.json` num_experts). Exact `results_dir` mechanism UNCONFIRMED → locate via glob after the run.
- Env: python >=3.12, torch==2.7.1, transformers==4.55.0, vllm==0.10.0, uv-managed, clone `--recursive` (vendored eval submodules); `bash scripts/build.sh`; `.env` from `.env.template` (HF_TOKEN etc.).
- Validated models incl. Qwen3-30B-A3B (their default), GLM-4.5-Air, GLM-4.6, Mixtral-8x7B, Llama-4-Scout.
- Published pre-pruned checkpoints (HF collection `cerebras/cerebras-reap`): `cerebras/GLM-4.5-Air-REAP-82B-A12B` (+`-FP8`), `cerebras/Qwen3-Coder-REAP-25B-A3B`, `cerebras/GLM-4.6-REAP-218B-A32B` and others. **No plain Qwen3-30B-A3B REAP checkpoint published** → custom prune has real value there. Community GGUFs of these exist (`bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF`, `bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF`) → buy-vs-build quick win.
- REAP checkpoints convert with the stock llama.cpp converter (community GGUFs prove it). Convert from **BF16** checkpoints, not FP8/NVFP4.

## 2. llama.cpp

- Converter: `convert_hf_to_gguf.py <hf_dir> --outfile model-bf16.gguf --outtype bf16`.
  `--outtype` choices: `f32,f16,bf16,q8_0,tq1_0,tq2_0,auto`. K-quants happen in llama-quantize, not here. Reads expert count from `config.json` — pruned MoE needs no special flag.
- Quantize: `llama-quantize [--imatrix f] in-bf16.gguf out-Q4_K_M.gguf Q4_K_M [nthreads]`.
  Confirmed spellings: `Q4_K_M`, `Q5_K_M`, `Q6_K` (also Q4_K_S, Q5_K_S, Q8_0, ...).
- Server: `llama-server -m model.gguf -c <ctx> --host 127.0.0.1 --port <p> [--metrics]`.
  OpenAI-compatible `/v1/chat/completions`; per-request timings on `/v1/*` when the request body sets `"timings_per_token": true` → response `timings` object with `prompt_per_second`, `predicted_per_second`. `--metrics` → Prometheus `/metrics`. `GET /props` = model metadata.
- Windows install: GitHub release zips (tag ~b9966): `llama-<tag>-bin-win-cuda-12.4-x64.zip` **plus** `cudart-llama-bin-win-cuda-12.4-x64.zip`, unzip into one folder. `winget install ggml.llamacpp` exists but is CPU-only — use release zips for GPU.

## 3. LM Studio (Windows)

- Models dir: `%USERPROFILE%\.lmstudio\models\<publisher>\<model>\<file>.gguf` — the two-level publisher/model layout is REQUIRED (root-dropped GGUFs are not detected). Or `lms import path\to\model.gguf`.
- OpenAI-compatible server at `http://localhost:1234/v1`: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings` (works with local embedding models → usable for near-dup filtering).
- Headless: `lms server start|stop`, `lms load <model-key> [--gpu=max] [--context-length=N] [-y]`, `lms unload --all`, `lms ls`, `lms ps`, `lms import`.

## 4. Remote GPU (one-shot 80 GB prune)

- **Vast.ai** — cheapest scripted one-shot: `vastai search offers` → `vastai create instance <id> --image pytorch/pytorch:2.x-cuda12.4-cudnn9-runtime --disk 200 --onstart file.sh --ssh --direct` → rsync/scp artifacts → `vastai destroy instance <id>`. H100 80GB ≈ $1.47–1.87/hr, A100 80GB ≈ $0.67–1.20/hr.
- **RunPod** — most predictable: REST/GraphQL + `runpodctl`; H100 ≈ $2.89–3.29/hr Secure, $1.80–2.40 Community; per-second billing.
- **Lambda** — simplest REST, free egress, but fixed VM images and thin single-GPU capacity; H100 ≈ $3.29/hr.
- Budget math: a 30B-class REAP run (download + observe + prune + upload) is a few hours → $75 cap is comfortable on any of these.

## Pipeline-relevant gotchas

1. REAP has no jsonl input — ship calibration as a messages-column dataset folder (or private HF dataset).
2. Pre-download the model into the HF cache before invoking prune.py.
3. Pin reap ≥ `3a44d0c` (router renormalization); default `1970473`.
4. Convert BF16 → GGUF, then quantize; never convert FP8 variants.
5. LM Studio needs publisher/model two-folder layout or `lms import`.
6. llama-server perf capture: request-level `timings_per_token: true`.
