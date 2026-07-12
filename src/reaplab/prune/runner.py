"""Per-retention artifact pipeline: dataset -> prune -> bf16 GGUF -> quant grid.

``build_artifacts`` is the orchestrator's entry point for one retention value.
It is resumable: stages already marked done in the StateDB (with their files
still on disk) are skipped, and their manifests are reloaded from the run dir.

Everything one sweep writes is namespaced by ``spec.config_hash()``:

    runs/<config_hash>/data/calibration_dataset/   REAP's messages-column dataset
    runs/<config_hash>/manifests/<artifact_id>.json
    artifacts/<config_hash>/<slug>-<rtag>-hf/      pruned HF checkpoint
    artifacts/<config_hash>/<slug>-<rtag>-bf16.gguf
    artifacts/<config_hash>/<slug>-<artifact_id>.gguf

so two specs sharing a workspace can never read (or clobber) each other's
artifacts — including the mock-vs-real case, where the profile is part of the
hash. The path itself is the proof of provenance: an existing file under this
config's directory belongs to this config.

Stage keys follow the shared contract: ``prune:r<retention:g>`` and
``convert:<artifact_id>`` (see :mod:`reaplab.prune.stages`).
"""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from reaplab import __version__
from reaplab.core.config import SweepSpec
from reaplab.core.hashing import artifact_hash
from reaplab.core.paths import Workspace
from reaplab.core.records import ArtifactManifest
from reaplab.core.state import StateDB
from reaplab.prune import gguf, profiles
from reaplab.prune.errors import NeedsManualStep, PruneError
from reaplab.prune.reap_cmd import DATASET_FILENAME, calib_to_dataset_dir, retention_tag
from reaplab.prune.stages import convert_stage, prune_stage, pruned_artifact_id

log = logging.getLogger("reaplab.prune")

#: Folder (under runs/<config_hash>/data) holding the messages-column calibration dataset.
CALIBRATION_DATASET_DIRNAME = "calibration_dataset"

#: config.json keys that carry the expert count, across MoE architectures.
EXPERT_COUNT_FIELDS = (
    "num_experts",  # qwen3_moe, mixtral (patched by REAP)
    "num_local_experts",  # mixtral (HF canonical)
    "n_routed_experts",  # deepseek v2/v3
    "moe_num_experts",  # glm/phi-moe variants
    "num_experts_per_layer",
)


def model_slug(model_id: str) -> str:
    """Filesystem-friendly model name: last path component of the HF id.

    Two orgs can publish the same basename (``unsloth/X`` vs ``org/X``); that is
    harmless because artifact paths are namespaced by config hash, and model_id is
    part of the hash — the two never share a directory.
    """
    return model_id.split("/")[-1]


def artifacts_dir(workspace: Workspace, config_hash: str) -> Path:
    """Per-config artifact directory: ``artifacts/<config_hash>`` (contract C3)."""
    d = workspace.artifacts / config_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_dir(workspace: Workspace, config_hash: str) -> Path:
    d = workspace.run_dir(config_hash) / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_manifest(manifest: ArtifactManifest, man_dir: Path) -> Path:
    path = man_dir / f"{manifest.artifact_id}.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def _load_manifest(man_dir: Path, artifact_id: str) -> ArtifactManifest | None:
    path = man_dir / f"{artifact_id}.json"
    if not path.exists():
        return None
    return ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _reusable_manifest(
    man_dir: Path, artifact_id: str, config_hash: str, expected_path: Path
) -> ArtifactManifest | None:
    """A manifest may only be reused when it belongs to THIS config and its file is
    still on disk under this config's artifact directory. Anything else (a manifest
    from an older hash, a path outside the namespace) is rebuilt."""
    loaded = _load_manifest(man_dir, artifact_id)
    if loaded is None or loaded.config_hash != config_hash:
        return None
    path = Path(loaded.path)
    if path != expected_path or not path.exists():
        return None
    return loaded


def validated_quants(spec: SweepSpec) -> list[str]:
    """Canonical llama.cpp quant names for the spec — validated eagerly.

    Called before any prune work: a typo (``q4km``) must cost a second, not a
    completed $75 remote prune (and the empty-list case must not surface as an
    empty result either).
    """
    if not spec.quants:
        raise PruneError(
            "spec.quants is empty -- add at least one quantization (e.g. Q4_K_M) to the sweep YAML."
        )
    return [gguf.validate_quant(q) for q in spec.quants]


@lru_cache(maxsize=8)
def _llama_cpp_build(quantize_bin: str) -> str:
    """The llama.cpp build behind a llama-quantize binary, e.g. "b9966 (a1b2c3d)".

    llama-quantize prints its build banner on stderr when invoked with no args (and
    exits nonzero — that is expected). Best-effort: an unreadable banner records
    "unknown" rather than failing a prune (PRD FR-3.5 wants versions pinned in the
    manifest, but a missing banner is not a reason to lose the run)."""
    try:
        proc = subprocess.run(  # noqa: S603 - path came from our own discovery
            [quantize_bin],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    blob = f"{proc.stderr or ''}\n{proc.stdout or ''}"
    match = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", blob)
    if match:
        return f"b{match.group(1)} ({match.group(2)})"
    match = re.search(r"\bbuild[:\s]+(\S+)", blob)
    return match.group(1) if match else "unknown"


def _tool_versions(spec: SweepSpec, tools: gguf.LlamaCppTools | None) -> dict[str, str]:
    """Provenance for one artifact (PRD FR-2.3/FR-3.5): what produced it, at what version."""
    versions = {
        "reaplab": __version__,
        "python": platform.python_version(),
        "reap_commit": spec.prune.reap_commit,
        "execution_profile": spec.prune.execution_profile,
    }
    if tools is not None:
        versions["convert_hf_to_gguf"] = str(tools.convert_script)
        versions["llama_quantize"] = str(tools.quantize_bin)
        versions["llama_cpp_build"] = _llama_cpp_build(str(tools.quantize_bin))
    else:
        versions["gguf_tools"] = "mock"
    return versions


def read_expert_stats(hf_dir: Path) -> dict[str, Any] | None:
    """Expert counts from a pruned checkpoint's ``config.json`` (PRD FR-2.3 provenance).

    REAP rewrites the expert count in the saved config, so this is the one piece of
    real post-prune evidence available locally without loading the weights: it lets a
    user confirm the checkpoint they downloaded actually has the experts they paid to
    prune. Returns None when the file is missing/unreadable or carries no known
    expert-count field (recording a wrong number would be worse than recording none).
    """
    config_path = Path(hf_dir) / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("could not read expert counts from %s: %s", config_path, e)
        return None
    if not isinstance(config, dict):
        return None
    for field in EXPERT_COUNT_FIELDS:
        value = config.get(field)
        if isinstance(value, int) and value > 0:
            stats: dict[str, Any] = {
                "num_experts_after": value,
                "num_experts_source_field": field,
            }
            active = config.get("num_experts_per_tok")
            if isinstance(active, int) and active > 0:
                stats["num_experts_per_tok"] = active
            model_type = config.get("model_type")
            if isinstance(model_type, str):
                stats["model_type"] = model_type
            return stats
    log.warning(
        "pruned checkpoint %s has no known expert-count field (%s); manifest saliency_stats "
        "left empty rather than guessed",
        config_path, ", ".join(EXPERT_COUNT_FIELDS),
    )
    return None


def ensure_calibration_dataset(
    calibration_path: Path, workspace: Workspace, config_hash: str
) -> Path:
    """Convert calibration.jsonl into the REAP dataset folder for THIS config.

    Lives at ``runs/<config_hash>/data/calibration_dataset`` (contract C2), so a
    second spec's calibration set can never be the one REAP calibrates against.
    Idempotent: an existing ``data.jsonl`` under this config's run dir is reused;
    a missing one is rebuilt.
    """
    dataset_dir = workspace.data_dir(config_hash) / CALIBRATION_DATASET_DIRNAME
    if (dataset_dir / DATASET_FILENAME).exists():
        return dataset_dir
    return calib_to_dataset_dir(Path(calibration_path), dataset_dir)


def build_artifacts(
    spec: SweepSpec,
    retention: float,
    calibration_path: Path,
    workspace: Workspace,
    state: StateDB,
) -> list[ArtifactManifest]:
    """Produce every GGUF artifact for one retention value (PRD FR-2.1..2.4).

    Flow: quant validation -> calibration dataset folder (once) -> REAP prune via
    the configured execution profile (stage ``prune:r<r:g>``) -> bf16 GGUF
    conversion -> ``llama-quantize`` per quant (stage ``convert:<artifact_id>``).
    One :class:`ArtifactManifest` per GGUF, persisted to
    ``runs/<config_hash>/manifests/<artifact_id>.json``.

    Resumable: done stages whose files still exist (and whose manifests carry this
    config hash) are skipped. Failures are marked in the StateDB and re-raised so
    the orchestrator can isolate them; a :class:`NeedsManualStep` marks the prune
    stage 'manual' instead of 'failed' (contract C6) — nothing is broken, the user
    simply has to run the generated remote script.
    """
    config_hash = spec.config_hash()
    workspace.ensure(config_hash)
    quants = validated_quants(spec)  # before any prune work (a typo must cost 0 GPU-hours)
    man_dir = _manifest_dir(workspace, config_hash)
    art_dir = artifacts_dir(workspace, config_hash)
    rtag = retention_tag(retention)
    mock = spec.prune.execution_profile == "mock"
    slug = model_slug(spec.model_id)

    dataset_dir = ensure_calibration_dataset(calibration_path, workspace, config_hash)

    # -- prune stage --------------------------------------------------------
    stage, key = prune_stage(retention)
    hf_dir = art_dir / f"{slug}-{rtag}-hf"
    prune_meta = state.meta(stage, key)
    prune_s = 0.0
    peak_mem_gb: float | None = None
    peak_mem_note: str | None = None
    saliency_stats: dict[str, Any] | None = None

    if state.is_done(stage, key) and (hf_dir / "config.json").exists():
        # resume: the checkpoint is under THIS config's artifact dir, so it is ours
        peak_mem_gb = prune_meta.get("peak_mem_gb")
        peak_mem_note = prune_meta.get("peak_mem_note")
        saliency_stats = prune_meta.get("saliency_stats")
    else:
        profile = profiles.get_profile(
            # per-config work dir: a tarball produced for a DIFFERENT pack (hence
            # different calibration data) must never be picked up as this config's
            # pruned checkpoint just because it shares a retention value
            spec,
            work_dir=workspace.run_dir(config_hash) / "prune",
            log_dir=workspace.logs(config_hash),
        )
        state.mark_running(stage, key)
        t0 = time.monotonic()
        try:
            hf_dir = Path(profile.run_prune(spec, retention, dataset_dir, hf_dir))
        except NeedsManualStep as e:
            # not a failure: the sweep is waiting on the user (contract C6)
            state.mark_manual(stage, key, str(e))
            raise
        except Exception as e:
            state.mark_failed(stage, key, str(e))
            raise
        prune_s = time.monotonic() - t0
        peak_mem_gb = profile.peak_mem_gb
        peak_mem_note = None if peak_mem_gb is not None else profile.peak_mem_note
        if not mock:
            saliency_stats = read_expert_stats(hf_dir)
        state.mark_done(
            stage,
            key,
            meta={
                "path": str(hf_dir),
                "peak_mem_gb": peak_mem_gb,
                "peak_mem_note": peak_mem_note,
                "saliency_stats": saliency_stats,
            },
        )

    # -- bf16 conversion (shared across the quant grid) ----------------------
    bf16_path = art_dir / f"{slug}-{rtag}-bf16.gguf"
    tools: gguf.LlamaCppTools | None = None
    bf16_s = 0.0
    if not bf16_path.exists():
        t0 = time.monotonic()
        if mock:
            gguf.write_fake_gguf(bf16_path, seed=f"{spec.model_id}|{rtag}|bf16")
        else:
            tools = gguf.LlamaCppTools.discover(
                convert_script=spec.prune.convert_script,
                quantize_bin=spec.prune.llama_quantize,
            )
            gguf.convert_to_gguf(hf_dir, bf16_path, tools, outtype="bf16")
        bf16_s = time.monotonic() - t0

    # -- quant grid -----------------------------------------------------------
    manifests: list[ArtifactManifest] = []
    for canonical in quants:
        artifact_id = pruned_artifact_id(retention, canonical)
        stage, key = convert_stage(artifact_id)
        gguf_path = art_dir / f"{slug}-{artifact_id}.gguf"

        if state.is_done(stage, key):
            loaded = _reusable_manifest(man_dir, artifact_id, config_hash, gguf_path)
            if loaded is not None:
                manifests.append(loaded)
                continue

        state.mark_running(stage, key)
        try:
            t0 = time.monotonic()
            if mock:
                gguf.write_fake_gguf(gguf_path, seed=f"{spec.model_id}|{artifact_id}")
            else:
                if tools is None:
                    tools = gguf.LlamaCppTools.discover(
                        convert_script=spec.prune.convert_script,
                        quantize_bin=spec.prune.llama_quantize,
                    )
                gguf.quantize(bf16_path, gguf_path, canonical, tools)
            quant_s = time.monotonic() - t0
            versions = _tool_versions(spec, tools)
            if peak_mem_gb is None and peak_mem_note:
                versions["peak_mem_gb"] = peak_mem_note
            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                kind="gguf",
                model_id=spec.model_id,
                retention=retention,
                quant=canonical,
                path=str(gguf_path),
                config_hash=config_hash,
                artifact_hash=artifact_hash(gguf_path),
                reap_commit=spec.prune.reap_commit,
                saliency_stats=saliency_stats,
                peak_mem_gb=peak_mem_gb,
                wall_clock_s=round(prune_s + bf16_s + quant_s, 3),
                versions=versions,
            )
            man_path = _save_manifest(manifest, man_dir)
        except Exception as e:
            state.mark_failed(stage, key, str(e))
            raise
        state.mark_done(stage, key, meta={"path": str(gguf_path), "manifest": str(man_path)})
        manifests.append(manifest)

    return manifests
