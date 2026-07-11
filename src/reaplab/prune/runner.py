"""Per-retention artifact pipeline: dataset -> prune -> bf16 GGUF -> quant grid.

``build_artifacts`` is the orchestrator's entry point for one retention value.
It is resumable: stages already marked done in the StateDB (with their files
still on disk) are skipped, and their manifests are reloaded from the run dir.

Stage keys follow the shared contract: ``prune:r<retention:g>`` and
``convert:<artifact_id>`` (see :mod:`reaplab.prune.stages`).
"""

from __future__ import annotations

import time
from pathlib import Path

from reaplab.core.config import SweepSpec
from reaplab.core.hashing import artifact_hash
from reaplab.core.paths import Workspace
from reaplab.core.records import ArtifactManifest
from reaplab.core.state import StateDB
from reaplab.prune import gguf, profiles
from reaplab.prune.errors import PruneError
from reaplab.prune.reap_cmd import DATASET_FILENAME, calib_to_dataset_dir, retention_tag
from reaplab.prune.stages import convert_stage, prune_stage, pruned_artifact_id

#: Folder (under workspace/data) holding the messages-column calibration dataset.
CALIBRATION_DATASET_DIRNAME = "calibration_dataset"


def model_slug(model_id: str) -> str:
    """Filesystem-friendly model name: last path component of the HF id."""
    return model_id.split("/")[-1]


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


def _tool_versions(spec: SweepSpec, tools: gguf.LlamaCppTools | None) -> dict[str, str]:
    versions = {
        "reap_commit": spec.prune.reap_commit,
        "execution_profile": spec.prune.execution_profile,
    }
    if tools is not None:
        versions["convert_hf_to_gguf"] = str(tools.convert_script)
        versions["llama_quantize"] = str(tools.quantize_bin)
    else:
        versions["gguf_tools"] = "mock"
    return versions


def ensure_calibration_dataset(calibration_path: Path, workspace: Workspace) -> Path:
    """Convert calibration.jsonl into the REAP dataset folder once per workspace.

    Idempotent: if ``data.jsonl`` already exists it is reused (the calibration
    file is part of the config hash, so its content is stable per run dir).
    """
    dataset_dir = workspace.data / CALIBRATION_DATASET_DIRNAME
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

    Flow: calibration dataset folder (once) -> REAP prune via the configured
    execution profile (stage ``prune:r<r:g>``) -> bf16 GGUF conversion ->
    ``llama-quantize`` per quant (stage ``convert:<artifact_id>``). One
    :class:`ArtifactManifest` per GGUF, persisted to
    ``runs/<config_hash>/manifests/<artifact_id>.json``.

    Resumable: done stages whose files still exist are skipped; their
    manifests are loaded from disk. Failures are marked in the StateDB and
    re-raised so the orchestrator can isolate them.
    """
    config_hash = spec.config_hash()
    workspace.ensure(config_hash)
    man_dir = _manifest_dir(workspace, config_hash)
    rtag = retention_tag(retention)
    mock = spec.prune.execution_profile == "mock"
    slug = model_slug(spec.model_id)

    dataset_dir = ensure_calibration_dataset(calibration_path, workspace)

    # -- prune stage --------------------------------------------------------
    stage, key = prune_stage(retention)
    default_hf_dir = workspace.artifacts / f"{slug}-{rtag}-hf"
    prune_s = 0.0
    hf_dir = Path(state.meta(stage, key).get("path") or default_hf_dir)
    if not (state.is_done(stage, key) and (hf_dir / "config.json").exists()):
        profile = profiles.get_profile(
            spec, work_dir=workspace.root / "prune", log_dir=workspace.logs(config_hash)
        )
        state.mark_running(stage, key)
        t0 = time.monotonic()
        try:
            hf_dir = profile.run_prune(spec, retention, dataset_dir, default_hf_dir)
        except Exception as e:
            state.mark_failed(stage, key, str(e))
            raise
        prune_s = time.monotonic() - t0
        state.mark_done(stage, key, meta={"path": str(hf_dir)})

    # -- bf16 conversion (shared across the quant grid) ----------------------
    bf16_path = workspace.artifacts / f"{slug}-{rtag}-bf16.gguf"
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
    for quant in spec.quants:
        canonical = gguf.validate_quant(quant)
        artifact_id = pruned_artifact_id(retention, canonical)
        stage, key = convert_stage(artifact_id)
        gguf_path = workspace.artifacts / f"{slug}-{artifact_id}.gguf"

        if state.is_done(stage, key):
            existing = Path(state.meta(stage, key).get("path") or gguf_path)
            loaded = _load_manifest(man_dir, artifact_id)
            if existing.exists() and loaded is not None:
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
                wall_clock_s=round(prune_s + bf16_s + quant_s, 3),
                versions=_tool_versions(spec, tools),
            )
            man_path = _save_manifest(manifest, man_dir)
        except Exception as e:
            state.mark_failed(stage, key, str(e))
            raise
        state.mark_done(stage, key, meta={"path": str(gguf_path), "manifest": str(man_path)})
        manifests.append(manifest)

    if not manifests:
        raise PruneError(
            "spec.quants is empty -- add at least one quantization (e.g. Q4_K_M) to the sweep YAML."
        )
    return manifests
