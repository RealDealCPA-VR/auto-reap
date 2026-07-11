"""Unpruned baseline artifacts (spec.include_baseline).

The quality-retention gate is relative: every pruned artifact is compared to
the *baseline at the same quant* (PRD §5), so baselines get first-class
artifact ids (``baseline-<quant lowercase>``) and manifests just like pruned
GGUFs.

Three paths:
- ``spec.baseline_gguf`` set -> register the user's pre-quantized file.
- mock profile -> fabricate one fake GGUF per quant (offline demo/tests).
- otherwise -> require the model in the local HF cache, convert bf16 once,
  then ``llama-quantize`` per quant.
"""

from __future__ import annotations

import time
from pathlib import Path

from reaplab.core.config import SweepSpec
from reaplab.core.hashing import artifact_hash
from reaplab.core.paths import Workspace
from reaplab.core.records import ArtifactManifest
from reaplab.core.state import StateDB

from reaplab.prune import gguf
from reaplab.prune.profiles import resolve_hf_model_dir
from reaplab.prune.errors import PruneError
from reaplab.prune.runner import _load_manifest, _manifest_dir, _save_manifest, _tool_versions, model_slug
from reaplab.prune.stages import baseline_artifact_id, convert_stage


def build_baseline(spec: SweepSpec, workspace: Workspace, state: StateDB) -> list[ArtifactManifest]:
    """Produce (or register) the baseline GGUF artifacts.

    Returns ``[]`` when ``spec.include_baseline`` is False. Resumable via the
    ``convert:baseline-<quant>`` stages: done stages with files intact reload
    their manifests instead of re-running. Failures are marked in the StateDB
    and re-raised.
    """
    if not spec.include_baseline:
        return []
    config_hash = spec.config_hash()
    workspace.ensure(config_hash)
    man_dir = _manifest_dir(workspace, config_hash)

    if spec.baseline_gguf:
        return [_register_user_baseline(spec, config_hash, man_dir, state)]

    mock = spec.prune.execution_profile == "mock"
    slug = model_slug(spec.model_id)
    base_dir = workspace.artifacts / "baseline"
    bf16_path = base_dir / f"{slug}-bf16.gguf"
    tools: gguf.LlamaCppTools | None = None
    hf_dir: Path | None = None

    manifests: list[ArtifactManifest] = []
    for quant in spec.quants:
        canonical = gguf.validate_quant(quant)
        artifact_id = baseline_artifact_id(canonical)
        stage, key = convert_stage(artifact_id)
        gguf_path = base_dir / f"{slug}-{artifact_id}.gguf"

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
                    tools = gguf.LlamaCppTools.discover()
                if not bf16_path.exists():
                    if hf_dir is None:
                        # raises PrerequisiteError with `hf download` guidance
                        hf_dir = resolve_hf_model_dir(spec.model_id)
                    gguf.convert_to_gguf(hf_dir, bf16_path, tools, outtype="bf16")
                gguf.quantize(bf16_path, gguf_path, canonical, tools)
            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                kind="baseline",
                model_id=spec.model_id,
                retention=None,
                quant=canonical,
                path=str(gguf_path),
                config_hash=config_hash,
                artifact_hash=artifact_hash(gguf_path),
                wall_clock_s=round(time.monotonic() - t0, 3),
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


def _register_user_baseline(
    spec: SweepSpec,
    config_hash: str,
    man_dir: Path,
    state: StateDB,
) -> ArtifactManifest:
    """Register a user-provided pre-quantized baseline GGUF as an artifact.

    The quant is detected from the filename (e.g. ``...-Q4_K_M.gguf``); when
    undetectable, the first quant in ``spec.quants`` is assumed and stated in
    the manifest versions for transparency.
    """
    src = Path(spec.baseline_gguf or "")
    if not src.exists():
        raise PruneError(
            f"spec.baseline_gguf points at a missing file: {src}\n"
            "Fix the path in the sweep YAML, or remove baseline_gguf to have "
            "reap-lab convert the baseline itself."
        )
    detected = gguf.detect_quant_from_name(src.name)
    if detected is None and not spec.quants:
        raise PruneError(
            f"Cannot determine the quant of baseline_gguf '{src.name}' and spec.quants is empty.\n"
            "Rename the file to include its quant (e.g. -Q4_K_M.gguf) or add quants to the spec."
        )
    quant = detected or gguf.validate_quant(spec.quants[0])
    artifact_id = baseline_artifact_id(quant)
    stage, key = convert_stage(artifact_id)

    if state.is_done(stage, key):
        loaded = _load_manifest(man_dir, artifact_id)
        if loaded is not None and Path(loaded.path).exists():
            return loaded

    state.mark_running(stage, key)
    try:
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            kind="baseline",
            model_id=spec.model_id,
            retention=None,
            quant=quant,
            path=str(src),
            config_hash=config_hash,
            artifact_hash=artifact_hash(src),
            versions={
                "source": "user-provided",
                "quant_detection": "filename" if detected else f"assumed spec.quants[0]={quant}",
            },
        )
        man_path = _save_manifest(manifest, man_dir)
    except Exception as e:
        state.mark_failed(stage, key, str(e))
        raise
    state.mark_done(stage, key, meta={"path": str(src), "manifest": str(man_path)})
    return manifest
