"""Unpruned baseline artifacts (spec.include_baseline).

The quality-retention gate is relative: every pruned artifact is compared to
the *baseline at the same quant* (PRD §5), so baselines get first-class
artifact ids (``baseline-<quant lowercase>``) and manifests just like pruned
GGUFs.

Three paths:
- ``spec.baseline_gguf`` set -> register the user's pre-quantized file. It must
  cover every quant in the grid (one file cannot be the Q4_K_M *and* the Q5_K_M
  baseline), otherwise we fail fast with the three ways out.
- mock profile -> fabricate one fake GGUF per quant (offline demo/tests).
- otherwise -> require the model in the local HF cache, convert bf16 once,
  then ``llama-quantize`` per quant.

Baseline GGUFs live under ``artifacts/<config_hash>/baseline/`` (contract C3).
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
from reaplab.prune.errors import PruneError
from reaplab.prune.profiles import resolve_hf_model_dir
from reaplab.prune.runner import (
    _load_manifest,
    _manifest_dir,
    _reusable_manifest,
    _save_manifest,
    _tool_versions,
    artifacts_dir,
    model_slug,
    validated_quants,
)
from reaplab.prune.stages import baseline_artifact_id, convert_stage


def _user_baseline_quant(spec: SweepSpec) -> tuple[str, bool]:
    """(quant, detected_from_filename) for a user-provided baseline GGUF.

    Detection is by filename (``...-Q4_K_M.gguf``); when undetectable we assume the
    single quant in ``spec.quants`` and say so in the manifest.
    """
    src = Path(spec.baseline_gguf or "")
    detected = gguf.detect_quant_from_name(src.name)
    if detected is not None:
        return detected, True
    if not spec.quants:
        raise PruneError(
            f"Cannot determine the quant of baseline_gguf '{src.name}' and spec.quants is empty.\n"
            "Rename the file to include its quant (e.g. -Q4_K_M.gguf) or add quants to the spec."
        )
    return gguf.validate_quant(spec.quants[0]), False


def expected_baseline_ids(spec: SweepSpec) -> list[str]:
    """The artifact ids :func:`build_baseline` will produce for this spec (contract C4).

    The orchestrator uses this to know which ``convert:<id>`` rows to expect. It is
    NOT simply one per quant: a user-provided ``baseline_gguf`` is a single file and
    therefore a single artifact (which is exactly why build_baseline insists it cover
    the whole grid).
    """
    if not spec.include_baseline:
        return []
    if spec.baseline_gguf:
        quant, _ = _user_baseline_quant(spec)
        return [baseline_artifact_id(quant)]
    return [baseline_artifact_id(q) for q in validated_quants(spec)]


def build_baseline(spec: SweepSpec, workspace: Workspace, state: StateDB) -> list[ArtifactManifest]:
    """Produce (or register) the baseline GGUF artifacts.

    Returns ``[]`` when ``spec.include_baseline`` is False. Resumable via the
    ``convert:baseline-<quant>`` stages: done stages with files intact reload
    their manifests instead of re-running. Failures are marked in the StateDB
    and re-raised. Quant names are validated up front, before any conversion.
    """
    if not spec.include_baseline:
        return []
    config_hash = spec.config_hash()
    workspace.ensure(config_hash)
    man_dir = _manifest_dir(workspace, config_hash)
    quants = validated_quants(spec)  # fail fast on a typo, before llama.cpp runs

    if spec.baseline_gguf:
        return [_register_user_baseline(spec, quants, config_hash, man_dir, state)]

    mock = spec.prune.execution_profile == "mock"
    slug = model_slug(spec.model_id)
    base_dir = artifacts_dir(workspace, config_hash) / "baseline"
    bf16_path = base_dir / f"{slug}-bf16.gguf"
    tools: gguf.LlamaCppTools | None = None
    hf_dir: Path | None = None

    manifests: list[ArtifactManifest] = []
    for canonical in quants:
        artifact_id = baseline_artifact_id(canonical)
        stage, key = convert_stage(artifact_id)
        gguf_path = base_dir / f"{slug}-{artifact_id}.gguf"

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

    return manifests


def _register_user_baseline(
    spec: SweepSpec,
    quants: list[str],
    config_hash: str,
    man_dir: Path,
    state: StateDB,
) -> ArtifactManifest:
    """Register a user-provided pre-quantized baseline GGUF as an artifact.

    The file is one quant, so it can only be the baseline for that quant: the
    retention gate compares like with like (a Q5_K_M candidate against a Q4_K_M
    baseline reads as a quality *gain*). If the grid asks for quants this file does
    not cover, we stop before any prune runs and name the three ways out.
    """
    src = Path(spec.baseline_gguf or "")
    if not src.exists():
        raise PruneError(
            f"spec.baseline_gguf points at a missing file: {src}\n"
            "Fix the path in the sweep YAML, or remove baseline_gguf to have "
            "reap-lab convert the baseline itself."
        )
    quant, detected = _user_baseline_quant(spec)
    uncovered = [q for q in quants if q != quant]
    if uncovered:
        how = (
            "detected from its filename" if detected else "assumed (nothing in the filename says)"
        )
        raise PruneError(
            f"baseline_gguf '{src.name}' is a {quant} file ({how}), but the sweep also builds "
            f"{', '.join(uncovered)} candidates.\n"
            "Every candidate is gated against the baseline AT THE SAME QUANT (PRD §5) -- comparing "
            f"a {uncovered[0]} candidate to a {quant} baseline would misreport quality retention, "
            "so reap-lab will not do it.\n"
            "Pick one:\n"
            f"  * quants: [{quant}]                 -- score only the quant your baseline covers\n"
            "  * remove baseline_gguf              -- reap-lab converts one baseline per quant "
            "itself (needs the HF model in the local cache + llama.cpp tools)\n"
            "  * include_baseline: false           -- no baseline at all; relative gates then "
            "report 'not measured' and pass"
        )
    artifact_id = baseline_artifact_id(quant)
    stage, key = convert_stage(artifact_id)

    if state.is_done(stage, key):
        loaded = _load_manifest(man_dir, artifact_id)
        if loaded is not None and loaded.config_hash == config_hash and Path(loaded.path).exists():
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
