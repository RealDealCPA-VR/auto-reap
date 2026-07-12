"""Winner promotion (PRD FR-4.4): LM Studio placement, decision page, smoke
test hook, and loser archival.

LM Studio only detects models in a two-level ``<models-dir>/<publisher>/
<model-name>/<file>.gguf`` layout (docs/RESEARCH_BRIEF.md section 3), so the
winner is copied — never moved — into exactly that shape. Losers are archived
by *moving* them into ``workspace.archive``; nothing is ever deleted.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from reaplab.core.config import SweepSpec
from reaplab.core.paths import Workspace
from reaplab.core.records import ArtifactManifest
from reaplab.orchestrate.scoring import GateResult

_SMOKE_TIMEOUT_S = 900.0
_OUTPUT_TAIL = 500  # chars of smoke output kept in messages/pages


class PromotionResult(BaseModel):
    """Outcome of promote_winner. ``stage`` names the step that decided the
    outcome: resolve | disk | copy | smoke | done."""

    ok: bool
    stage: str
    message: str
    dest_path: Path | None = None
    decision_page: Path | None = None


def _fs_safe(name: str) -> str:
    """Filesystem-safe folder/file component: keep [A-Za-z0-9._-], collapse
    the rest to '-'. Never returns an empty string."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return cleaned or "model"


def _free_bytes(path: Path) -> int:
    """Free bytes on the volume that will hold ``path`` (walks up to the
    nearest existing ancestor first, since the destination may not exist yet)."""
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def split_command(command: str, windows: bool | None = None) -> list[str]:
    """Split a smoke command into argv, correctly on Windows AND POSIX.

    ``shlex.split`` in its default POSIX mode treats ``\\`` as an escape, so a
    Windows path (``C:\\tools\\smoke.exe``) comes out as ``C:toolssmoke.exe`` and
    the command can never be found. Windows therefore splits with
    ``posix=False`` — which keeps quote characters *inside* the tokens
    (``"C:\\Program Files\\smoke.exe"`` stays quoted and would still fail
    ``subprocess.run``), so one surrounding pair of matching quotes is stripped
    from each token afterwards.

    Placeholder substitution ({model}/{path}) happens AFTER splitting, so a
    substituted path containing spaces stays a single argv token.
    """
    if windows is None:
        windows = os.name == "nt"
    tokens = shlex.split(command, posix=not windows)
    if not windows:
        return tokens
    stripped: list[str] = []
    for token in tokens:
        if len(token) >= 2 and token[0] in ('"', "'") and token[-1] == token[0]:
            token = token[1:-1]
        stripped.append(token)
    return stripped


def _archive_losers(
    workspace: Workspace, winner_src: Path, losers: list[ArtifactManifest]
) -> int:
    """Move exactly the given loser artifacts into workspace.archive (move, never
    delete). Returns the count moved.

    Only artifacts that were EVALUATED and lost are ever passed in by the
    orchestrator — never a never-evaluated build, never a bf16 intermediate,
    never a baseline. Archiving by globbing the artifacts tree (the old
    behavior) permanently stranded candidates a partially-completed grid still
    needed, so this function does no discovery of its own.
    """
    winner = winner_src.resolve()
    moved = 0
    for manifest in losers:
        src = Path(manifest.path)
        if not src.is_file() or src.resolve() == winner:
            continue
        workspace.archive.mkdir(parents=True, exist_ok=True)
        target = workspace.archive / src.name
        counter = 1
        while target.exists():
            target = workspace.archive / f"{src.stem}-{counter}{src.suffix}"
            counter += 1
        shutil.move(str(src), str(target))
        moved += 1
    return moved


def _write_decision_page(
    spec: SweepSpec,
    manifest: ArtifactManifest,
    report_path: Path,
    workspace: Workspace,
    dest: Path,
    gates: list[GateResult] | None,
    rationale: str,
    smoke_status: str,
) -> Path:
    """Write the promotion decision page markdown; returns its path."""
    out_dir = Path(spec.promote.decision_dir) if spec.promote.decision_dir else workspace.reports
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / f"decision-{_fs_safe(manifest.artifact_id)}-{manifest.config_hash}.md"

    lines: list[str] = [
        f"# Promotion decision: {manifest.artifact_id}",
        "",
        f"- **Model:** `{spec.model_id}`",
        f"- **Artifact:** `{manifest.artifact_id}`",
        f"- **Artifact hash:** `{manifest.artifact_hash or 'n/a'}`",
        f"- **Config hash:** `{manifest.config_hash}`",
        f"- **Promoted to:** `{dest}`",
        f"- **Report:** `{report_path}`",
        f"- **Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Gates",
        "",
    ]
    if gates:
        lines.append("| Gate | Value | Limit | Blocking | Result |")
        lines.append("|---|---|---|---|---|")
        for g in gates:
            value = "n/a" if g.value is None else f"{g.value:.4f}"
            limit = "n/a" if g.limit is None else f"{g.limit:g}"
            result = "PASS" if g.passed else "FAIL"
            blocking = "yes" if g.blocking else "advisory"
            lines.append(f"| {g.name} | {value} | {limit} | {blocking} | {result} |")
    else:
        lines.append("Gate results were not provided to the promotion step.")
    lines += [
        "",
        "## Rationale",
        "",
        rationale
        or "Selected by reap-lab: highest weighted score among candidates passing all blocking gates.",
        "",
        "## Smoke test",
        "",
        smoke_status,
        "",
    ]
    page.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return page


def promote_winner(
    spec: SweepSpec,
    manifest: ArtifactManifest,
    report_path: Path,
    workspace: Workspace,
    *,
    gates: list[GateResult] | None = None,
    rationale: str = "",
    losers: list[ArtifactManifest] | None = None,
) -> PromotionResult:
    """Promote a winning GGUF into LM Studio and record the decision.

    Steps, in order:
      1. resolve — locate the source GGUF and the LM Studio models dir
         (``spec.promote.lmstudio_dir`` or ``~/.lmstudio/models``); build the
         REQUIRED two-level ``<publisher>/<model-name>/<file>.gguf`` layout.
      2. disk — verify free space >= the GGUF size before copying.
      3. copy — shutil.copy2 (the source stays in workspace.artifacts).
      4. smoke — run ``spec.promote.smoke_command`` if set. The command is
         split platform-correctly (see :func:`split_command`), then ``{model}``
         (publisher/model-name key) and ``{path}`` (destination GGUF) are
         substituted per token, so Windows paths survive intact. Non-zero exit
         => ok=False, stage="smoke", and losers are NOT archived.
      5. decision page — markdown with the gates table, rationale, artifact
         hash, report reference, and smoke outcome (written even on smoke
         failure, so there is always a record).
      6. archive — move exactly the ``losers`` manifests (the EVALUATED
         non-winner candidates the caller passes in) into workspace.archive,
         only when ``spec.promote.archive_losers`` and the smoke test passed.
         With ``losers=None`` nothing is archived: promotion never goes looking
         for files to move, so a never-evaluated candidate the grid still needs
         can't be stranded.

    Nothing is ever deleted; losers are moved, the winner is copied.
    """
    src = Path(manifest.path)
    if not src.is_file():
        return PromotionResult(
            ok=False,
            stage="resolve",
            message=(
                f"Winner GGUF not found at {src}. Re-run `uv run reap-lab sweep` to rebuild it "
                "(completed stages resume automatically), or check that the workspace artifacts "
                "directory was not moved."
            ),
        )

    lms_root = (
        Path(spec.promote.lmstudio_dir)
        if spec.promote.lmstudio_dir
        else Path.home() / ".lmstudio" / "models"
    )
    publisher = _fs_safe(spec.promote.publisher or "reap-lab")
    model_name = _fs_safe(f"{spec.model_id.split('/')[-1]}-{manifest.artifact_id}")
    dest_dir = lms_root / publisher / model_name
    dest = dest_dir / f"{model_name}.gguf"

    size = src.stat().st_size
    free = _free_bytes(dest_dir)
    if free < size:
        return PromotionResult(
            ok=False,
            stage="disk",
            message=(
                f"Not enough free disk space to promote: the GGUF is {size / 1e9:.2f} GB but only "
                f"{free / 1e9:.2f} GB is free on the volume holding {lms_root}. Free up space or "
                "point `promote.lmstudio_dir` in the sweep YAML at a larger drive."
            ),
        )

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    except OSError as e:
        return PromotionResult(
            ok=False,
            stage="copy",
            message=(
                f"Copying the winner to {dest} failed: {e}. Check permissions on the LM Studio "
                "models directory and that no other process holds the file open."
            ),
        )

    # Smoke test -------------------------------------------------------------
    smoke_ok = True
    smoke_status = "Not configured (`promote.smoke_command` unset)."
    if spec.promote.smoke_command:
        model_key = f"{publisher}/{model_name}"
        try:
            tokens = split_command(spec.promote.smoke_command)
        except ValueError as e:
            smoke_ok = False
            smoke_status = f"FAILED — could not parse smoke_command: {e}"
            tokens = []
        if tokens:
            tokens = [
                t.replace("{model}", model_key).replace("{path}", str(dest)) for t in tokens
            ]
            try:
                proc = subprocess.run(
                    tokens,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=_SMOKE_TIMEOUT_S,
                )
            except FileNotFoundError:
                smoke_ok = False
                smoke_status = (
                    f"FAILED — smoke command not found: `{tokens[0]}`. Install it or fix "
                    "`promote.smoke_command` in the sweep YAML."
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                smoke_ok = False
                smoke_status = f"FAILED — smoke command error: {e}"
            else:
                output = ((proc.stdout or "") + (proc.stderr or "")).strip()
                tail = output[-_OUTPUT_TAIL:] if output else "(no output)"
                if proc.returncode != 0:
                    smoke_ok = False
                    smoke_status = f"FAILED — exit code {proc.returncode}. Output tail: {tail}"
                else:
                    smoke_status = f"PASSED (exit 0). Output tail: {tail}"

    page = _write_decision_page(
        spec, manifest, report_path, workspace, dest, gates, rationale, smoke_status
    )

    if spec.promote.smoke_command and not smoke_ok:
        return PromotionResult(
            ok=False,
            stage="smoke",
            message=(
                f"Smoke test failed for {manifest.artifact_id}: {smoke_status} "
                f"The GGUF was copied to {dest} but losers were NOT archived; fix the smoke "
                "command or the model, then re-run promotion."
            ),
            dest_path=dest,
            decision_page=page,
        )

    archived = 0
    if spec.promote.archive_losers and losers:
        archived = _archive_losers(workspace, src, losers)

    message = f"Promoted {manifest.artifact_id} to {dest}."
    if spec.promote.smoke_command:
        message += " Smoke test passed."
    if archived:
        message += f" Archived {archived} loser GGUF(s) to {workspace.archive}."
    return PromotionResult(
        ok=True, stage="done", message=message, dest_path=dest, decision_page=page
    )
