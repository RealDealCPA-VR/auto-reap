"""Execution profiles for the REAP prune step (PRD FR-2.2).

One interface, three backends:

- :class:`MockProfile` -- fabricates a tiny pruned HF checkpoint instantly.
  Used by tests and ``reap-lab demo``; zero GPU, zero network.
- :class:`LocalOffloadProfile` -- clones the pinned reap repo locally and runs
  the prune on this box (48 GB VRAM + system RAM offload; slow, free).
- :class:`RemoteProfile` -- generates a self-contained provisioning bash
  script for a rented 80 GB GPU box, under a budget kill switch. With
  ``prune.remote.ssh_host`` set it orchestrates upload/run/download itself
  (ssh/scp subprocesses); otherwise it writes the script + numbered
  instructions and raises :class:`NeedsManualStep`.

All profiles share ``run_prune(spec, retention, dataset_dir, out_dir) -> Path``
returning the local pruned HF checkpoint directory.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from reaplab.core.config import SweepSpec
from reaplab.prune.errors import NeedsManualStep, PrerequisiteError, PruneError
from reaplab.prune.gguf import deterministic_bytes
from reaplab.prune.reap_cmd import (
    DATASET_FILENAME,
    build_prune_command,
    format_ratio,
    retention_tag,
)

#: Default number of experts assumed by the mock profile when fabricating a
#: checkpoint (Qwen3-30B-A3B's real shape: 128 experts / 8 active).
MOCK_BASE_EXPERTS = 128

#: Remote work dir, relative to $HOME on the rented box.
REMOTE_WORKDIR = "reap-work"

#: Escape hatch for the local-offload Windows block (see LocalOffloadProfile).
ALLOW_LOCAL_OFFLOAD_ENV = "REAPLAB_ALLOW_LOCAL_OFFLOAD"

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_@%+=:,./\-]+$")

#: Any ``HF_TOKEN=<value>`` occurrence in a rendered command line. Secrets must never
#: reach the on-disk log, an exception message, the state DB, or the sweep report.
_TOKEN_RE = re.compile(r"(HF_TOKEN=)\S+")


def redact(text: str) -> str:
    """Mask HF_TOKEN values in any string we log or raise (PRD/doc promise: never logged)."""
    return _TOKEN_RE.sub(r"\1***", text)


def _display(argv: list[str]) -> str:
    """Redacted, human-readable form of an argv list — the only form that is ever
    written to a log or embedded in an error message."""
    return redact(" ".join(argv))


def _shell_join(tokens: list[str]) -> str:
    """Join argv into a bash line; tokens starting with ``$`` stay unquoted so
    shell variables (e.g. ``$DATASET``) expand on the remote box."""
    parts: list[str] = []
    for t in tokens:
        if t.startswith("$") or _SAFE_TOKEN.match(t):
            parts.append(t)
        else:
            parts.append("'" + t.replace("'", "'\"'\"'") + "'")
    return " ".join(parts)


def budget_timeout_seconds(spec: SweepSpec) -> int:
    """Kill-switch duration: ``budget_usd / usd_per_hour`` hours, in seconds."""
    remote = spec.prune.remote
    if remote.usd_per_hour <= 0:
        raise PruneError(
            f"prune.remote.usd_per_hour must be > 0 (got {remote.usd_per_hour}); "
            "it converts the budget cap into a wall-clock kill switch."
        )
    return int(remote.budget_usd / remote.usd_per_hour * 3600)


def resolve_hf_model_dir(model_id: str) -> Path:
    """Locate *model_id* in the local Hugging Face cache (snapshot dir with config.json).

    REAP loads models with ``local_files_only=True`` (research brief section 1),
    so the model must be pre-downloaded. Honors ``HUGGINGFACE_HUB_CACHE`` then
    ``HF_HOME`` then the default ``~/.cache/huggingface/hub``.
    """
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not cache:
        hf_home = os.environ.get("HF_HOME")
        cache = str(Path(hf_home) / "hub") if hf_home else str(Path.home() / ".cache" / "huggingface" / "hub")
    snapshots = Path(cache) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if snapshots.is_dir():
        candidates = [p for p in snapshots.iterdir() if p.is_dir() and (p / "config.json").exists()]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    raise PrerequisiteError(
        f"Model '{model_id}' is not in the local Hugging Face cache ({cache}).\n"
        f"REAP loads with local_files_only=True, so download it first:\n"
        f"    hf download {model_id}\n"
        f"(or: huggingface-cli download {model_id}; set HF_TOKEN for gated models)."
    )


def _tree_rss_gb(proc_handle: object) -> float:
    """Resident memory of a process plus its children, in GB. 0.0 when unmeasurable."""
    try:
        import psutil  # noqa: PLC0415 - optional-ish, only used while a prune runs
    except ImportError:  # pragma: no cover - psutil is a hard dependency, defensive only
        return 0.0
    total = 0
    try:
        procs = [proc_handle, *proc_handle.children(recursive=True)]  # type: ignore[attr-defined]
        for p in procs:
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        return 0.0
    return total / 1e9


def _run_logged(
    argv: list[str], *, cwd: Path | None, log_path: Path, track_peak_mem: bool = False
) -> float | None:
    """Run one subprocess, streaming combined stdout/stderr into *log_path*.

    List argv only (never shell=True). Raises :class:`PruneError` with the log
    location on non-zero exit; the rendered command is always redacted, so a
    secret in the environment can never leak into the log or the message.

    With ``track_peak_mem`` the process tree's resident memory is sampled while
    output streams (at most every 2s — the prune prints progress continuously),
    and the peak is returned in GB. Returns None when not tracking or unmeasurable.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    peak_gb = 0.0
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n$ {_display(argv)}\n")
        log.flush()
        try:
            proc = subprocess.Popen(  # noqa: S603 - list argv, no shell
                argv,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as e:
            raise PrerequisiteError(
                f"Cannot run '{argv[0]}': executable not found. Install it and re-run."
            ) from e
        handle = None
        if track_peak_mem:
            try:
                import psutil  # noqa: PLC0415

                handle = psutil.Process(proc.pid)
            except Exception:  # noqa: BLE001 - sampling is best-effort, never fatal
                handle = None
        last_sample = 0.0
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if handle is not None and time.monotonic() - last_sample >= 2.0:
                last_sample = time.monotonic()
                peak_gb = max(peak_gb, _tree_rss_gb(handle))
        code = proc.wait()
    if code != 0:
        raise PruneError(
            f"Command failed (exit {code}): {_display(argv)}\nFull output: {log_path}"
        )
    if not track_peak_mem or peak_gb <= 0:
        return None
    return round(peak_gb, 3)


def _is_hf_checkpoint(d: Path) -> bool:
    """A directory only counts as a checkpoint when it carries BOTH a config.json
    and at least one weight shard — a bare config.json is a config file, not a model."""
    return (d / "config.json").exists() and any(d.glob("*.safetensors"))


def _find_pruned_checkpoint(root: Path) -> Path:
    """Locate the pruned HF checkpoint under *root* via glob.

    REAP writes ``results_dir/pruned_models/{method}-{seed}-{ratio}`` but the
    exact results-dir mechanism is UNCONFIRMED (research brief), so we search:
    prefer ``config.json`` parents living under a ``pruned_models`` directory.

    Outside ``pruned_models`` the search is deliberately strict: a fallback
    candidate must look like a real checkpoint (config.json AND at least one
    ``*.safetensors``), otherwise the newest random ``config.json`` in the reap
    clone (e.g. a tokenizer or hydra config folder) would be shipped as "the
    pruned model" and only fail much later, in llama.cpp.
    """
    candidates = [p.parent for p in root.rglob("config.json")]
    preferred = [c for c in candidates if "pruned_models" in c.parts]
    pool = preferred or [c for c in candidates if _is_hf_checkpoint(c)]
    if not pool:
        seen = "\n".join(f"  - {c}" for c in sorted(candidates)[:10]) or "  (none)"
        raise PruneError(
            f"REAP finished but no pruned checkpoint was found under {root}.\n"
            "A checkpoint is a directory holding config.json AND at least one *.safetensors "
            "(or any directory under 'pruned_models/').\n"
            f"Directories with a config.json that were rejected:\n{seen}\n"
            "The repo's results layout may have changed -- check the run log, find the "
            "'pruned_models' output directory manually, and copy it into the workspace."
        )
    return max(pool, key=lambda p: p.stat().st_mtime)


class ExecutionProfile(ABC):
    """Common interface: produce a local pruned HF checkpoint directory.

    After :meth:`run_prune`, ``peak_mem_gb`` carries the measured peak resident
    memory of the prune when this machine actually hosted it, else None with
    ``peak_mem_note`` stating WHY it is missing (PRD FR-2.3 provenance: an absent
    number must be explained, never silently null).
    """

    name: str = "base"

    #: peak host RSS of the prune, GB — set by profiles that run it locally.
    peak_mem_gb: float | None = None
    #: why peak_mem_gb is None (recorded in the manifest's versions map).
    peak_mem_note: str = "not measured"

    @abstractmethod
    def run_prune(self, spec: SweepSpec, retention: float, dataset_dir: Path, out_dir: Path) -> Path:
        """Run (or fabricate) one REAP prune at *retention*.

        ``dataset_dir`` is the messages-column dataset folder from
        :func:`reaplab.prune.reap_cmd.calib_to_dataset_dir`. Returns the local
        checkpoint directory (normally ``out_dir``).
        """


class MockProfile(ExecutionProfile):
    """Instant fake prune for tests and the offline demo.

    Fabricates a checkpoint dir with a ``config.json`` whose ``num_experts``
    is the base count scaled by retention (128 -> 64 at 0.5), a few KB of
    deterministic ``model.safetensors`` bytes, and a tokenizer stub.
    """

    name = "mock"
    peak_mem_note = "not measured: the mock profile fabricates the checkpoint (no prune runs)"

    def __init__(self, base_experts: int = MOCK_BASE_EXPERTS):
        self.base_experts = base_experts

    def run_prune(self, spec: SweepSpec, retention: float, dataset_dir: Path, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        num_experts = max(1, int(round(self.base_experts * retention)))
        config = {
            "model_type": "qwen3_moe",
            "num_experts": num_experts,
            "num_experts_per_tok": min(8, num_experts),
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "base_num_experts": self.base_experts,
            "retention": retention,
            "reaplab_mock": True,
        }
        (out_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        seed = f"{spec.model_id}|{retention_tag(retention)}|{spec.seeds[0] if spec.seeds else 42}"
        (out_dir / "model.safetensors").write_bytes(deterministic_bytes(seed, 4096))
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "MockTokenizer", "reaplab_mock": True}, indent=2),
            encoding="utf-8",
        )
        return out_dir


class LocalOffloadProfile(ExecutionProfile):
    """Run REAP on this machine (GPU + system-RAM offload). Slow but free.

    Requires: a Linux-like host, git, uv, and the model pre-downloaded into the HF
    cache. Every missing prerequisite raises :class:`PrerequisiteError` with the
    install command; nothing surfaces as a raw stack trace.

    NOT supported on Windows: reap's locked environment pins vllm (and the CUDA
    torch build it needs), which has no Windows wheels -- ``uv sync`` fails during
    resolution, minutes into the run. We say so up front instead. Set
    ``REAPLAB_ALLOW_LOCAL_OFFLOAD=1`` to try anyway (e.g. inside WSL with a
    Windows-looking os.name, or once upstream ships wheels).
    """

    name = "local-offload"
    peak_mem_note = "not measured: the prune process exposed no sampleable memory"

    def __init__(self, work_dir: Path, log_dir: Path | None = None):
        self.work_dir = Path(work_dir)
        self.log_dir = Path(log_dir) if log_dir else self.work_dir

    def run_prune(self, spec: SweepSpec, retention: float, dataset_dir: Path, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        if (out_dir / "config.json").exists():
            return out_dir  # already produced (resume)

        if os.name == "nt" and os.environ.get(ALLOW_LOCAL_OFFLOAD_ENV) != "1":
            raise PrerequisiteError(
                "execution_profile: local-offload cannot run on Windows.\n"
                "REAP's pinned environment requires vllm (plus its CUDA torch build), which "
                "publishes no Windows wheels -- `uv sync` would fail during resolution.\n"
                "Pick one:\n"
                "  * prune.execution_profile: remote   -- rent an 80 GB GPU box; reap-lab "
                "generates and (with prune.remote.ssh_host) drives the whole script.\n"
                "  * prune.execution_profile: mock     -- offline dry run of the full pipeline.\n"
                "  * run reap-lab from Linux/WSL with the GPU passed through.\n"
                f"Override (at your own risk): set {ALLOW_LOCAL_OFFLOAD_ENV}=1."
            )

        git = shutil.which("git")
        if not git:
            raise PrerequisiteError(
                "git is required to clone the REAP repo but was not found on PATH.\n"
                "Install it: winget install Git.Git  (then reopen the terminal)."
            )
        uv = shutil.which("uv")
        if not uv:
            raise PrerequisiteError(
                "uv is required to build the REAP environment but was not found on PATH.\n"
                "Install it: winget install astral-sh.uv  (or: pip install uv)."
            )
        resolve_hf_model_dir(spec.model_id)  # raises with `hf download` guidance

        self.work_dir.mkdir(parents=True, exist_ok=True)
        repo = self.work_dir / "reap"
        log = self.log_dir / f"prune-{retention_tag(retention)}.log"
        if not (repo / ".git").exists():
            _run_logged(
                [git, "clone", "--recursive", spec.prune.reap_repo, str(repo)],
                cwd=self.work_dir,
                log_path=log,
            )
        _run_logged([git, "checkout", spec.prune.reap_commit], cwd=repo, log_path=log)
        _run_logged([git, "submodule", "update", "--init", "--recursive"], cwd=repo, log_path=log)
        # reap is uv-managed; scripts/build.sh is bash-only so on Windows we
        # build the env directly with uv.
        _run_logged([uv, "sync"], cwd=repo, log_path=log)
        # REAP hands the path to HF `load_dataset()`; a Windows drive path with
        # backslashes is mangled by its glob/URI handling, so pass POSIX form
        # ("C:/Users/.../dataset"), which pathlib and datasets both accept.
        dataset_arg = Path(dataset_dir).as_posix()
        prune_cmd = [uv, "run", *build_prune_command(spec, retention, dataset_arg)]
        peak = _run_logged(prune_cmd, cwd=repo, log_path=log, track_peak_mem=True)
        if peak is not None:
            self.peak_mem_gb = peak

        found = _find_pruned_checkpoint(repo)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(found), str(out_dir))
        return out_dir


def build_remote_script(spec: SweepSpec, retention: float) -> str:
    """Self-contained provisioning bash script for a rented 80 GB GPU box.

    Clones reap at the pinned commit, builds via ``scripts/build.sh`` (uv),
    pre-downloads the model (HF_TOKEN passed through the environment, never argv),
    expects the dataset folder at ``$WORK/dataset``, runs the prune, and tars the
    pruned output for download.

    The budget kill switch wraps the WHOLE script, not just the prune: the box bills
    by wall clock, and a hung clone, a stuck ``uv sync``, or a 60 GB model download
    that stalls costs exactly as much as a hung prune. The script re-executes itself
    under ``timeout`` (guard: ``REAP_LAB_TIMED``) and prints a loud message on 124.

    It deliberately does NOT power the box down: on most rentals a shutdown from
    inside the guest keeps billing the (still-allocated) instance while destroying
    the shell you would use to fetch the result. Destroying the instance is step 5
    of the instructions, and it is the user's call.
    """
    rtag = retention_tag(retention)
    seconds = budget_timeout_seconds(spec)
    remote = spec.prune.remote
    prune_line = _shell_join(["uv", "run", *build_prune_command(spec, retention, "$DATASET")])
    hours = seconds / 3600
    return f"""#!/usr/bin/env bash
# reap-lab remote prune -- generated, self-contained. Retention {retention:g} ({rtag}).
# Budget kill switch: ${remote.budget_usd:g} at ${remote.usd_per_hour:g}/h -> {seconds}s ({hours:.1f}h)
# covering the ENTIRE run (clone + build + model download + prune + package).
set -euo pipefail

# --- budget kill switch: re-exec the whole script under `timeout` -------------
if [ "${{REAP_LAB_TIMED:-0}}" != "1" ]; then
  export REAP_LAB_TIMED=1
  set +e
  timeout -s TERM -k 60s {seconds}s bash "$0" "$@"
  code=$?
  set -e
  if [ $code -eq 124 ] || [ $code -eq 137 ]; then
    # single-quoted: a literal '$75' would otherwise expand as the shell variable $7
    echo '' >&2
    echo '=====================================================================' >&2
    echo ' BUDGET KILL SWITCH: {seconds}s ({hours:.1f}h) elapsed -- run TERMINATED.' >&2
    echo ' That is {remote.budget_usd:g} USD at {remote.usd_per_hour:g} USD/h (prune.remote.budget_usd).' >&2
    echo ' Nothing was powered off: DESTROY THE INSTANCE YOURSELF to stop billing.' >&2
    echo ' Then either raise prune.remote.budget_usd or rent a faster 80 GB box.' >&2
    echo '=====================================================================' >&2
  fi
  exit $code
fi

WORK="${{REAP_WORK:-$HOME/{REMOTE_WORKDIR}}}"
mkdir -p "$WORK"
cd "$WORK"

echo "== [1/6] clone reap at pinned commit {spec.prune.reap_commit}"
if [ ! -d reap/.git ]; then
  git clone --recursive {spec.prune.reap_repo} reap
fi
cd reap
git checkout {spec.prune.reap_commit}
git submodule update --init --recursive

echo "== [2/6] build environment (uv)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
bash scripts/build.sh

echo "== [3/6] pre-download model into the HF cache (HF_TOKEN read from the environment)"
export HF_TOKEN="${{HF_TOKEN:-}}"
uv run hf download {spec.model_id} || uv run huggingface-cli download {spec.model_id}

echo "== [4/6] verify calibration dataset folder"
DATASET="$WORK/dataset"
if [ ! -f "$DATASET/{DATASET_FILENAME}" ]; then
  echo "ERROR: dataset missing -- upload the dataset folder to $DATASET first." >&2
  exit 2
fi

echo "== [5/6] prune (compression-ratio {format_ratio(retention)})"
{prune_line}

echo "== [6/6] package pruned checkpoint"
OUT=$(find . -type d -name pruned_models | head -n 1)
if [ -z "$OUT" ]; then
  echo "ERROR: no pruned_models directory found after the run." >&2
  exit 3
fi
tar czf "$WORK/pruned_{rtag}.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "DONE: $WORK/pruned_{rtag}.tar.gz"
"""


class RemoteProfile(ExecutionProfile):
    """Scripted remote prune (PRD FR-2.2): provision -> prune -> download.

    With ``spec.prune.remote.ssh_host`` set, the whole flow runs through
    ssh/scp subprocesses. Without it, the script + numbered instructions are
    written to the workspace and :class:`NeedsManualStep` explains exactly
    what to run; once the tarball lands at the expected path, re-running the
    sweep picks it up automatically.
    """

    name = "remote"
    peak_mem_note = (
        "not measured: the prune ran on the remote GPU box, which this process never "
        "observes (only the tarball comes back)"
    )

    def __init__(self, work_dir: Path, log_dir: Path | None = None):
        self.work_dir = Path(work_dir)
        self.log_dir = Path(log_dir) if log_dir else self.work_dir

    # -- public entry ------------------------------------------------------

    def run_prune(self, spec: SweepSpec, retention: float, dataset_dir: Path, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        dataset_dir = Path(dataset_dir)
        rtag = retention_tag(retention)
        if (out_dir / "config.json").exists():
            return out_dir  # already downloaded + extracted (resume)
        if not (dataset_dir / DATASET_FILENAME).exists():
            raise PruneError(
                f"Dataset folder is missing {DATASET_FILENAME}: {dataset_dir}\n"
                "Run calibration conversion first (calib_to_dataset_dir)."
            )

        self.work_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.work_dir / f"prune_remote_{rtag}.sh"
        script_path.write_text(build_remote_script(spec, retention), encoding="utf-8", newline="\n")
        tar_local = self.work_dir / f"pruned_{rtag}.tar.gz"

        host = spec.prune.remote.ssh_host
        if host:
            self._run_over_ssh(spec, host, script_path, dataset_dir, tar_local, rtag)
        elif not tar_local.exists():
            instructions = self._write_instructions(spec, retention, script_path, dataset_dir, tar_local)
            raise NeedsManualStep(
                f"Remote prune for {rtag} needs one manual step (no prune.remote.ssh_host configured).\n"
                f"Everything is prepared:\n"
                f"  script:       {script_path}\n"
                f"  dataset:      {dataset_dir}\n"
                f"  instructions: {instructions}\n"
                f"Run the numbered steps in the instructions file on a rented GPU box "
                f"({spec.prune.remote.gpu_hint}), place the downloaded tarball at:\n"
                f"  {tar_local}\n"
                f"then re-run the sweep -- it resumes from there automatically.\n"
                f"Tip: set prune.remote.ssh_host (user@host) in the sweep YAML to automate all of this."
            )
        return self._extract_tar(tar_local, out_dir, rtag)

    # -- ssh orchestration ---------------------------------------------------

    def _run_over_ssh(
        self,
        spec: SweepSpec,
        host: str,
        script_path: Path,
        dataset_dir: Path,
        tar_local: Path,
        rtag: str,
    ) -> None:
        """Upload dataset + script, execute remotely, download the tarball.

        Plain OpenSSH subprocesses (list argv). The remote run gets a timeout
        of the budget kill switch plus margin.

        HF_TOKEN never touches a command line. When set, the run step reads it from
        stdin (``IFS= read -r HF_TOKEN``) and exports it inside the remote shell, so
        the secret stays out of: this machine's argv, the on-disk log, any raised
        error, and the REMOTE box's process table (``ps`` shows the ssh command line
        of every user).
        """
        script_name = f"prune_remote_{rtag}.sh"
        token = os.environ.get("HF_TOKEN")
        if token:
            run_cmd = (
                f"IFS= read -r HF_TOKEN; export HF_TOKEN; bash {REMOTE_WORKDIR}/{script_name}"
            )
            run_input = token.strip() + "\n"
        else:
            run_cmd = f"bash {REMOTE_WORKDIR}/{script_name}"
            run_input = None
        steps: list[tuple[list[str], str, float | None, str | None]] = [
            (
                ["ssh", host, f"mkdir -p {REMOTE_WORKDIR} && rm -rf {REMOTE_WORKDIR}/dataset"],
                "prepare remote work dir",
                300.0,
                None,
            ),
            (
                ["scp", str(script_path), f"{host}:{REMOTE_WORKDIR}/{script_name}"],
                "upload provisioning script",
                600.0,
                None,
            ),
            (
                ["scp", "-r", str(dataset_dir), f"{host}:{REMOTE_WORKDIR}/dataset"],
                "upload calibration dataset",
                1800.0,
                None,
            ),
            (
                ["ssh", host, run_cmd],
                "run remote prune",
                budget_timeout_seconds(spec) + 1800.0,
                run_input,
            ),
            (
                ["scp", f"{host}:{REMOTE_WORKDIR}/pruned_{rtag}.tar.gz", str(tar_local)],
                "download pruned checkpoint tarball",
                None,
                None,
            ),
        ]
        log_path = self.log_dir / f"remote-{rtag}.log"
        for argv, what, timeout, stdin_text in steps:
            self._ssh_step(
                argv, what=what, host=host, timeout=timeout, log_path=log_path, stdin_text=stdin_text
            )
        if not tar_local.exists() or tar_local.stat().st_size == 0:
            raise PruneError(
                f"Download reported success but {tar_local} is missing or empty.\n"
                f"Check {log_path} and the remote box ({host}): the tarball should be at "
                f"~/{REMOTE_WORKDIR}/pruned_{rtag}.tar.gz."
            )

    def _ssh_step(
        self,
        argv: list[str],
        *,
        what: str,
        host: str,
        timeout: float | None,
        log_path: Path,
        stdin_text: str | None = None,
    ) -> None:
        try:
            result = subprocess.run(  # noqa: S603 - list argv, no shell
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                input=stdin_text,
            )
        except FileNotFoundError as e:
            raise PrerequisiteError(
                f"'{argv[0]}' not found -- the OpenSSH client is required for remote pruning.\n"
                "On Windows: Settings > System > Optional features > add 'OpenSSH Client', "
                "or run: winget install Microsoft.OpenSSH.Beta"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise PruneError(
                f"Remote step timed out: {what} ({' '.join(argv[:2])}...).\n"
                "If this was the prune itself, the budget kill switch window was exceeded -- "
                "raise prune.remote.budget_usd or check the box is actually a fast 80 GB GPU."
            ) from e
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as log:
            log.write(f"\n$ {_display(argv)}\n")
            if getattr(result, "stdout", None):
                log.write(redact(result.stdout))
            if getattr(result, "stderr", None):
                log.write(redact(result.stderr))
        if result.returncode != 0:
            tail = ((getattr(result, "stderr", "") or "") + (getattr(result, "stdout", "") or ""))[-1500:]
            raise PruneError(
                f"Remote step failed: {what} (exit {result.returncode}).\n"
                f"Command: {_display(argv)}\n"
                f"Verify you can connect manually: ssh {host}\n"
                f"Output tail:\n{redact(tail)}\nFull log: {log_path}"
            )

    # -- tarball handling ------------------------------------------------------

    def _extract_tar(self, tar_local: Path, out_dir: Path, rtag: str) -> Path:
        if not tar_local.exists() or tar_local.stat().st_size == 0:
            raise PruneError(
                f"Pruned-checkpoint tarball missing or empty: {tar_local}\n"
                "Download it from the remote box (see the instructions file) and re-run."
            )
        extract_dir = self.work_dir / f"extract_{rtag}"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with tarfile.open(tar_local, "r:gz") as tf:
            try:
                tf.extractall(extract_dir, filter="data")
            except TypeError:  # Python < 3.11.4 lacks the filter parameter
                tf.extractall(extract_dir)  # noqa: S202 - tar produced by our own script
        found = _find_pruned_checkpoint(extract_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(found), str(out_dir))
        return out_dir

    # -- manual-mode instructions ------------------------------------------------

    def _write_instructions(
        self,
        spec: SweepSpec,
        retention: float,
        script_path: Path,
        dataset_dir: Path,
        tar_local: Path,
    ) -> Path:
        rtag = retention_tag(retention)
        remote = spec.prune.remote
        seconds = budget_timeout_seconds(spec)
        script_name = f"prune_remote_{rtag}.sh"
        text = f"""# Remote prune -- manual steps ({rtag}, {spec.model_id})

No `prune.remote.ssh_host` is configured, so run these steps yourself.
Budget guard: the script kills ITSELF -- clone, build, model download, prune and
packaging together -- after {seconds}s (= ${remote.budget_usd:g} / ${remote.usd_per_hour:g} per hour).
It does not power the box down: destroying the instance (step 5) is what stops billing.
Provider hint: {remote.provider}, recommended box: {remote.gpu_hint}.

1. Rent a GPU instance (1x 80 GB, e.g. A100/H100) with SSH access.
   Vast.ai / RunPod / Lambda all work; see docs/REMOTE_GPU.md.

2. Upload the script and the calibration dataset (replace user@HOST):

       ssh user@HOST "mkdir -p {REMOTE_WORKDIR}"
       scp "{script_path}" user@HOST:{REMOTE_WORKDIR}/{script_name}
       scp -r "{dataset_dir}" user@HOST:{REMOTE_WORKDIR}/dataset

3. Run the prune:

       ssh user@HOST "bash {REMOTE_WORKDIR}/{script_name}"

   Gated model? Pipe the token in on stdin -- never put it on the command line
   (it would show up in `ps` on the remote box and in your shell history):

       printf '%s\\n' "$HF_TOKEN" | ssh user@HOST \\
         "IFS= read -r HF_TOKEN; export HF_TOKEN; bash {REMOTE_WORKDIR}/{script_name}"

4. Download the result to EXACTLY this path:

       scp user@HOST:{REMOTE_WORKDIR}/pruned_{rtag}.tar.gz "{tar_local}"

5. Destroy the instance (billing stops), then re-run your sweep --
   reap-lab detects the tarball and continues automatically.
"""
        path = self.work_dir / f"REMOTE_STEPS_{rtag}.md"
        path.write_text(text, encoding="utf-8", newline="\n")
        return path


def get_profile(spec: SweepSpec, work_dir: Path, log_dir: Path | None = None) -> ExecutionProfile:
    """Factory: ``spec.prune.execution_profile`` -> profile instance.

    ``work_dir`` holds clones, generated scripts, tarballs, and extraction
    scratch (the runner passes ``<workspace>/prune``).
    """
    profile = spec.prune.execution_profile
    if profile == "mock":
        return MockProfile()
    if profile == "local-offload":
        return LocalOffloadProfile(work_dir=work_dir, log_dir=log_dir)
    if profile == "remote":
        return RemoteProfile(work_dir=work_dir, log_dir=log_dir)
    raise PruneError(
        f"Unknown execution profile '{profile}'. Valid values: mock, local-offload, remote."
    )
