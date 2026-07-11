"""Instructive error types for the pruning engine (C2).

Every error raised out of reaplab.prune tells a non-expert user exactly what to
install, download, or run next -- never a bare stack trace over a missing tool.
"""

from __future__ import annotations


class PruneError(RuntimeError):
    """Base class for all C2 failures. The message is always actionable."""


class PrerequisiteError(PruneError):
    """A required external tool, model download, or file is missing.

    The message names the missing thing and the exact command that fixes it
    (e.g. ``winget install Git.Git`` or ``hf download <model_id>``).
    """


class ToolNotFoundError(PrerequisiteError):
    """llama.cpp tooling (convert_hf_to_gguf.py / llama-quantize) could not be
    located. The message points at the GitHub release zips and the config/env
    knobs that override discovery."""


class NeedsManualStep(PruneError):
    """The remote profile prepared everything it could locally (provisioning
    script + dataset + numbered instructions) but no ``prune.remote.ssh_host``
    is configured, so the user must run the remote step themselves. The message
    lists the exact commands, then the sweep can simply be re-run."""
