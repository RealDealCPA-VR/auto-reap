"""Provider that shells out to the `claude` CLI (Claude Code print mode).

Runs inside the user's existing subscription - no API key handling. Prompt goes in
via stdin (avoids Windows command-length limits); output parsed from
`--output-format json`.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from reaplab.core.providers.base import (
    JSON_ONLY_INSTRUCTION,
    LLMProvider,
    LLMResponse,
    ProviderError,
)


class ClaudeCliProvider(LLMProvider):
    name = "claude-cli"

    def _binary(self) -> str:
        path = shutil.which("claude")
        if not path:
            raise ProviderError(
                "`claude` CLI not found on PATH. Install Claude Code (https://claude.com/claude-code) "
                "and log in, or switch this provider to 'openai-compat' / 'anthropic-api'."
            )
        return path

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        # The CLI does not expose temperature/max_tokens; both are advisory here.
        body = prompt
        if json_mode:
            body = f"{prompt}\n\n{JSON_ONLY_INSTRUCTION}"
        cmd = [self._binary(), "-p", "--output-format", "json"]
        if self.cfg.model:
            cmd += ["--model", self.cfg.model]
        if system:
            cmd += ["--append-system-prompt", system]
        try:
            proc = subprocess.run(
                cmd,
                input=body,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.cfg.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise ProviderError(f"claude CLI timed out after {self.cfg.timeout_s}s") from e
        if proc.returncode != 0:
            raise ProviderError(
                f"claude CLI exited {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
            )
        try:
            payload = json.loads(proc.stdout)
            text = payload.get("result", "")
            usage = payload.get("usage") or {}
            return LLMResponse(
                text=text,
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                raw={"session_id": payload.get("session_id"), "cost_usd": payload.get("total_cost_usd")},
            )
        except json.JSONDecodeError:
            # older CLI or plain-text mode: treat stdout as the answer
            return LLMResponse(text=proc.stdout.strip())
