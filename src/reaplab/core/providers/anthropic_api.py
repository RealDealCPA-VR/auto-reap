"""Direct Anthropic Messages API provider (for users with an API key)."""

from __future__ import annotations

import os

import httpx

from reaplab.core.providers.base import (
    JSON_ONLY_INSTRUCTION,
    LLMProvider,
    LLMResponse,
    ProviderError,
)

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicApiProvider(LLMProvider):
    name = "anthropic-api"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        key_env = self.cfg.api_key_env or "ANTHROPIC_API_KEY"
        key = os.environ.get(key_env)
        if not key:
            raise ProviderError(
                f"No API key in ${key_env}. Set it, or use the 'claude-cli' provider "
                "(subscription, no key needed)."
            )
        body = f"{prompt}\n\n{JSON_ONLY_INSTRUCTION}" if json_mode else prompt
        payload: dict = {
            "model": self.cfg.model or DEFAULT_MODEL,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "messages": [{"role": "user", "content": body}],
        }
        if system:
            payload["system"] = system
        try:
            resp = httpx.post(
                API_URL,
                json=payload,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=self.cfg.timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"Anthropic API call failed: {e}") from e
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )
