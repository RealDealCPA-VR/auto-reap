"""Any OpenAI-compatible endpoint: LM Studio (default), Ollama, llama-server,
OpenRouter, OpenAI. Also serves /embeddings for the embedding dedup backend."""

from __future__ import annotations

import os

import httpx

from reaplab.core.providers.base import (
    JSON_ONLY_INSTRUCTION,
    LLMProvider,
    LLMResponse,
    ProviderError,
)

DEFAULT_BASE_URL = "http://localhost:1234/v1"  # LM Studio's default server


class OpenAICompatProvider(LLMProvider):
    name = "openai-compat"

    @property
    def base_url(self) -> str:
        return (self.cfg.base_url or DEFAULT_BASE_URL).rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key_env:
            key = os.environ.get(self.cfg.api_key_env)
            if not key:
                raise ProviderError(
                    f"api_key_env={self.cfg.api_key_env!r} is set but that environment variable is empty."
                )
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        body = f"{prompt}\n\n{JSON_ONLY_INSTRUCTION}" if json_mode else prompt
        messages.append({"role": "user", "content": body})
        payload: dict = {
            "model": self.cfg.model or "default",
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        payload.update(self.cfg.extra.get("request_overrides", {}))
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.cfg.timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"openai-compat call to {self.base_url} failed: {e}") from e
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"unexpected response shape from {self.base_url}: {str(data)[:300]}") from e
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        model = self.cfg.extra.get("embedding_model") or self.cfg.model
        try:
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": model, "input": texts},
                headers=self._headers(),
                timeout=self.cfg.timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"embeddings call to {self.base_url} failed: {e}") from e
        data = resp.json()
        rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [r["embedding"] for r in rows]
