"""Deterministic offline provider for tests and `reap-lab demo`.

- complete(): returns canned responses from cfg.extra["responses"] (substring -> reply),
  else a deterministic pseudo-response derived from the prompt hash.
- embed(): stable pseudo-embeddings (same text -> same vector) so near-dup filtering
  is exercisable offline.
"""

from __future__ import annotations

import hashlib
import json

from reaplab.core.providers.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    name = "mock"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        responses = self.cfg.extra.get("responses") or {}
        for needle, reply in responses.items():
            if needle in prompt or (system and needle in system):
                return LLMResponse(text=reply, prompt_tokens=len(prompt) // 4, completion_tokens=len(reply) // 4)
        digest = hashlib.sha256((system or "").encode() + prompt.encode()).hexdigest()[:8]
        if json_mode:
            text = json.dumps({"mock": True, "digest": digest})
        else:
            text = f"[mock:{digest}] deterministic response for testing."
        return LLMResponse(text=text, prompt_tokens=len(prompt) // 4, completion_tokens=16)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str, dim: int = 32) -> list[float]:
        h = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        # cycle the 32 digest bytes into a unit-ish vector; identical text -> identical vector
        raw = [(h[i % len(h)] - 127.5) / 127.5 for i in range(dim)]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]
