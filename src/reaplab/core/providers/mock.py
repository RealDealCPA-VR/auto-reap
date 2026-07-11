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
        judged = self._judge_pair(prompt)
        if judged is not None:
            return LLMResponse(text=judged, prompt_tokens=len(prompt) // 4, completion_tokens=24)
        digest = hashlib.sha256((system or "").encode() + prompt.encode()).hexdigest()[:8]
        if json_mode:
            text = json.dumps({"mock": True, "digest": digest})
        else:
            text = f"[mock:{digest}] deterministic response for testing."
        return LLMResponse(text=text, prompt_tokens=len(prompt) // 4, completion_tokens=16)

    @staticmethod
    def _judge_pair(prompt: str) -> str | None:
        """Deterministic pairwise judging so the offline demo's judge discriminates.

        Recognizes the evalharness judge-prompt shape (Response A: / Response B:)
        and votes for the more substantive answer: refusals and near-empty replies
        lose, close lengths tie. Returns None for non-judge prompts."""
        if "Response A:" not in prompt or "Response B:" not in prompt:
            return None
        try:
            _, rest = prompt.split("Response A:", 1)
            a, rest = rest.split("Response B:", 1)
            b = rest.split("Which response is better?", 1)[0]
        except ValueError:
            return None

        def substance(text: str) -> int:
            t = text.strip()
            low = t.lower()
            refusal_markers = ("i can't", "i cannot", "i'm not able", "i won't", "unable to help")
            if not t or any(low.startswith(m) or m in low[:80] for m in refusal_markers):
                return 0
            return len(t)

        sa, sb = substance(a), substance(b)
        if sa == 0 and sb == 0:
            winner = "tie"
        elif min(sa, sb) > 0 and abs(sa - sb) <= max(sa, sb) * 0.25:
            winner = "tie"
        else:
            winner = "A" if sa > sb else "B"
        return json.dumps({"winner": winner, "reason": "mock heuristic: substance comparison"})

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str, dim: int = 32) -> list[float]:
        h = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        # cycle the 32 digest bytes into a unit-ish vector; identical text -> identical vector
        raw = [(h[i % len(h)] - 127.5) / 127.5 for i in range(dim)]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]
