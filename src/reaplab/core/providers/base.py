"""LLM provider contract. datagen (generation), evalharness (judging), and the init
wizard all speak this interface; swapping providers is a config change, not a code change."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from reaplab.core.config import ProviderCfg


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised when a provider call fails after any internal retries."""


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, cfg: ProviderCfg):
        self.cfg = cfg

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """One completion. json_mode asks the model to emit a single JSON value only."""

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Optional capability; return None when the provider cannot embed."""
        return None


JSON_ONLY_INSTRUCTION = (
    "Respond with a single valid JSON value only - no markdown fences, no commentary."
)


def extract_json(text: str) -> Any:
    """Tolerant JSON extraction: strips markdown fences and leading/trailing prose."""
    import json
    import re

    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # fall back to the FIRST balanced JSON value in the text — try the opener that
    # appears earliest so "...prose... [{...}]" yields the array, not the inner object
    candidates = [(s.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if s.find(o) != -1]
    for _, opener, closer in sorted(candidates):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(s[start : i + 1])
    raise ValueError(f"no JSON found in provider response: {text[:200]!r}")
