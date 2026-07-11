"""Provider factory: config -> concrete LLMProvider."""

from __future__ import annotations

from reaplab.core.config import ProviderCfg
from reaplab.core.providers.base import (
    JSON_ONLY_INSTRUCTION,
    LLMProvider,
    LLMResponse,
    ProviderError,
    extract_json,
)


def get_provider(cfg: ProviderCfg) -> LLMProvider:
    if cfg.kind == "mock":
        from reaplab.core.providers.mock import MockProvider

        return MockProvider(cfg)
    if cfg.kind == "claude-cli":
        from reaplab.core.providers.claude_cli import ClaudeCliProvider

        return ClaudeCliProvider(cfg)
    if cfg.kind == "openai-compat":
        from reaplab.core.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(cfg)
    if cfg.kind == "anthropic-api":
        from reaplab.core.providers.anthropic_api import AnthropicApiProvider

        return AnthropicApiProvider(cfg)
    raise ValueError(f"unknown provider kind: {cfg.kind}")


__all__ = [
    "JSON_ONLY_INSTRUCTION",
    "LLMProvider",
    "LLMResponse",
    "ProviderError",
    "extract_json",
    "get_provider",
]
