from __future__ import annotations

import httpx
import pytest
import respx

from reaplab.core.config import ProviderCfg
from reaplab.core.providers import ProviderError, extract_json, get_provider
from reaplab.core.providers.mock import MockProvider
from reaplab.core.providers.openai_compat import OpenAICompatProvider


def test_factory_dispatch():
    assert isinstance(get_provider(ProviderCfg(kind="mock")), MockProvider)
    assert isinstance(get_provider(ProviderCfg(kind="openai-compat")), OpenAICompatProvider)


def test_mock_deterministic():
    p = get_provider(ProviderCfg(kind="mock"))
    a = p.complete("generate a prompt about depreciation")
    b = p.complete("generate a prompt about depreciation")
    assert a.text == b.text
    assert p.complete("different").text != a.text


def test_mock_canned_responses():
    p = get_provider(ProviderCfg(kind="mock", extra={"responses": {"REFUSAL_CHECK": "I can't help with that."}}))
    assert "can't help" in p.complete("please do REFUSAL_CHECK now").text


def test_mock_embeddings_stable_and_normalized():
    p = get_provider(ProviderCfg(kind="mock"))
    v1, v2 = p.embed(["same text", "same text"])
    assert v1 == v2
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here you go:\n```\n[1, 2]\n```\nAnything else?', [1, 2]),
        ('prefix text {"a": {"b": "}"}} suffix', {"a": {"b": "}"}}),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


@respx.mock
def test_openai_compat_complete():
    respx.post("http://localhost:9999/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "4"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )
    )
    p = get_provider(ProviderCfg(kind="openai-compat", base_url="http://localhost:9999/v1", model="local"))
    r = p.complete("2+2?", system="be terse")
    assert r.text == "4"
    assert r.prompt_tokens == 10


@respx.mock
def test_openai_compat_embeddings_order():
    respx.post("http://localhost:9999/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}]},
        )
    )
    p = get_provider(ProviderCfg(kind="openai-compat", base_url="http://localhost:9999/v1"))
    assert p.embed(["a", "b"]) == [[0.1], [0.2]]


@respx.mock
def test_openai_compat_http_error_wrapped():
    respx.post("http://localhost:9999/v1/chat/completions").mock(return_value=httpx.Response(500))
    p = get_provider(ProviderCfg(kind="openai-compat", base_url="http://localhost:9999/v1"))
    with pytest.raises(ProviderError):
        p.complete("hi")


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = get_provider(ProviderCfg(kind="anthropic-api"))
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        p.complete("hi")
