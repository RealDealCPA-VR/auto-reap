from __future__ import annotations

import json

import httpx
import pytest
import respx

from reaplab.core.config import RuntimeCfg
from reaplab.evalharness.runners import OpenAICompatRunner, RunnerError, runner_from_runtime

BASE = "http://localhost:1234/v1"


def _runner(cfg: RuntimeCfg | None = None, **kw) -> OpenAICompatRunner:
    return OpenAICompatRunner(cfg or RuntimeCfg(kind="openai-compat", base_url=BASE), **kw)


def _models(*names: str) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": n, "object": "model"} for n in names]})


def _chat_response(content="hello", tool_calls=None, timings=None):
    body = {
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }
    if timings is not None:
        body["timings"] = timings
    return httpx.Response(200, json=body)


@respx.mock
def test_start_verifies_server(make_manifest):
    route = respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    _runner().start(make_manifest(), 4096)
    assert route.called


@respx.mock
def test_start_failure_is_instructive(make_manifest):
    respx.get(f"{BASE}/models").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(RunnerError) as exc:
        _runner().start(make_manifest(), 4096)
    msg = str(exc.value)
    assert "LM Studio" in msg and BASE in msg  # tells the user what to start


@respx.mock
def test_complete_sends_timings_flag_and_tools(make_manifest, demo_tools):
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    chat = respx.post(f"{BASE}/chat/completions").mock(
        return_value=_chat_response(
            content="",
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": '{"city": "Reno"}'}}],
            timings={"prompt_per_second": 512.5, "predicted_per_second": 41.2},
        )
    )
    runner = _runner(model="my-model")
    runner.start(make_manifest(), 4096)
    resp = runner.complete("weather in reno?", tools=demo_tools, max_tokens=64)

    payload = json.loads(chat.calls.last.request.content)
    assert payload["timings_per_token"] is True
    assert payload["tools"] == demo_tools  # passthrough untouched
    assert payload["model"] == "my-model"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 64

    assert resp.tool_calls[0]["function"]["name"] == "get_weather"
    assert resp.timings == {"prompt_per_second": 512.5, "predicted_per_second": 41.2}
    assert resp.prompt_tokens == 12 and resp.completion_tokens == 5
    assert resp.latency_ms >= 0


@respx.mock
def test_complete_without_tools_omits_param(make_manifest):
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    chat = respx.post(f"{BASE}/chat/completions").mock(return_value=_chat_response("plain answer"))
    runner = _runner()
    runner.start(make_manifest(artifact_id="r0.5-q4_k_m", kind="gguf", retention=0.5), 4096)
    resp = runner.complete("2+2?", max_tokens=16)
    payload = json.loads(chat.calls.last.request.content)
    assert "tools" not in payload
    assert payload["model"] == "r0.5-q4_k_m"  # falls back to the artifact id
    assert resp.text == "plain answer"
    assert resp.timings is None
    assert resp.tool_calls is None


@respx.mock
def test_http_error_becomes_runner_error(make_manifest):
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(500, text="boom"))
    runner = _runner()
    runner.start(make_manifest(), 4096)
    with pytest.raises(RunnerError, match="chat/completions"):
        runner.complete("hi", max_tokens=8)


@respx.mock
def test_malformed_body_becomes_runner_error(make_manifest):
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json={"nope": 1}))
    runner = _runner()
    runner.start(make_manifest(), 4096)
    with pytest.raises(RunnerError, match="unexpected response shape"):
        runner.complete("hi", max_tokens=8)


def test_stop_is_a_noop_and_record_ignored(make_manifest):
    runner = _runner()
    runner.stop()  # never raises, launches nothing
    assert runner.load_time_s is None and runner.peak_vram_mb is None


# -- runtime.model: the server-side name, verified against GET /models -------------


@respx.mock
def test_runtime_model_is_requested_and_verified(make_manifest):
    """LM Studio/Ollama route on THEIR model name, which is never our artifact id."""
    respx.get(f"{BASE}/models").mock(return_value=_models("qwen3-30b-a3b", "nomic-embed"))
    chat = respx.post(f"{BASE}/chat/completions").mock(return_value=_chat_response("ok"))
    cfg = RuntimeCfg(kind="openai-compat", base_url=BASE, model="qwen3-30b-a3b")
    runner = runner_from_runtime(cfg)  # the wiring users actually get
    assert isinstance(runner, OpenAICompatRunner)

    runner.start(make_manifest(artifact_id="r0.5-q4_k_m", kind="gguf", retention=0.5), 4096)
    runner.complete("hi", max_tokens=8)
    assert json.loads(chat.calls.last.request.content)["model"] == "qwen3-30b-a3b"


@respx.mock
def test_unserved_model_is_an_instructive_error(make_manifest):
    respx.get(f"{BASE}/models").mock(return_value=_models("qwen3-30b-a3b", "llama-3.1-8b"))
    cfg = RuntimeCfg(kind="openai-compat", base_url=BASE, model="qwen-30b")  # typo
    with pytest.raises(RunnerError) as exc:
        _runner(cfg).start(make_manifest(), 4096)
    msg = str(exc.value)
    assert "'qwen-30b'" in msg  # what we asked for
    assert "'qwen3-30b-a3b'" in msg and "'llama-3.1-8b'" in msg  # what the server has
    assert "runtime.model" in msg  # how to fix it


@respx.mock
def test_artifact_id_fallback_is_verified_too(make_manifest):
    """With no runtime.model set, the artifact id is sent — and a server that clearly
    does not serve it must say so instead of misrouting every request."""
    respx.get(f"{BASE}/models").mock(return_value=_models("qwen3-30b-a3b"))
    with pytest.raises(RunnerError, match="does not serve a model named 'r0.5-q4_k_m'"):
        _runner().start(make_manifest(artifact_id="r0.5-q4_k_m", kind="gguf", retention=0.5), 4096)


@respx.mock
def test_unenumerable_model_list_does_not_block_the_eval(make_manifest):
    """Servers that answer /models with an empty or odd body must not fail the run:
    'cannot enumerate' is not 'does not serve it'."""
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.post(f"{BASE}/chat/completions").mock(return_value=_chat_response("ok"))
    runner = _runner(model="whatever")
    runner.start(make_manifest(), 4096)  # no raise
    assert runner.served_models == []

    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, text="not json"))
    _runner(model="whatever").start(make_manifest(), 4096)  # no raise
