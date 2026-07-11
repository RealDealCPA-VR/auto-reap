from __future__ import annotations

from reaplab.evalharness.perf import capture_perf
from reaplab.evalharness.runners import ModelRunner, RunnerError, RunnerResponse


class StubRunner(ModelRunner):
    """Perf-test double: replays canned RunnerResponses."""

    def __init__(self, responses: list[RunnerResponse], load_time_s=None, peak_vram_mb=None):
        self._responses = responses
        self._i = 0
        self.load_time_s = load_time_s
        self.peak_vram_mb = peak_vram_mb

    def start(self, manifest, context) -> None:  # pragma: no cover - not exercised
        pass

    def stop(self) -> None:
        pass

    def complete(self, prompt, *, tools=None, max_tokens, temperature=0.0, record=None):
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


def test_uses_server_timings_when_present():
    responses = [
        RunnerResponse(text="a", latency_ms=1000,
                       timings={"prompt_per_second": 400.0, "predicted_per_second": 30.0}),
        RunnerResponse(text="b", latency_ms=1000,
                       timings={"prompt_per_second": 600.0, "predicted_per_second": 40.0}),
    ]
    pm = capture_perf(StubRunner(responses, load_time_s=7.5, peak_vram_mb=31000.0), 4096, ["p1", "p2"])
    assert pm.context == 4096
    assert pm.prefill_tps == 500.0  # mean of 400/600
    assert pm.decode_tps == 35.0  # mean of 30/40
    assert pm.load_time_s == 7.5
    assert pm.peak_vram_mb == 31000.0


def test_derives_from_token_counts_without_timings():
    responses = [
        RunnerResponse(text="a", latency_ms=2000.0, prompt_tokens=100, completion_tokens=50, timings=None),
    ]
    pm = capture_perf(StubRunner(responses), 32768, ["p1"])
    assert pm.decode_tps == 25.0  # 50 tokens / 2 s (coarse fallback)
    assert pm.prefill_tps == 50.0  # 100 tokens / 2 s
    assert pm.load_time_s is None and pm.peak_vram_mb is None


def test_all_samples_failing_yields_none_not_crash():
    class FailingRunner(StubRunner):
        def complete(self, prompt, *, tools=None, max_tokens, temperature=0.0, record=None):
            raise RunnerError("server fell over")

    pm = capture_perf(FailingRunner([]), 4096, ["p1", "p2"])
    assert pm.decode_tps is None and pm.prefill_tps is None
    assert pm.context == 4096


def test_empty_prompts_get_default():
    responses = [RunnerResponse(text="a", latency_ms=1000.0, completion_tokens=10)]
    pm = capture_perf(StubRunner(responses), 4096, ["", "   "])
    assert pm.decode_tps == 10.0  # one default prompt was substituted and used
