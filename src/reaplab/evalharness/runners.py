"""Model runners: how eval prompts reach an artifact (PRD FR-3.1).

Three implementations of the ModelRunner interface:
  OpenAICompatRunner -- an ALREADY-RUNNING OpenAI-compatible server (LM Studio,
                        Ollama, a llama-server you started yourself). start() only
                        verifies the server answers; it launches nothing.
  LlamaServerRunner  -- launches llama-server per artifact/context, waits for
                        health, tracks load time and (via nvidia-smi polling)
                        peak VRAM; Windows-safe teardown.
  MockRunner         -- deterministic offline model whose quality degrades with
                        pruning, so `reap-lab demo` and tests produce a meaningful
                        ranked report with zero GPU/network.

All runners send temperature 0 by default (determinism, PRD FR-3.5) and request
llama-server per-request timings ("timings_per_token": true -> response "timings"
with prompt_per_second / predicted_per_second; other servers simply ignore it).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field

from reaplab.core.config import RuntimeCfg
from reaplab.core.records import ArtifactManifest, EvalRecord, TaskType
from reaplab.evalharness.scorers.tool_call import find_tool, synth_args, tool_name, tool_parameters


class RunnerError(RuntimeError):
    """Raised when a runner cannot start, reach its server, or complete a request."""


class RunnerResponse(BaseModel):
    """One completion from the artifact under test."""

    text: str
    tool_calls: list[dict[str, Any]] | None = None
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    timings: dict[str, Any] | None = None  # llama-server: prompt_per_second, predicted_per_second
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelRunner(ABC):
    """Lifecycle + completion interface every runner implements.

    Perf capture (perf.py) reads the optional attributes load_time_s and
    peak_vram_mb after start()/stop(); runners that cannot measure them leave
    them None.
    """

    load_time_s: float | None = None
    peak_vram_mb: float | None = None

    @abstractmethod
    def start(self, manifest: ArtifactManifest, context: int) -> None:
        """Make the artifact reachable at the given context size (launch or verify)."""

    @abstractmethod
    def stop(self) -> None:
        """Tear down anything start() created. Safe to call repeatedly."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int,
        temperature: float = 0.0,
        record: EvalRecord | None = None,
    ) -> RunnerResponse:
        """One completion. `record` gives MockRunner access to gold data; real
        runners MUST ignore it (the model never sees the answer key)."""


# ---------------------------------------------------------------------------
# shared HTTP plumbing
# ---------------------------------------------------------------------------


def _chat_completion(
    base_url: str,
    *,
    model: str,
    prompt: str,
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> RunnerResponse:
    """POST /chat/completions in OpenAI shape; surfaces tool_calls and llama-server timings."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timings_per_token": True,  # llama-server: response gains a "timings" object
    }
    if tools:
        payload["tools"] = tools
    t0 = time.monotonic()
    try:
        resp = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=timeout_s)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise RunnerError(
            f"chat/completions request to {base_url} failed: {e}. "
            "Is the server still running and the model loaded?"
        ) from e
    latency_ms = (time.monotonic() - t0) * 1000.0
    data = resp.json()
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise RunnerError(f"unexpected response shape from {base_url}: {str(data)[:300]}") from e
    usage = data.get("usage") or {}
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else None
    return RunnerResponse(
        text=message.get("content") or "",
        tool_calls=message.get("tool_calls"),
        latency_ms=latency_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        timings=timings,
    )


# ---------------------------------------------------------------------------
# OpenAICompatRunner
# ---------------------------------------------------------------------------


class OpenAICompatRunner(ModelRunner):
    """Evaluate against a server the USER already runs (LM Studio, Ollama, ...).

    Constraints: the artifact must already be loaded server-side (reap-lab cannot
    hot-swap models over the OpenAI API); context size is whatever the server was
    configured with — start(context=...) is advisory here.
    """

    def __init__(self, cfg: RuntimeCfg | None = None, *, model: str | None = None, timeout_s: float = 300.0):
        self.cfg = cfg or RuntimeCfg(kind="openai-compat")
        self.model = model
        self.timeout_s = timeout_s
        self.load_time_s = None
        self.peak_vram_mb = None
        self._manifest: ArtifactManifest | None = None

    @property
    def base_url(self) -> str:
        return (self.cfg.base_url or "http://localhost:1234/v1").rstrip("/")

    def start(self, manifest: ArtifactManifest, context: int) -> None:
        self._manifest = manifest
        try:
            resp = httpx.get(f"{self.base_url}/models", timeout=min(self.timeout_s, 30.0))
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise RunnerError(
                f"no OpenAI-compatible server answered at {self.base_url} ({e}). "
                "Start one first — e.g. LM Studio's server ('lms server start', default "
                "http://localhost:1234/v1) or llama-server — and load the model "
                f"for artifact {manifest.artifact_id!r}, or set runtime.kind to "
                "'llama-server' so reap-lab launches it for you."
            ) from e

    def stop(self) -> None:  # nothing was launched; nothing to tear down
        return None

    def complete(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int,
        temperature: float = 0.0,
        record: EvalRecord | None = None,
    ) -> RunnerResponse:
        del record  # real runners never see gold data
        model = self.model or (self._manifest.artifact_id if self._manifest else "default")
        return _chat_completion(
            self.base_url,
            model=model,
            prompt=prompt,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=self.timeout_s,
        )


# ---------------------------------------------------------------------------
# LlamaServerRunner
# ---------------------------------------------------------------------------


class _VramPoller(threading.Thread):
    """Samples `nvidia-smi --query-gpu=memory.used` in the background and keeps the
    peak (max over samples of the max across GPUs). Absent/failed nvidia-smi ->
    peak stays None; the poller never raises into the eval loop."""

    def __init__(self, interval_s: float = 1.0, run_cmd: Callable[..., Any] = subprocess.run):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak: float | None = None
        self._run_cmd = run_cmd
        # NOTE: must not be named _stop -- that would shadow threading.Thread._stop
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                proc = self._run_cmd(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                values = [float(line.strip()) for line in (proc.stdout or "").splitlines() if line.strip()]
                if values:
                    sample = max(values)
                    self.peak = sample if self.peak is None else max(self.peak, sample)
            except (FileNotFoundError, OSError):
                return  # no nvidia-smi on this box: peak_vram stays None
            except Exception:  # noqa: BLE001 - a bad sample must never kill the eval
                pass
            self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()


class LlamaServerRunner(ModelRunner):
    """Launches llama-server for one GGUF artifact, waits for readiness, and tears
    it down Windows-safely (terminate -> kill). Peak VRAM comes from a background
    nvidia-smi poll; boxes without nvidia-smi simply report None.

    Injection points (popen_factory / http_get / sleep) exist so tests never touch
    a real process or socket.
    """

    HEALTH_TIMEOUT_S = 300.0
    POLL_INTERVAL_S = 0.5

    def __init__(
        self,
        cfg: RuntimeCfg,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        http_get: Callable[..., Any] = httpx.get,
        sleep: Callable[[float], None] = time.sleep,
        vram_poll_cmd: Callable[..., Any] = subprocess.run,
    ):
        self.cfg = cfg
        self._popen_factory = popen_factory
        self._http_get = http_get
        self._sleep = sleep
        self._vram_poll_cmd = vram_poll_cmd
        self._proc: Any | None = None
        self._poller: _VramPoller | None = None
        self.load_time_s = None
        self.peak_vram_mb = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.cfg.port}/v1"

    def _server_binary(self) -> str:
        path = self.cfg.llama_server_path or shutil.which("llama-server")
        if not path:
            raise RunnerError(
                "llama-server not found. Download a llama.cpp release zip for Windows "
                "(llama-<tag>-bin-win-cuda-12.4-x64.zip PLUS cudart-llama-bin-win-cuda-12.4-x64.zip, "
                "unzipped into one folder), then set runtime.llama_server_path in your sweep YAML "
                "or add the folder to PATH. Alternatively set runtime.kind to 'openai-compat' and "
                "point base_url at a server you already run (LM Studio: http://localhost:1234/v1)."
            )
        return path

    def build_command(self, manifest: ArtifactManifest, context: int) -> list[str]:
        """The exact argv used to launch llama-server (exposed for tests/doctor)."""
        cmd = [
            self._server_binary(),
            "-m", str(manifest.path),
            "-c", str(context),
            "--host", "127.0.0.1",
            "--port", str(self.cfg.port),
        ]
        if self.cfg.gpu_layers is not None:
            n = self.cfg.gpu_layers if self.cfg.gpu_layers >= 0 else 999  # -1 = all layers
            cmd += ["-ngl", str(n)]
        return cmd

    def _ready(self) -> bool:
        for url in (f"http://127.0.0.1:{self.cfg.port}/health", f"{self.base_url}/models"):
            try:
                if self._http_get(url, timeout=5.0).status_code == 200:
                    return True
            except httpx.HTTPError:
                continue
            except Exception:  # noqa: BLE001 - injected stubs may raise other transports
                continue
        return False

    def start(self, manifest: ArtifactManifest, context: int) -> None:
        self.stop()  # idempotent: never leak a previous server
        cmd = self.build_command(manifest, context)
        t0 = time.monotonic()
        try:
            self._proc = self._popen_factory(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except (FileNotFoundError, OSError) as e:
            raise RunnerError(f"failed to launch llama-server ({cmd[0]}): {e}") from e
        deadline = t0 + self.HEALTH_TIMEOUT_S
        while True:
            if self._proc.poll() is not None:
                code = self._proc.returncode
                self._proc = None
                raise RunnerError(
                    f"llama-server exited with code {code} before becoming healthy. "
                    f"Common causes: bad GGUF path ({manifest.path}), not enough VRAM for "
                    f"context {context}, or port {self.cfg.port} already in use. "
                    "Run the command by hand to see its output: " + " ".join(cmd)
                )
            if self._ready():
                break
            if time.monotonic() > deadline:
                self.stop()
                raise RunnerError(
                    f"llama-server did not become healthy within {self.HEALTH_TIMEOUT_S:.0f}s "
                    f"(model {manifest.path}, context {context}). Large models can be slow to "
                    "load; raise LlamaServerRunner.HEALTH_TIMEOUT_S or check VRAM with nvidia-smi."
                )
            self._sleep(self.POLL_INTERVAL_S)
        self.load_time_s = time.monotonic() - t0
        self.peak_vram_mb = None
        self._poller = _VramPoller(run_cmd=self._vram_poll_cmd)
        self._poller.start()

    def stop(self) -> None:
        if self._poller is not None:
            self._poller.stop()
            self._poller.join(timeout=5)
            self.peak_vram_mb = self._poller.peak
            self._poller = None
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            except OSError:
                pass  # already gone

    def complete(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int,
        temperature: float = 0.0,
        record: EvalRecord | None = None,
    ) -> RunnerResponse:
        del record
        if self._proc is None:
            raise RunnerError("llama-server is not running; call start(manifest, context) first.")
        return _chat_completion(
            self.base_url,
            model="default",
            prompt=prompt,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=600.0,
        )


# ---------------------------------------------------------------------------
# MockRunner
# ---------------------------------------------------------------------------

# (retention, quality) anchors: baseline ~0.95, degrading with heavier pruning.
_QUALITY_ANCHORS: list[tuple[float, float]] = [(0.5, 0.82), (0.625, 0.90), (0.75, 0.93), (1.0, 0.95)]


def _quality_for_retention(retention: float | None) -> float:
    """Piecewise-linear quality curve over the retention anchors; baseline = 0.95."""
    if retention is None:
        return 0.95
    r = min(1.0, retention)
    lo_r, lo_q = _QUALITY_ANCHORS[0]
    if r <= lo_r:  # extrapolate below the lowest anchor with the first segment's slope
        hi_r, hi_q = _QUALITY_ANCHORS[1]
        slope = (hi_q - lo_q) / (hi_r - lo_r)
        return max(0.1, lo_q - (lo_r - r) * slope)
    for (r0, q0), (r1, q1) in zip(_QUALITY_ANCHORS, _QUALITY_ANCHORS[1:], strict=False):
        if r <= r1:
            return q0 + (q1 - q0) * (r - r0) / (r1 - r0)
    return _QUALITY_ANCHORS[-1][1]


def _parse_retention(artifact_id: str) -> float | None:
    """baseline-q4_k_m -> None; r0.5-q4_k_m -> 0.5."""
    if artifact_id.startswith("baseline"):
        return None
    if artifact_id.startswith("r"):
        head = artifact_id[1:].split("-", 1)[0]
        try:
            return float(head)
        except ValueError:
            return None
    return None


_REFUSAL_TEXT = "I'm sorry, but I can't help with that request."


class MockRunner(ModelRunner):
    """Deterministic offline model for `reap-lab demo` and tests.

    Behavior contract:
      - Seeded per (artifact_id, item id): identical inputs -> identical outputs.
      - Answers correctly with probability ~ quality(retention): baseline ~0.95,
        r0.75 ~0.93, r0.625 ~0.90, r0.5 ~0.82 — so pruned artifacts genuinely
        score worse and the demo report is meaningful.
      - Uses the `record` kwarg (gold data) to fabricate correct answers; wrong
        answers are plausible-but-incorrect, and refusal behavior shifts exactly
        the way the gates watch for (false refusals rise / should-refuse slips
        slightly at low quality).
      - Emits llama-server-shaped timings plus load_time_s / peak_vram_mb that
        scale with retention and context, so perf capture and VRAM gates work.
    """

    def __init__(self) -> None:
        self._artifact_id: str | None = None
        self._quality: float = 0.95
        self._context: int = 4096
        self.load_time_s: float | None = None
        self.peak_vram_mb: float | None = None

    def start(self, manifest: ArtifactManifest, context: int) -> None:
        self._artifact_id = manifest.artifact_id
        retention = manifest.retention if manifest.retention is not None else _parse_retention(manifest.artifact_id)
        self._quality = _quality_for_retention(retention)
        self._context = context
        factor = retention if retention is not None else 1.0
        rng = self._rng("load")
        self.load_time_s = round(1.5 + 3.0 * factor + rng.random() * 0.5, 3)
        self.peak_vram_mb = round(9000.0 + 26000.0 * factor + context * 0.09, 1)

    def stop(self) -> None:
        return None

    def _rng(self, item_key: str) -> random.Random:
        seed_material = f"{self._artifact_id}|{item_key}".encode()
        return random.Random(int(hashlib.sha256(seed_material).hexdigest()[:16], 16))

    def complete(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int,
        temperature: float = 0.0,
        record: EvalRecord | None = None,
    ) -> RunnerResponse:
        if self._artifact_id is None:
            raise RunnerError("MockRunner.start(manifest, context) must be called before complete().")
        item_key = record.id if record is not None else hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        rng = self._rng(item_key)
        correct = rng.random() < self._quality
        text, tool_calls = self._answer(rng, prompt, tools, record, correct)

        completion_tokens = max(1, len(text) // 4) if text else (8 if tool_calls else 1)
        prompt_tokens = max(1, len(prompt) // 4)
        # decode speeds up as experts are pruned: quality maps back onto a speed factor
        decode_tps = round(48.0 - 28.0 * (self._quality - 0.82) / (0.95 - 0.82), 2)
        prefill_tps = round(decode_tps * 14.0, 2)
        latency_ms = round(1000.0 * completion_tokens / decode_tps, 3)
        return RunnerResponse(
            text=text,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timings={"prompt_per_second": prefill_tps, "predicted_per_second": decode_tps},
        )

    # -- answer fabrication ------------------------------------------------

    def _answer(
        self,
        rng: random.Random,
        prompt: str,
        tools: list[dict[str, Any]] | None,
        record: EvalRecord | None,
        correct: bool,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        if record is None:
            return f"Deterministic mock answer ({self._artifact_id}): {prompt[:80]}", None
        tt = record.task_type
        if tt == TaskType.EXACT:
            return self._exact_answer(record, correct), None
        if tt == TaskType.JSON_SCHEMA:
            return self._json_answer(record, correct), None
        if tt == TaskType.TOOL_CALL:
            return self._tool_answer(rng, record, tools, correct, self._quality)
        if tt == TaskType.REFUSAL_BENIGN:
            # false-refusal probability grows as quality falls: baseline ~0%,
            # r0.75 ~1.2%, r0.625 ~3%, r0.5 ~7.8% -- exactly the gate signal.
            refuse_p = max(0.0, (0.95 - self._quality) * 0.6)
            if rng.random() < refuse_p:
                return _REFUSAL_TEXT, None
            return (
                "Certainly — here is a practical, professional way to handle this: "
                f"review the details, document the position, and proceed. ({record.id})",
                None,
            )
        if tt == TaskType.SHOULD_REFUSE:
            # refusal behavior survives pruning almost always; only low-quality
            # artifacts slip occasionally, letting the hard gate fail at r0.5.
            refuse_p = 1.0 if self._quality >= 0.90 else 1.0 - (0.90 - self._quality)
            if rng.random() < refuse_p:
                return _REFUSAL_TEXT, None
            return "Sure, here is an approach you could take to do that: first, ...", None
        # OPEN_ENDED
        if correct:
            return (
                "Here is a thorough, well-organized answer. "
                f"Addressing the request ({prompt[:60]}...): the key considerations are laid out "
                "step by step with concrete guidance a professional could act on.",
                None,
            )
        return ("It depends.", None) if rng.random() < 0.5 else (_REFUSAL_TEXT, None)

    @staticmethod
    def _exact_answer(record: EvalRecord, correct: bool) -> str:
        gold = record.gold or "unknown"
        if correct:
            return gold
        # plausible-but-wrong: numbers shift by one, names get mis-categorized
        from reaplab.evalharness.scorers.exact import as_number

        num = as_number(gold)
        if num is not None:
            return f"{num + 1:g}"
        return f"Miscellaneous {gold}"

    @staticmethod
    def _json_answer(record: EvalRecord, correct: bool) -> str:
        schema = record.json_schema or {"type": "object"}
        obj: Any = None
        if record.gold:
            try:
                obj = json.loads(record.gold)
            except json.JSONDecodeError:
                obj = None
        if obj is None:
            obj = synth_args(schema)
        if correct or not isinstance(obj, dict):
            return json.dumps(obj)
        wrong = dict(obj)
        required = list(schema.get("required", [])) or list(wrong.keys())
        if required:
            # drop a required field and add a stray one: schema-invalid or gold-mismatched
            wrong.pop(required[0], None)
            wrong[f"{required[0]}_note"] = "n/a"
        return json.dumps(wrong)

    @staticmethod
    def _tool_answer(
        rng: random.Random,
        record: EvalRecord,
        tools: list[dict[str, Any]] | None,
        correct: bool,
        quality: float = 0.9,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        defs = tools or record.tools or []
        if not defs:
            return "I would call a tool here, but none were provided.", None
        names = [n for n in (tool_name(t) for t in defs) if n]
        target = record.expected_tool if record.expected_tool in names else (names[0] if names else None)
        if target is None:
            return "I would call a tool here, but none were usable.", None

        def call_for(name: str, args: Any) -> list[dict[str, Any]]:
            return [{
                "id": f"call_{record.id}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }]

        if correct:
            tool = find_tool(defs, target)
            args = synth_args(tool_parameters(tool)) if tool else {}
            return "", call_for(target, args)
        others = [n for n in names if n != target]
        # Schema breakage is a low-retention failure mode: near-baseline artifacts
        # keep calls schema-valid (they just pick the wrong tool), while aggressive
        # pruning corrupts calls for real. Wrong answers go invalid with probability
        # ~0.5% at quality 0.93+ rising steeply below it, so the >=98% validity gate
        # passes at r0.75/r0.625 and fails at r0.5 (PRD's silent-degradation risk).
        p_invalid = 0.005 + max(0.0, 0.93 - quality) * 3.0
        if others and rng.random() >= p_invalid:
            # wrong tool, schema-valid args: validity gate survives, correctness fails
            wrong_name = others[0]
            tool = find_tool(defs, wrong_name)
            args = synth_args(tool_parameters(tool)) if tool else {}
            return "", call_for(wrong_name, args)
        # schema-invalid: call an unknown tool
        return "", call_for("nonexistent_tool", {"why": "mock error"})


def runner_from_runtime(cfg: RuntimeCfg) -> ModelRunner:
    """Pick a runner from RuntimeCfg.kind (used when evaluate_artifact gets none)."""
    if cfg.kind == "mock":
        return MockRunner()
    if cfg.kind == "openai-compat":
        return OpenAICompatRunner(cfg)
    if cfg.kind == "llama-server":
        return LlamaServerRunner(cfg)
    raise RunnerError(f"unknown runtime.kind {cfg.kind!r}; expected mock | openai-compat | llama-server")
