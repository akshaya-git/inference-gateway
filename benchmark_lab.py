"""
Multi-axis benchmark lab: model x framework x harness.

Frameworks (serving backends, all OpenAI-compatible, all MTP-enabled):
  - omlx:     resident oMLX server (:8000); full load/unload residency via admin API
  - mtplx:    MTPLX (native MTP speculative decoding); model fixed at start
  - llamacpp: llama-server with --spec-type draft-mtp; model fixed at start

Harnesses (agent wrappers around the model):
  - raw:      direct streaming HTTP to /v1/chat/completions (baseline)
  - pi:       pi CLI, full agent (tools enabled), --mode json event stream
  - omp:      omp CLI (pi fork; built-in `bench` workload — request-level,
              no tools, no file writes)
  - opencode: opencode + oh-my-opencode plugin (Sisyphus agent)

Every harness emits a verbose transcript (prompt, every iteration/loop, tool
calls, assistant messages, final response) via a log callback so the
dashboard can show a live scrolling view and the full output block after the
run. Files the agent generates in its work dir are listed in the result.

Residency model (load-run-unload): only the model under test is loaded in a
non-resident framework, on top of the always-resident oMLX routing models.
When a cell finishes, the gateway drops the framework's model copy, so the
next framework starts from a clean slate (no duplicate model copies in RAM).

Metrics captured per cell: TTFT, TPS, output tokens, total time, and the
number of agent iterations (turns/steps) each harness performed. Benchmarks
run long-running tasks with NO timeout limits; every framework loads the
model with a 240K context window and 32K max output tokens, MTP enabled.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
GGUF_DIR = MODELS_DIR / "gguf"
BENCH_STATE_DIR = PROJECT_DIR / ".inference-stack" / "bench"

# Benchmark context/output budget (applied to every framework server).
CONTEXT_WINDOW = 245760    # 240K context window
MAX_OUTPUT_TOKENS = 32768  # 32K max output tokens

# ---------------------------------------------------------------------------
# Model registry: logical model -> per-framework representation
# ---------------------------------------------------------------------------


MODELS: dict[str, dict[str, Any]] = {
    "qwen3.8-27b": {
        "label": "Qwen3.8 27B (6-bit, MTP)",
        "artifacts": {
            "omlx": {"id": "scottlowry--Qwen3.8-27B-oQ6e-mtp"},
            # MTPLX: native MTP speculative-decoding build (its own artifact).
            "mtplx": {"id": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"},
            # MTP GGUF (Q6_K) for llama.cpp (--spec-type draft-mtp).
            "llamacpp": {"path": str(GGUF_DIR / "Qwen3.8-27B-MTP-Q6_K.gguf"),
                         "id": "qwen3_8_27b"},
        },
    },
}

FRAMEWORKS: dict[str, dict[str, Any]] = {
    "omlx": {"port": 8000, "resident": True, "label": "oMLX (resident)"},
    "mtplx": {"port": 8400, "resident": False, "label": "MTPLX (MTP MLX)"},
    "llamacpp": {"port": 8300, "resident": False, "label": "llama.cpp (llama-server)"},
    # Dropped: mlx (mlx_lm.server strips MTP weights — no MTP spec decoding),
    # ollama (redundant with llama.cpp for GGUF models).
}

HARNESSES: dict[str, dict[str, Any]] = {
    "raw": {"label": "raw (streaming HTTP)", "available": True},
    "pi": {"label": "pi", "available": shutil.which("pi") is not None},
    "omp": {"label": "omp (bench)", "available": shutil.which("omp") is not None},
    "opencode": {"label": "opencode (Sisyphus)", "available": shutil.which("opencode") is not None},
    # Dropped: bionic (LM Studio GUI), dsh (deepseek harness).
}

# Standard benchmark prompt (unique suffix added per iteration to avoid cache).
BENCH_PROMPT = (
    "Explain, in exactly three short sentences, how a load balancer decides "
    "which backend server should handle an incoming request."
)


def _noop(stage: str) -> None:
    """Default no-op progress callback."""


class LabLog:
    """Collects a verbose transcript of a benchmark cell.

    Every line is timestamped and stored in ``lines`` (returned in the result
    as ``transcript``). Each entry is also pushed to ``log_cb(kind, message)``
    so a caller (the dashboard) can render a live scrolling view.
    """

    MAX_MESSAGE = 8000   # per-line cap (full assistant messages / tool args)
    MAX_LINES = 3000     # total cap (bounds memory on very long runs)

    def __init__(self, log_cb=None) -> None:
        self.lines: list[str] = []
        self._cb = log_cb

    def emit(self, kind: str, message: str) -> None:
        message = str(message).replace("\r", " ")
        if len(message) > self.MAX_MESSAGE:
            message = message[: self.MAX_MESSAGE] + f" …[+{len(message) - self.MAX_MESSAGE} chars]"
        stamp = time.strftime("%H:%M:%S")
        self.lines.append(f"[{stamp}] {kind}: {message}")
        del self.lines[: -self.MAX_LINES]
        if self._cb:
            try:
                self._cb(kind, message)
            except Exception:  # noqa: BLE001 - logging must never break a run
                pass


def _artifact(framework: str, model_key: str) -> dict[str, Any] | None:
    """The per-framework artifact spec for a logical model (or None)."""
    return MODELS.get(model_key, {}).get("artifacts", {}).get(framework)


def model_id_for(framework: str, model_key: str) -> str:
    """The model identifier to put in a request for a logical model."""
    art = _artifact(framework, model_key)
    if not art:
        raise ValueError(f"no {framework} artifact for model {model_key}")
    return art.get("id") or art.get("name")


def server_model_path(framework: str, model_key: str) -> str:
    """The on-disk model path used to start a server framework (may be empty)."""
    art = _artifact(framework, model_key) or {}
    return art.get("path", "")


def framework_available_for(framework: str, model_key: str) -> bool:
    """True if a (model, framework) combo has an artifact (else grey it out)."""
    return framework in MODELS.get(model_key, {}).get("artifacts", {})


# ---------------------------------------------------------------------------
# Framework manager
# ---------------------------------------------------------------------------


class FrameworkManager:
    """Starts/stops framework servers and manages model residency."""

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}
        self._server_model: dict[str, str] = {}  # framework -> model_key running
        BENCH_STATE_DIR.mkdir(parents=True, exist_ok=True)

    def endpoint(self, framework: str) -> str:
        return f"http://127.0.0.1:{FRAMEWORKS[framework]['port']}"

    # -- health -----------------------------------------------------------

    def _http_ok(self, url: str, timeout: float = 3.0) -> bool:
        try:
            r = httpx.get(url, timeout=timeout)
            return r.status_code < 500
        except httpx.HTTPError:
            return False

    def health(self, framework: str) -> bool:
        ep = self.endpoint(framework)
        if framework == "omlx":
            return self._http_ok(f"{ep}/admin/api/models")
        return self._http_ok(f"{ep}/health") or self._http_ok(f"{ep}/v1/models")

    def _wait_healthy(self, framework: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.health(framework):
                return True
            time.sleep(1.0)
        return self.health(framework)

    # -- oMLX (resident) --------------------------------------------------

    def _omlx_models(self) -> dict[str, dict]:
        r = httpx.get(f"{self.endpoint('omlx')}/admin/api/models", timeout=10)
        r.raise_for_status()
        return {m["id"]: m for m in r.json().get("models", [])}

    def _omlx_load(self, model_id: str) -> None:
        httpx.post(f"{self.endpoint('omlx')}/admin/api/models/{model_id}/load",
                   timeout=600).raise_for_status()

    def _omlx_unload(self, model_id: str) -> None:
        try:
            httpx.post(f"{self.endpoint('omlx')}/admin/api/models/{model_id}/unload",
                       timeout=120).raise_for_status()
        except httpx.HTTPError:
            pass

    def _omlx_apply_settings(self, model_id: str) -> None:
        """Apply the benchmark context/output budget to the oMLX model.

        Safe on a loaded model (settings update in place, no unload).
        """
        try:
            httpx.put(
                f"{self.endpoint('omlx')}/admin/api/models/{model_id}/settings",
                json={"max_context_window": CONTEXT_WINDOW,
                      "max_tokens": MAX_OUTPUT_TOKENS},
                timeout=30,
            ).raise_for_status()
        except httpx.HTTPError:
            pass  # non-fatal: the model keeps its current settings

    # -- server frameworks (mtplx, llamacpp) --------------------------------

    def _server_cmd(self, framework: str, model_key: str) -> list[str]:
        port = FRAMEWORKS[framework]["port"]
        if framework == "llamacpp":
            model_ref = server_model_path("llamacpp", model_key)
            alias = model_id_for("llamacpp", model_key)
            # --spec-type draft-mtp enables MTP speculative decoding when the
            # GGUF carries MTP tensors (the MTP build does).
            return ["llama-server", "-m", model_ref,
                    "--host", "127.0.0.1", "--port", str(port),
                    "--alias", alias, "--ctx-size", str(CONTEXT_WINDOW),
                    "--n-predict", str(MAX_OUTPUT_TOKENS),
                    "--spec-type", "draft-mtp"]
        if framework == "mtplx":
            model_id = model_id_for("mtplx", model_key)
            return ["mtplx", "serve", "--model", model_id,
                    "--host", "127.0.0.1", "--port", str(port),
                    "--no-auth", "--download", "--yes",
                    "--context-window", str(CONTEXT_WINDOW),
                    "--max-tokens", str(MAX_OUTPUT_TOKENS)]
        raise ValueError(f"unknown server framework: {framework}")

    def _stop_server(self, framework: str) -> None:
        proc = self._procs.get(framework)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        self._procs.pop(framework, None)
        self._server_model.pop(framework, None)

    # -- public API -------------------------------------------------------

    def ensure(self, framework: str, model_key: str, load_timeout: float = 600.0) -> tuple[str, str]:
        """Ensure `framework` is serving `model_key`. Returns (endpoint, model_id)."""
        if framework not in FRAMEWORKS:
            raise ValueError(f"unknown framework: {framework}")
        if model_key not in MODELS:
            raise ValueError(f"unknown model: {model_key}")

        if framework == "omlx":
            if not self.health("omlx"):
                raise RuntimeError("oMLX is not running (expected resident at :8000)")
            model_id = model_id_for("omlx", model_key)
            models = self._omlx_models()
            info = models.get(model_id)
            if info is None:
                raise RuntimeError(f"model {model_id} not registered in oMLX")
            # Apply the 240K/32K benchmark budget (in place; no unload).
            self._omlx_apply_settings(model_id)
            if not info.get("loaded") and not info.get("is_loading"):
                self._omlx_load(model_id)
                self._wait_loaded_omlx(model_id, load_timeout)
            return self.endpoint("omlx"), model_id

        # mtplx / llamacpp: one server per framework, model fixed at start.
        if self._server_model.get(framework) == model_key and self.health(framework):
            return self.endpoint(framework), model_id_for(framework, model_key)
        # Restart if running a different model or not healthy (drops the
        # previous model copy before loading the new one).
        self._stop_server(framework)
        cmd = self._server_cmd(framework, model_key)
        log_path = BENCH_STATE_DIR / f"{framework}.log"
        log_file = open(log_path, "ab")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                cwd=str(PROJECT_DIR))
        self._procs[framework] = proc
        self._server_model[framework] = model_key
        if not self._wait_healthy(framework, load_timeout):
            self._stop_server(framework)
            raise RuntimeError(f"{framework} server did not become healthy in {load_timeout}s "
                               f"(see {log_path})")
        return self.endpoint(framework), model_id_for(framework, model_key)

    def _wait_loaded_omlx(self, model_id: str, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                info = self._omlx_models().get(model_id, {})
            except httpx.HTTPError:
                info = {}
            if info.get("loaded"):
                return
            if info.get("is_loading"):
                time.sleep(2.0)
                continue
            time.sleep(1.0)
        raise RuntimeError(f"oMLX model {model_id} did not load in {timeout}s")

    def release(self, framework: str, model_key: str) -> None:
        """Unload the model / stop the server (load-run-unload residency).

        The gateway drops the previous framework's model copy here, so the
        next cell never runs with two model copies resident at once.
        """
        if framework == "omlx":
            # Leave oMLX resident models as-is unless explicitly asked to free RAM.
            return
        self._stop_server(framework)

    def shutdown(self) -> None:
        for fw in list(self._procs):
            self._stop_server(fw)


# ---------------------------------------------------------------------------
# Harness runners
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    harness: str
    framework: str
    model: str
    iterations: int
    ok: int = 0
    failed: int = 0
    ttft_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    agent_iters: list[int] = field(default_factory=list)  # agent turns/steps per iteration
    error: str = ""
    work_dir: str = ""
    transcript: list[str] = field(default_factory=list)  # verbose log lines
    files: list[dict] = field(default_factory=list)      # files generated in the work dir

    def summary(self) -> dict[str, Any]:
        if not self.ok:
            return {"status": "failed", "error": self.error or "no successful runs",
                    "ok": 0, "failed": self.failed, "work_dir": self.work_dir,
                    "transcript": self.transcript, "files": self.files}
        total = statistics.mean(self.total_ms)
        toks = statistics.mean(self.tokens)
        if self.ttft_ms:
            # Streaming harness: decode TPS over (total - ttft).
            ttft = statistics.mean(self.ttft_ms)
            decode = max(total - ttft, 1.0)
            tps = toks / (decode / 1000.0)
            ttft_field = {"avg_ttft_ms": round(ttft, 1),
                          "p95_ttft_ms": round(sorted(self.ttft_ms)[int(len(self.ttft_ms) * 0.95)], 1)}
        else:
            # Non-streaming harness (agent CLIs): TPS over total task time.
            tps = toks / (total / 1000.0) if total > 0 else 0.0
            ttft_field = {"avg_ttft_ms": None}
        agent_iters = statistics.mean(self.agent_iters) if self.agent_iters else None
        return {
            "status": "success",
            "ok": self.ok,
            "failed": self.failed,
            **ttft_field,
            "avg_total_ms": round(total, 1),
            "avg_tokens": round(toks, 1),
            "avg_tps": round(tps, 2),
            "avg_agent_iters": round(agent_iters, 2) if agent_iters is not None else None,
            "work_dir": self.work_dir,
            "transcript": self.transcript,
            "files": self.files,
        }


def _workdir_files(work_dir: Path) -> list[dict]:
    """List files generated in the work dir (excluding harness config dirs)."""
    files: list[dict] = []
    if not work_dir.exists():
        return files
    skip_dirs = {"sessions", ".git", "node_modules", ".omlx", ".omc"}
    for root, dirs, names in os.walk(work_dir):
        dirs[:] = [d for d in dirs
                   if not d.startswith(".")
                   and d not in skip_dirs and not d.endswith("-config")]
        for name in sorted(names):
            if name.startswith("."):
                continue
            p = Path(root) / name
            try:
                size = p.stat().st_size
            except OSError:
                continue
            files.append({"name": str(p.relative_to(work_dir)), "size": size})
    return files


def _run_raw(endpoint: str, model_id: str, prompt: str, iterations: int,
             max_tokens: int, config_dir: Path, progress_cb=None,
             log: LabLog | None = None) -> BenchResult:
    """Direct streaming HTTP to /v1/chat/completions (baseline harness).

    One streaming call per iteration with ``stream_options.include_usage``.
    TTFT and total time are measured locally (portable); the token count comes
    from the standard ``usage.completion_tokens`` in the final chunk. The full
    response text is captured in the transcript. No timeout: long-running
    tasks run to completion.
    """
    cb = progress_cb or _noop
    log = log or LabLog()
    res = BenchResult(harness="raw", framework="", model=model_id, iterations=iterations)
    url = f"{endpoint}/v1/chat/completions"
    with httpx.Client(timeout=None) as client:
        for i in range(iterations):
            cb(f"raw iteration {i + 1}/{iterations}")
            unique = f"{prompt} [id:{uuid.uuid4().hex[:8]}]"
            body = {"model": model_id,
                    "messages": [{"role": "user", "content": unique}],
                    "max_tokens": max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True}}
            log.emit("prompt", f"raw iteration {i + 1}/{iterations}: POST {url} "
                               f"(model={model_id}, max_tokens={max_tokens})")
            log.emit("prompt", f"Prompt: {unique}")
            t0 = time.time()
            ttft = None
            tokens = 0
            text_parts: list[str] = []
            try:
                with client.stream("POST", url, json=body) as r:
                    if r.status_code != 200:
                        res.failed += 1
                        res.error = f"HTTP {r.status_code}: {r.read()[:200]}"
                        log.emit("error", res.error)
                        continue
                    for line in r.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        usage = chunk.get("usage")
                        if usage and usage.get("completion_tokens"):
                            tokens = int(usage["completion_tokens"])
                        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                        content = delta.get("content") or delta.get("reasoning_content") \
                            or delta.get("reasoning")
                        if content:
                            if ttft is None:
                                ttft = time.time() - t0
                                log.emit("stream", f"first token after {ttft * 1000:.0f} ms")
                            text_parts.append(content)
                total = time.time() - t0
                if ttft is None:
                    res.failed += 1
                    res.error = "no tokens received"
                    log.emit("error", res.error)
                    continue
                if tokens <= 0:
                    tokens = max(int(total * 20), 1)  # fallback estimate
                res.ok += 1
                res.ttft_ms.append(ttft * 1000)
                res.total_ms.append(total * 1000)
                res.tokens.append(tokens)
                res.agent_iters.append(1)  # one request per raw iteration
                tps = tokens / max(total - ttft, 0.001)
                log.emit("result", f"done: {tokens} tokens in {total:.1f}s "
                                   f"(TTFT {ttft * 1000:.0f} ms, {tps:.1f} t/s decode)")
                log.emit("response", "".join(text_parts))
            except httpx.HTTPError as e:
                res.failed += 1
                res.error = str(e)
                log.emit("error", str(e))
    res.transcript = log.lines
    return res


def _write_pi_config(config_dir: Path, endpoint: str, model_id: str) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "providers": {
            "bench": {
                "baseUrl": f"{endpoint}/v1",
                "api": "openai-completions",
                "apiKey": "local",
                "compat": {"supportsStore": False, "maxTokensField": "max_tokens"},
                "models": [{
                    "id": model_id,
                    "name": "bench",
                    "input": ["text"],
                    "contextWindow": CONTEXT_WINDOW,
                    "maxTokens": MAX_OUTPUT_TOKENS,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                }],
            }
        }
    }
    (config_dir / "models.json").write_text(json.dumps(cfg, indent=2))


def _run_pi(endpoint: str, model_id: str, prompt: str, iterations: int,
            max_tokens: int, config_dir: Path, progress_cb=None,
            log: LabLog | None = None) -> BenchResult:
    """Run the pi CLI as a full agent (tools enabled) in an isolated work dir.

    Uses ``--mode json`` and streams the event stream live to capture exact
    metrics: agent iterations (``turn_end`` events), output tokens (per
    assistant ``message_end`` usage), and TTFT (first streamed delta). Every
    event (turns, tool calls, assistant messages) is written to the verbose
    transcript. No timeout: long-running tasks run to completion.
    """
    cb = progress_cb or _noop
    log = log or LabLog()
    res = BenchResult(harness="pi", framework="", model=model_id, iterations=iterations)
    config_dir.mkdir(parents=True, exist_ok=True)
    work_dir = config_dir.parent / "pi-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    res.work_dir = str(work_dir)
    _write_pi_config(config_dir, endpoint, model_id)
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(config_dir)
    env["PI_OFFLINE"] = "1"
    env["PI_SKIP_VERSION_CHECK"] = "1"
    env.pop("PI_PROVIDER", None)
    env.pop("PI_MODEL", None)
    for i in range(iterations):
        cb(f"pi iteration {i + 1}/{iterations} (agent working in {work_dir.name})")
        unique = f"{prompt} [id:{uuid.uuid4().hex[:8]}]"
        log.emit("prompt", f"pi iteration {i + 1}/{iterations}: agent started "
                           f"(tools enabled, work dir {work_dir.name})")
        log.emit("prompt", f"Prompt: {unique}")
        t0 = time.time()
        turns = 0
        out_tokens = 0
        ttft = None
        try:
            proc = subprocess.Popen(
                ["pi", "-p", unique, "--model", f"bench/{model_id}",
                 "--no-approve", "--mode", "json"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=env, cwd=str(work_dir),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("type")
                if etype == "turn_start":
                    log.emit("turn", f"turn {turns + 1} started")
                elif etype == "turn_end":
                    turns += 1
                    log.emit("turn", f"turn {turns} ended")
                elif etype == "tool_execution_start":
                    args = json.dumps(ev.get("args") or {}, ensure_ascii=False)
                    log.emit("tool", f"running {ev.get('toolName', 'tool')}: {args}")
                elif etype == "tool_execution_end":
                    state = "FAILED" if ev.get("isError") else "completed"
                    log.emit("tool", f"{ev.get('toolName', 'tool')} {state}")
                elif etype == "message_end":
                    msg = ev.get("message") or {}
                    if msg.get("role") == "assistant":
                        out_tokens += int((msg.get("usage") or {}).get("output") or 0)
                        text = "".join(
                            c.get("text", "") for c in (msg.get("content") or [])
                            if isinstance(c, dict) and c.get("type") == "text")
                        if text.strip():
                            log.emit("assistant", text)
                elif etype == "message_update" and ttft is None:
                    ame = ev.get("assistantMessageEvent") or {}
                    if ame.get("delta"):
                        ttft = time.time() - t0
                        log.emit("stream", f"first token after {ttft * 1000:.0f} ms")
                elif etype in ("agent_end", "agent_settled"):
                    log.emit("agent", "agent finished")
            proc.wait()
            total = time.time() - t0
            stderr = proc.stderr.read() if proc.stderr else ""
            if proc.returncode == 0 and out_tokens > 0:
                res.ok += 1
                res.total_ms.append(total * 1000)
                res.tokens.append(out_tokens)
                res.agent_iters.append(turns)
                if ttft is not None:
                    res.ttft_ms.append(ttft * 1000)
                log.emit("result", f"done: {turns} turns, {out_tokens} output tokens, "
                                   f"{total:.1f}s total")
            else:
                res.failed += 1
                res.error = (stderr or f"rc={proc.returncode}")[-300:]
                log.emit("error", res.error)
        except Exception as e:  # noqa: BLE001
            res.failed += 1
            res.error = str(e)[-300:]
            log.emit("error", str(e))
    res.transcript = log.lines
    return res


def _run_omp(endpoint: str, model_id: str, prompt: str, iterations: int,
             max_tokens: int, profile_dir: Path, progress_cb=None,
             log: LabLog | None = None) -> BenchResult:
    """Use omp's built-in `bench` against an isolated bench provider.

    Each bench run is a single request, so agent iterations = 1 per run.
    NOTE: `omp bench` is a request-level throughput benchmark — it has no
    tools and no file-system access, so it never creates files (the model's
    response text is measured and discarded). Use pi or opencode for
    agent tasks that produce files. No timeout: long-running tasks run to
    completion.
    """
    cb = progress_cb or _noop
    log = log or LabLog()
    res = BenchResult(harness="omp", framework="", model=model_id, iterations=iterations)
    profile_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "providers": {
            "bench": {
                "baseUrl": f"{endpoint}/v1",
                "api": "openai-completions",
                "apiKey": "local",
                "models": [{
                    "id": model_id, "name": "bench", "input": ["text"],
                    "contextWindow": CONTEXT_WINDOW, "maxTokens": MAX_OUTPUT_TOKENS,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                }],
            }
        }
    }
    (profile_dir / "models.json").write_text(json.dumps(cfg, indent=2))
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(profile_dir)
    env["PI_OFFLINE"] = "1"
    env["PI_SKIP_VERSION_CHECK"] = "1"
    cb(f"omp bench ({iterations} runs)")
    log.emit("prompt", f"omp bench: {iterations} runs, profile=chat, "
                       f"max_tokens={max_tokens}, model={model_id}")
    log.emit("prompt", f"Prompt: {prompt}")
    log.emit("note", "omp bench is a request-level benchmark (no tools, no file "
                     "writes) — the model's response is measured and discarded")
    try:
        proc = subprocess.run(
            ["omp", "bench", f"bench/{model_id}", "--runs", str(iterations),
             "--profile", "chat", "--max-tokens", str(max_tokens), "--json"],
            capture_output=True, text=True, env=env, cwd=str(PROJECT_DIR),
        )
        out = proc.stdout.strip()
        start = out.find("{")
        if start < 0:
            res.error = (proc.stderr or out or "no JSON output")[-300:]
            log.emit("error", res.error)
            res.transcript = log.lines
            return res
        data = json.loads(out[start:])
        models = data.get("models") or []
        if not models:
            res.error = "no model results in omp bench output"
            log.emit("error", res.error)
            res.transcript = log.lines
            return res
        m = models[0]
        results = m.get("results") or []
        stats = m.get("stats") or {}

        def _mean(key: str) -> float:
            block = stats.get(key) or {}
            return float(block.get("mean") or 0.0)

        ok_runs = [r for r in results if r.get("ok")]
        res.ok = len(ok_runs)
        res.failed = len(results) - len(ok_runs)
        for idx, r in enumerate(results, 1):
            if r.get("ok"):
                tps = (int(r.get("outputTokens") or 0) /
                       max(float(r.get("durationMs") or 0) - float(r.get("ttftMs") or 0), 1.0) * 1000.0)
                log.emit("run", f"run {idx}: ok — TTFT {r.get('ttftMs')} ms, "
                                f"{r.get('outputTokens')} tokens, {r.get('durationMs')} ms total, "
                                f"{tps:.1f} t/s decode")
            else:
                log.emit("run", f"run {idx}: FAILED — {str(r.get('error') or r)[:200]}")
        if ok_runs:
            res.ttft_ms = [float(r.get("ttftMs") or 0) for r in ok_runs]
            res.total_ms = [float(r.get("durationMs") or 0) for r in ok_runs]
            res.tokens = [int(r.get("outputTokens") or 0) for r in ok_runs]
            res.agent_iters = [1] * len(ok_runs)  # one request per bench run
        if res.ok == 0:
            res.error = (m.get("error") or "all runs failed")[-300:]
            log.emit("error", res.error)
        else:
            log.emit("result", f"done: {res.ok}/{len(results)} runs ok, "
                               f"mean TTFT {statistics.mean(res.ttft_ms):.0f} ms, "
                               f"mean {statistics.mean(res.tokens):.0f} tokens")
    except json.JSONDecodeError as e:
        res.error = f"JSON parse error: {e}"
        log.emit("error", res.error)
    res.transcript = log.lines
    return res


def _run_opencode(endpoint: str, model_id: str, prompt: str, iterations: int,
                  max_tokens: int, config_dir: Path, progress_cb=None,
                  log: LabLog | None = None) -> BenchResult:
    """Run opencode with the Sisyphus (oh-my-opencode) agent against an isolated provider.

    The agent runs in an isolated work dir (``config_dir.parent / "opencode-work"``)
    so any files it creates (e.g. an HTML page) land there, not in the project.
    The work dir is recorded in ``res.work_dir``.

    Metrics from the ``--format json`` event stream: agent iterations
    (``step_finish`` events) and output tokens (per-step usage). opencode
    emits completed parts only (no per-token deltas), so TTFT is not
    measurable and TPS is output tokens over total task time. Every event
    (steps, tool calls, assistant text) is written to the verbose transcript.
    No timeout: long-running tasks run to completion.
    """
    cb = progress_cb or _noop
    log = log or LabLog()
    res = BenchResult(harness="opencode", framework="", model=model_id, iterations=iterations)
    config_dir.mkdir(parents=True, exist_ok=True)
    work_dir = config_dir.parent / "opencode-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    res.work_dir = str(work_dir)
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "plugin": ["oh-my-openagent@latest"],
        "provider": {
            "bench": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Bench",
                "options": {"baseURL": f"{endpoint}/v1", "apiKey": "local"},
                "models": {model_id: {"name": "bench",
                                      "limit": {"context": CONTEXT_WINDOW,
                                               "output": MAX_OUTPUT_TOKENS}}},
            }
        },
    }
    cfg_path = config_dir / "opencode.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    env = dict(os.environ)
    env["OPENCODE_CONFIG"] = str(cfg_path)
    for i in range(iterations):
        cb(f"opencode iteration {i + 1}/{iterations} (agent working in {work_dir.name})")
        unique = f"{prompt} [id:{uuid.uuid4().hex[:8]}]"
        log.emit("prompt", f"opencode iteration {i + 1}/{iterations}: Sisyphus agent "
                           f"started (work dir {work_dir.name})")
        log.emit("prompt", f"Prompt: {unique}")
        t0 = time.time()
        steps = 0
        out_tokens = 0
        try:
            proc = subprocess.run(
                ["opencode", "run", unique, "-m", f"bench/{model_id}",
                 "--agent", "Sisyphus", "--format", "json",
                 "--dir", str(work_dir)],
                capture_output=True, text=True, env=env, cwd=str(work_dir),
            )
            total = time.time() - t0
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("type")
                part = ev.get("part") or {}
                if etype == "step_start":
                    log.emit("step", f"step {steps + 1} started")
                elif etype == "step_finish":
                    steps += 1
                    step_out = int((part.get("tokens") or {}).get("output") or 0)
                    out_tokens += step_out
                    log.emit("step", f"step {steps} finished ({step_out} output tokens this step)")
                elif etype == "text":
                    text = part.get("text") or ""
                    if text.strip():
                        log.emit("assistant", text)
                elif etype == "tool_use":
                    state = part.get("state") or {}
                    status = state.get("status") or "unknown"
                    title = state.get("title") or ""
                    output = str(state.get("output") or "")[:300]
                    log.emit("tool", f"{part.get('tool', 'tool')} "
                                     f"{('· ' + title) if title else ''} → {status}"
                             f"{(' — ' + output) if output else ''}")
            if proc.returncode == 0 and out_tokens > 0:
                res.ok += 1
                res.total_ms.append(total * 1000)
                res.tokens.append(out_tokens)
                res.agent_iters.append(steps)
                log.emit("result", f"done: {steps} steps, {out_tokens} output tokens, "
                                   f"{total:.1f}s total")
            else:
                res.failed += 1
                res.error = (proc.stderr or proc.stdout or f"rc={proc.returncode}")[-300:]
                log.emit("error", res.error)
        except Exception as e:  # noqa: BLE001
            res.failed += 1
            res.error = str(e)[-300:]
            log.emit("error", str(e))
    res.transcript = log.lines
    return res


HARNESS_RUNNERS = {
    "raw": _run_raw,
    "pi": _run_pi,
    "omp": _run_omp,
    "opencode": _run_opencode,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_benchmark(framework: str, model_key: str, harness: str,
                  iterations: int = 3, max_tokens: int = 128,
                  prompt: str = BENCH_PROMPT,
                  unload_after: bool = True,
                  warmup: int = 1,
                  manager: FrameworkManager | None = None,
                  work_dir: Path | None = None,
                  progress_cb=None,
                  log_cb=None) -> dict[str, Any]:
    """Run one (framework, model, harness) benchmark cell. Returns a result dict.

    ``warmup`` uncounted iterations are run first to absorb server/model warmup
    (important for server frameworks where the first request is slow).

    ``progress_cb`` is an optional callable ``cb(stage: str)`` invoked at key
    points so a caller (e.g. the dashboard) can show live progress.

    ``log_cb`` is an optional callable ``cb(kind: str, message: str)`` that
    receives every verbose transcript line (prompt, iterations, tool calls,
    assistant messages, final response) for a live scrolling view.
    """
    cb = progress_cb or _noop
    log = LabLog(log_cb)
    own_manager = manager is None
    manager = manager or FrameworkManager()
    work_dir = work_dir or (BENCH_STATE_DIR / "work" / f"{framework}-{model_key}-{harness}")
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        cb(f"starting {framework} server")
        log.emit("server", f"starting {framework} server (model {model_key})")
        endpoint, model_id = manager.ensure(framework, model_key)
        cb("server ready")
        log.emit("server", f"server ready at {endpoint} (model {model_id})")
        runner = HARNESS_RUNNERS.get(harness)
        if runner is None:
            return {"status": "failed", "error": f"no runner for harness '{harness}'",
                    "transcript": log.lines}
        # Warmup (uncounted) — absorbs first-request latency.
        for _ in range(max(warmup, 0)):
            cb("warmup")
            log.emit("warmup", "warmup iteration (uncounted)")
            _warmup_once(runner, harness, endpoint, model_id, prompt, max_tokens, work_dir, log)
        cb(f"running {harness} ({iterations} iterations)")
        res = runner(endpoint, model_id, prompt, iterations, max_tokens,
                     work_dir / f"{harness}-config", cb, log)
        summary = res.summary()
        summary["files"] = _workdir_files(work_dir)
        summary.update({
            "framework": framework,
            "model": model_key,
            "model_id": model_id,
            "harness": harness,
            "endpoint": endpoint,
            "iterations": iterations,
            "warmup": warmup,
            "max_tokens": max_tokens,
            "work_dir": str(work_dir),
            "elapsed_s": round(time.time() - started, 1),
        })
        return summary
    except Exception as e:  # noqa: BLE001 - report any failure in the result
        return {"status": "failed", "framework": framework, "model": model_key,
                "harness": harness, "error": str(e),
                "transcript": log.lines,
                "elapsed_s": round(time.time() - started, 1)}
    finally:
        if unload_after:
            cb("releasing server")
            log.emit("server", f"releasing {framework} server (load-run-unload)")
            try:
                manager.release(framework, model_key)
            except Exception:  # noqa: BLE001
                pass
        if own_manager:
            manager.shutdown()


def _warmup_once(runner, harness: str, endpoint: str, model_id: str,
                 prompt: str, max_tokens: int, work_dir: Path,
                 log: LabLog | None = None) -> None:
    """Run one uncounted iteration to warm up the server/model."""
    try:
        runner(endpoint, model_id, prompt, 1, max_tokens, work_dir / f"{harness}-config",
               None, log)
    except Exception:  # noqa: BLE001 - warmup failures are non-fatal
        pass


def available_options() -> dict[str, Any]:
    """Options for the dashboard dropdowns."""
    models = []
    for key, spec in MODELS.items():
        arts = spec.get("artifacts", {})
        frameworks: dict[str, bool] = {}
        for fw in FRAMEWORKS:
            art = arts.get(fw)
            if not art:
                frameworks[fw] = False  # no artifact for this model -> grey out
                continue
            if fw == "llamacpp":
                frameworks[fw] = bool(art.get("path")) and Path(art["path"]).exists()
            else:  # omlx, mtplx (model resolved by id at run time)
                frameworks[fw] = True
        models.append({"key": key, "label": spec["label"], "frameworks": frameworks})
    return {
        "models": models,
        "frameworks": [
            {"key": k, "label": v["label"], "resident": v["resident"]}
            for k, v in FRAMEWORKS.items()
        ],
        "harnesses": [
            {"key": k, "label": v["label"], "available": v["available"]}
            for k, v in HARNESSES.items()
        ],
        "default_prompt": BENCH_PROMPT,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Multi-axis benchmark lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("options", help="print available options")

    p_run = sub.add_parser("run", help="run one benchmark cell")
    p_run.add_argument("--framework", required=True, choices=list(FRAMEWORKS))
    p_run.add_argument("--model", required=True, choices=list(MODELS))
    p_run.add_argument("--harness", required=True, choices=list(HARNESS_RUNNERS))
    p_run.add_argument("--iterations", type=int, default=3)
    p_run.add_argument("--max-tokens", type=int, default=32768)
    p_run.add_argument("--keep-loaded", action="store_true",
                       help="do not unload after the run")

    args = p.parse_args()
    if args.cmd == "options":
        print(json.dumps(available_options(), indent=2))
    elif args.cmd == "run":
        result = run_benchmark(
            args.framework, args.model, args.harness,
            iterations=args.iterations, max_tokens=args.max_tokens,
            unload_after=not args.keep_loaded,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
