"""
Multi-axis benchmark lab: model x framework x harness.

Frameworks (serving backends, all OpenAI-compatible):
  - omlx:     resident oMLX server (:8000); full load/unload residency via admin API
  - mlx:      mlx_lm.server; model fixed at start (start/stop == load/unload)
  - llamacpp: llama-server; model fixed at start (start/stop == load/unload)
  - ollama:   ollama serve; models pulled, auto-load on first request

Harnesses (agent wrappers around the model):
  - raw:      direct HTTP to /v1/chat/completions (baseline, no harness overhead)
  - omp:      omp CLI (pi fork; built-in `bench` workload)
  - pi:       pi CLI
  - sisyphus: opencode + oh-my-opencode plugin
  - dsh:      deepseek harness (npx @deepseek-ai/dsh)

Residency model (load-run-unload): only the model under test is loaded in a
non-resident framework, on top of the always-resident oMLX routing models.
This keeps RAM bounded so benchmarks can run while the gateway stays usable.
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
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
MLX_VENV = Path.home() / ".omlx" / "bench-venv"
MLX_SERVER = MLX_VENV / "bin" / "mlx_lm.server"
BENCH_STATE_DIR = PROJECT_DIR / ".inference-stack" / "bench"

# Benchmark context/output budget (applied to every framework server).
CONTEXT_WINDOW = 131072   # 131K context (large prefill support)
MAX_OUTPUT_TOKENS = 32768  # 32K max output tokens

# ---------------------------------------------------------------------------
# Model registry: logical model -> per-framework representation
# ---------------------------------------------------------------------------


def _mlx_snapshot(repo: str) -> str:
    """Resolve the single HF snapshot dir for an MLX model repo."""
    base = HF_CACHE / f"models--{repo}" / "snapshots"
    if base.exists():
        snaps = [p for p in base.iterdir() if p.is_dir()]
        if snaps:
            return str(snaps[0])
    return str(base / "unknown")


MODELS: dict[str, dict[str, Any]] = {
    "qwen3.6-35b-a3b": {
        "label": "Qwen3.6 35B-A3B (6-bit)",
        "artifacts": {
            "omlx": {"id": "mlx-community--Qwen3.6-35B-A3B-6bit"},
            "mlx": {"path": _mlx_snapshot("mlx-community--Qwen3.6-35B-A3B-6bit"),
                    "id": "mlx-community/Qwen3.6-35B-A3B-6bit"},
            "llamacpp": {"path": str(GGUF_DIR / "Qwen3.6-35B-A3B-UD-Q6_K.gguf"),
                         "id": "qwen3_6_35b_a3b"},
            "ollama": {"name": "bench/qwen36-35b-a3b-q6",
                       "gguf": str(GGUF_DIR / "Qwen3.6-35B-A3B-UD-Q6_K.gguf")},
        },
    },
    "qwen3.8-27b": {
        "label": "Qwen3.8 27B (6-bit, MTP)",
        "artifacts": {
            "omlx": {"id": "scottlowry--Qwen3.8-27B-oQ6e-mtp"},
            # MTP MLX model (same OptiQ 6-bit MTP build oMLX serves; MLX-format,
            # native 256K context, mtp_num_hidden_layers=1). Already in HF cache.
            "mlx": {"path": _mlx_snapshot("scottlowry--Qwen3.8-27B-oQ6e-mtp"),
                    "id": "scottlowry/Qwen3.8-27B-oQ6e-mtp"},
            # MTPLX: native MTP speculative-decoding build (its own artifact).
            "mtplx": {"id": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"},
            # MTP GGUF (Q6_K) for llama.cpp / Ollama.
            "llamacpp": {"path": str(GGUF_DIR / "Qwen3.8-27B-MTP-Q6_K.gguf"),
                         "id": "qwen3_8_27b"},
            "ollama": {"name": "bench/qwen38-27b-q6",
                       "gguf": str(GGUF_DIR / "Qwen3.8-27B-MTP-Q6_K.gguf")},
        },
    },
}

FRAMEWORKS: dict[str, dict[str, Any]] = {
    "omlx": {"port": 8000, "resident": True, "label": "oMLX (resident)"},
    "mlx": {"port": 8200, "resident": False, "label": "MLX (mlx_lm.server)"},
    "mtplx": {"port": 8400, "resident": False, "label": "MTPLX (MTP MLX)"},
    "llamacpp": {"port": 8300, "resident": False, "label": "llama.cpp (llama-server)"},
    "ollama": {"port": 11434, "resident": False, "label": "Ollama"},
}

HARNESSES: dict[str, dict[str, Any]] = {
    "raw": {"label": "raw HTTP (baseline)", "available": True},
    "bionic": {"label": "Bionic (LM Studio GUI)", "available": True},
    "omp": {"label": "omp", "available": shutil.which("omp") is not None},
    "pi": {"label": "pi", "available": shutil.which("pi") is not None},
    "sisyphus": {"label": "Sisyphus (opencode)", "available": shutil.which("opencode") is not None},
    "dsh": {"label": "dsh (deepseek harness)", "available": shutil.which("npx") is not None},
}

# Standard benchmark prompt (unique suffix added per iteration to avoid cache).
BENCH_PROMPT = (
    "Explain, in exactly three short sentences, how a load balancer decides "
    "which backend server should handle an incoming request."
)


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


def ollama_gguf(model_key: str) -> str:
    """The GGUF path used to create an Ollama model for a logical model."""
    art = _artifact("ollama", model_key) or {}
    return art.get("gguf", "")


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
        if framework == "ollama":
            return self._http_ok(f"{ep}/api/tags")
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

    # -- server frameworks (mlx, llamacpp) --------------------------------

    def _server_cmd(self, framework: str, model_key: str) -> list[str]:
        port = FRAMEWORKS[framework]["port"]
        if framework == "mlx":
            model_ref = server_model_path("mlx", model_key)
            # mlx_lm.server: context is the model's native window (256K for the
            # MTP build); --max-tokens caps default output. KV cache grows lazily.
            return [str(MLX_SERVER), "--model", model_ref,
                    "--host", "127.0.0.1", "--port", str(port),
                    "--max-tokens", str(MAX_OUTPUT_TOKENS)]
        if framework == "llamacpp":
            model_ref = server_model_path("llamacpp", model_key)
            alias = model_id_for("llamacpp", model_key)
            # --spec-type draft-mtp enables MTP speculative decoding when the
            # GGUF carries MTP tensors (the MTP build does).
            return ["llama-server", "-m", model_ref,
                    "--host", "127.0.0.1", "--port", str(port),
                    "--alias", alias, "--ctx-size", str(CONTEXT_WINDOW),
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
            if not info.get("loaded") and not info.get("is_loading"):
                self._omlx_load(model_id)
                self._wait_loaded_omlx(model_id, load_timeout)
            return self.endpoint("omlx"), model_id

        if framework == "ollama":
            return self._ensure_ollama(model_key)

        # mlx / llamacpp: one server per framework, model fixed at start.
        if self._server_model.get(framework) == model_key and self.health(framework):
            return self.endpoint(framework), model_id_for(framework, model_key)
        # Restart if running a different model or not healthy.
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

    def _ensure_ollama(self, model_key: str) -> tuple[str, str]:
        ep = self.endpoint("ollama")
        if not self.health("ollama"):
            log_file = open(BENCH_STATE_DIR / "ollama.log", "ab")
            proc = subprocess.Popen(["ollama", "serve"], stdout=log_file,
                                    stderr=subprocess.STDOUT, cwd=str(PROJECT_DIR))
            self._procs["ollama"] = proc
            self._server_model["ollama"] = ""
            if not self._wait_healthy("ollama", 60):
                raise RuntimeError("ollama serve did not become healthy in 60s")
        model_id = model_id_for("ollama", model_key)
        # Ensure the model is pulled (no-op if present).
        r = httpx.get(f"{ep}/api/tags", timeout=10)
        have = {m.get("name") for m in r.json().get("models", [])}
        if model_id not in have:
            self._ollama_create_from_gguf(model_key, model_id)
        return ep, model_id

    def _ollama_create_from_gguf(self, model_key: str, model_id: str) -> None:
        """Create an Ollama model from a local GGUF via a Modelfile."""
        gguf = Path(ollama_gguf(model_key))
        if not gguf.exists():
            raise RuntimeError(f"GGUF not found for ollama import: {gguf}")
        modelfile = BENCH_STATE_DIR / f"Modelfile-{model_key}"
        modelfile.write_text(
            f"FROM {gguf}\n"
            f"PARAMETER num_ctx {CONTEXT_WINDOW}\n"
            f"PARAMETER num_predict {MAX_OUTPUT_TOKENS}\n"
        )
        r = subprocess.run(["ollama", "create", model_id, "-f", str(modelfile)],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"ollama create failed: {r.stderr[-500:]}")

    def release(self, framework: str, model_key: str) -> None:
        """Unload the model / stop the server (load-run-unload residency)."""
        if framework == "omlx":
            # Leave oMLX resident models as-is unless explicitly asked to free RAM.
            return
        if framework == "ollama":
            model_id = model_id_for("ollama", model_key)
            try:
                httpx.post(f"{self.endpoint('ollama')}/api/delete",
                           json={"name": model_id}, timeout=60)
            except httpx.HTTPError:
                pass
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
    error: str = ""

    def summary(self) -> dict[str, Any]:
        if not self.ok:
            return {"status": "failed", "error": self.error or "no successful runs",
                    "ok": 0, "failed": self.failed}
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
        return {
            "status": "success",
            "ok": self.ok,
            "failed": self.failed,
            **ttft_field,
            "avg_total_ms": round(total, 1),
            "avg_tokens": round(toks, 1),
            "avg_tps": round(tps, 2),
        }


def _run_raw(endpoint: str, model_id: str, prompt: str, iterations: int,
             max_tokens: int) -> BenchResult:
    """Direct streaming HTTP to /v1/chat/completions (baseline harness).

    One streaming call per iteration with ``stream_options.include_usage``.
    TTFT and total time are measured locally (portable); the token count comes
    from the standard ``usage.completion_tokens`` in the final chunk.
    """
    res = BenchResult(harness="raw", framework="", model=model_id, iterations=iterations)
    url = f"{endpoint}/v1/chat/completions"
    with httpx.Client(timeout=600) as client:
        for _ in range(iterations):
            unique = f"{prompt} [id:{uuid.uuid4().hex[:8]}]"
            body = {"model": model_id,
                    "messages": [{"role": "user", "content": unique}],
                    "max_tokens": max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True}}
            t0 = time.time()
            ttft = None
            tokens = 0
            try:
                with client.stream("POST", url, json=body) as r:
                    if r.status_code != 200:
                        res.failed += 1
                        res.error = f"HTTP {r.status_code}: {r.read()[:200]}"
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
                        if ttft is None and (delta.get("content")
                                             or delta.get("reasoning_content")
                                             or delta.get("reasoning")):
                            ttft = time.time() - t0
                total = time.time() - t0
                if ttft is None:
                    res.failed += 1
                    res.error = "no tokens received"
                    continue
                if tokens <= 0:
                    tokens = max(int(total * 20), 1)  # fallback estimate
                res.ok += 1
                res.ttft_ms.append(ttft * 1000)
                res.total_ms.append(total * 1000)
                res.tokens.append(tokens)
            except httpx.HTTPError as e:
                res.failed += 1
                res.error = str(e)
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
                    "contextWindow": 32000,
                    "maxTokens": 2048,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                }],
            }
        }
    }
    (config_dir / "models.json").write_text(json.dumps(cfg, indent=2))


def _run_pi(endpoint: str, model_id: str, prompt: str, iterations: int,
            max_tokens: int, config_dir: Path) -> BenchResult:
    res = BenchResult(harness="pi", framework="", model=model_id, iterations=iterations)
    _write_pi_config(config_dir, endpoint, model_id)
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(config_dir)
    env["PI_OFFLINE"] = "1"
    env["PI_SKIP_VERSION_CHECK"] = "1"
    env.pop("PI_PROVIDER", None)
    env.pop("PI_MODEL", None)
    for _ in range(iterations):
        unique = f"{prompt} [id:{uuid.uuid4().hex[:8]}]"
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["pi", "-p", unique, "--model", f"bench/{model_id}",
                 "--no-tools", "--no-approve"],
                capture_output=True, text=True, timeout=300, env=env,
                cwd=str(PROJECT_DIR),
            )
            out = proc.stdout.strip()
            if proc.returncode == 0 and out:
                res.ok += 1
                res.total_ms.append((time.time() - t0) * 1000)
                # Estimate tokens from output length (~4 chars/token).
                res.tokens.append(max(len(out) // 4, 1))
            else:
                res.failed += 1
                res.error = (proc.stderr or proc.stdout or f"rc={proc.returncode}")[-300:]
        except subprocess.TimeoutExpired:
            res.failed += 1
            res.error = "timeout"
    return res


def _run_omp(endpoint: str, model_id: str, prompt: str, iterations: int,
             max_tokens: int, profile_dir: Path) -> BenchResult:
    """Use omp's built-in `bench` against an isolated bench provider."""
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
                    "contextWindow": 32000, "maxTokens": max(2048, max_tokens),
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
    try:
        proc = subprocess.run(
            ["omp", "bench", f"bench/{model_id}", "--runs", str(iterations),
             "--profile", "chat", "--max-tokens", str(max_tokens), "--json"],
            capture_output=True, text=True, timeout=900, env=env, cwd=str(PROJECT_DIR),
        )
        out = proc.stdout.strip()
        start = out.find("{")
        if start < 0:
            res.error = (proc.stderr or out or "no JSON output")[-300:]
            return res
        data = json.loads(out[start:])
        models = data.get("models") or []
        if not models:
            res.error = "no model results in omp bench output"
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
        if ok_runs:
            res.ttft_ms = [float(r.get("ttftMs") or 0) for r in ok_runs]
            res.total_ms = [float(r.get("durationMs") or 0) for r in ok_runs]
            res.tokens = [int(r.get("outputTokens") or 0) for r in ok_runs]
        if res.ok == 0:
            res.error = (m.get("error") or "all runs failed")[-300:]
    except subprocess.TimeoutExpired:
        res.error = "timeout"
    except json.JSONDecodeError as e:
        res.error = f"JSON parse error: {e}"
    return res


def _run_sisyphus(endpoint: str, model_id: str, prompt: str, iterations: int,
                  max_tokens: int, config_dir: Path) -> BenchResult:
    """Run opencode with the Sisyphus (oh-my-opencode) agent against an isolated provider."""
    res = BenchResult(harness="sisyphus", framework="", model=model_id, iterations=iterations)
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "plugin": ["oh-my-openagent@latest"],
        "provider": {
            "bench": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Bench",
                "options": {"baseURL": f"{endpoint}/v1", "apiKey": "local"},
                "models": {model_id: {"name": "bench"}},
            }
        },
    }
    cfg_path = config_dir / "opencode.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    env = dict(os.environ)
    env["OPENCODE_CONFIG"] = str(cfg_path)
    for _ in range(iterations):
        unique = f"{prompt} [id:{uuid.uuid4().hex[:8]}]"
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["opencode", "run", unique, "-m", f"bench/{model_id}",
                 "--agent", "Sisyphus", "--format", "json"],
                capture_output=True, text=True, timeout=600, env=env, cwd=str(PROJECT_DIR),
            )
            total = time.time() - t0
            out_tokens = 0
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "step_finish":
                    out_tokens += int((ev.get("part", {}).get("tokens") or {}).get("output") or 0)
            if proc.returncode == 0 and out_tokens > 0:
                res.ok += 1
                res.total_ms.append(total * 1000)
                res.tokens.append(out_tokens)
            else:
                res.failed += 1
                res.error = (proc.stderr or proc.stdout or f"rc={proc.returncode}")[-300:]
        except subprocess.TimeoutExpired:
            res.failed += 1
            res.error = "timeout"
    return res


def _run_dsh(endpoint: str, model_id: str, prompt: str, iterations: int,
             max_tokens: int, config_dir: Path) -> BenchResult:
    """Run the DeepSeek Harness (dsh) headless profile against an isolated DSH_HOME."""
    res = BenchResult(harness="dsh", framework="", model=model_id, iterations=iterations)
    dsh_home = config_dir / "dsh-home"
    dsh_home.mkdir(parents=True, exist_ok=True)
    patch = (
        "- id: agent-default-model\n"
        "  config:\n"
        "    provider: deepseek-official\n"
        f"    model: {model_id}\n"
    )
    (dsh_home / "cordis.patch.yml").write_text(patch)
    env = dict(os.environ)
    env["DSH_HOME"] = str(dsh_home)
    env["DEEPSEEK_BASE_URL"] = f"{endpoint}/v1"
    env["DEEPSEEK_API_KEY"] = "local"
    env["DSH_TELEMETRY_DISABLED"] = "1"
    for _ in range(iterations):
        unique = f"{prompt} [id:{uuid.uuid4().hex[:8]}]"
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["npx", "-y", "@deepseek-ai/dsh", "--profile", "headless", unique],
                capture_output=True, text=True, timeout=600, env=env, cwd=str(PROJECT_DIR),
            )
            total = time.time() - t0
            out = proc.stdout.strip()
            # dsh prints "reasoning:\n<reasoning>\n\n<response>"; take the response.
            response = out
            if "reasoning:" in out:
                after = out.split("reasoning:", 1)[1]
                parts = after.split("\n\n", 1)
                if len(parts) == 2:
                    response = parts[1].strip()
            if proc.returncode == 0 and response:
                res.ok += 1
                res.total_ms.append(total * 1000)
                res.tokens.append(max(len(response) // 4, 1))
            else:
                res.failed += 1
                res.error = (proc.stderr or proc.stdout or f"rc={proc.returncode}")[-300:]
        except subprocess.TimeoutExpired:
            res.failed += 1
            res.error = "timeout"
    return res


def _run_bionic(endpoint: str, model_id: str, prompt: str, iterations: int,
                max_tokens: int) -> BenchResult:
    """Bionic (LM Studio GUI coding agent) harness.

    Bionic is a GUI app (``/Applications/Bionic.app``) that drives the LM Studio
    server on port 1234. Its full agent behaviour (tool use, multi-step) can
    only be exercised through the GUI, so this harness measures the LM Studio
    server that Bionic uses. The model must be loaded in LM Studio (via
    Bionic.app) before running; the first loaded model is used.
    """
    lms_ep = "http://127.0.0.1:1234"
    res = BenchResult(harness="bionic", framework="lmstudio", model="", iterations=iterations)
    try:
        r = httpx.get(f"{lms_ep}/v1/models", timeout=5)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
    except httpx.HTTPError as e:
        res.failed = iterations
        res.error = (f"bionic's LM Studio server ({lms_ep}) is not reachable ({e}). "
                     "Start Bionic.app, load the model in its GUI, then retry.")
        return res
    if not models:
        res.failed = iterations
        res.error = ("No model loaded in the LM Studio server. Load one via "
                     "Bionic.app, then retry.")
        return res
    target = model_id if model_id in models else models[0]
    res.model = target
    # Delegate to the raw streaming harness against the LM Studio server.
    inner = _run_raw(lms_ep, target, prompt, iterations, max_tokens)
    res.ok = inner.ok
    res.failed = inner.failed
    res.ttft_ms = inner.ttft_ms
    res.total_ms = inner.total_ms
    res.tokens = inner.tokens
    res.error = inner.error
    return res


HARNESS_RUNNERS = {
    "raw": _run_raw,
    "bionic": _run_bionic,
    "pi": _run_pi,
    "omp": _run_omp,
    "sisyphus": _run_sisyphus,
    "dsh": _run_dsh,
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
                  work_dir: Path | None = None) -> dict[str, Any]:
    """Run one (framework, model, harness) benchmark cell. Returns a result dict.

    ``warmup`` uncounted iterations are run first to absorb server/model warmup
    (important for server frameworks where the first request is slow).
    """
    own_manager = manager is None
    manager = manager or FrameworkManager()
    work_dir = work_dir or (BENCH_STATE_DIR / "work" / f"{framework}-{model_key}-{harness}")
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        if harness == "bionic":
            # Bionic drives the LM Studio server (port 1234), not the framework
            # endpoint, so skip the framework ensure/release for this cell.
            endpoint, model_id = "http://127.0.0.1:1234", ""
            res = _run_bionic(endpoint, model_id, prompt, iterations, max_tokens)
            summary = res.summary()
            summary.update({
                "framework": framework, "model": model_key, "model_id": res.model,
                "harness": harness, "endpoint": endpoint, "iterations": iterations,
                "warmup": 0, "max_tokens": max_tokens,
                "elapsed_s": round(time.time() - started, 1),
            })
            return summary
        endpoint, model_id = manager.ensure(framework, model_key)
        runner = HARNESS_RUNNERS.get(harness)
        if runner is None:
            return {"status": "failed", "error": f"no runner for harness '{harness}'"}
        # Warmup (uncounted) — absorbs first-request latency.
        for _ in range(max(warmup, 0)):
            _warmup_once(runner, harness, endpoint, model_id, prompt, max_tokens, work_dir)
        if harness in ("raw", "bionic"):
            res = runner(endpoint, model_id, prompt, iterations, max_tokens)
        elif harness in ("pi", "omp", "sisyphus", "dsh"):
            res = runner(endpoint, model_id, prompt, iterations, max_tokens,
                         work_dir / f"{harness}-config")
        else:
            return {"status": "failed", "error": f"no runner for harness '{harness}'"}
        summary = res.summary()
        summary.update({
            "framework": framework,
            "model": model_key,
            "model_id": model_id,
            "harness": harness,
            "endpoint": endpoint,
            "iterations": iterations,
            "warmup": warmup,
            "max_tokens": max_tokens,
            "elapsed_s": round(time.time() - started, 1),
        })
        return summary
    except Exception as e:  # noqa: BLE001 - report any failure in the result
        return {"status": "failed", "framework": framework, "model": model_key,
                "harness": harness, "error": str(e),
                "elapsed_s": round(time.time() - started, 1)}
    finally:
        if unload_after and harness != "bionic":
            try:
                manager.release(framework, model_key)
            except Exception:  # noqa: BLE001
                pass
        if own_manager:
            manager.shutdown()


def _warmup_once(runner, harness: str, endpoint: str, model_id: str,
                 prompt: str, max_tokens: int, work_dir: Path) -> None:
    """Run one uncounted iteration to warm up the server/model."""
    try:
        if harness == "raw":
            runner(endpoint, model_id, prompt, 1, max_tokens)
        elif harness in ("pi", "omp", "sisyphus", "dsh"):
            runner(endpoint, model_id, prompt, 1, max_tokens, work_dir / f"{harness}-config")
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
            if fw in ("mlx", "llamacpp"):
                frameworks[fw] = bool(art.get("path")) and Path(art["path"]).exists()
            elif fw == "ollama":
                frameworks[fw] = bool(art.get("gguf")) and Path(art["gguf"]).exists()
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
    p_run.add_argument("--max-tokens", type=int, default=128)
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
