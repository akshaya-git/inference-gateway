#!/usr/bin/env python3
"""
Local Inference Gateway — Phase 2 (oMLX runtime)
--------------------------------
OpenAI-compatible local router for Pi -> oMLX.

Features:
- Transparent /v1/chat/completions streaming proxy
- /v1/models pass-through
- Real-time browser dashboard
- TTFT / latency / token usage / tok/s metrics
- macOS memory monitoring
- Admission control when available memory falls below a safety threshold
- Optional memory-aware queue
- No proxy idle timeout
- Designed for future routing/tokenomics extensions
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Optional

import httpx
import psutil
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import benchmark_lab
from backends import BackendInterface, build_backend
from routing_logic import RoutingEngine, strip_routing_controls, text_content

ROUTING_ENGINE = RoutingEngine(os.path.join(os.path.dirname(__file__), "routing_rules.json"))

logger = logging.getLogger("inference-gateway")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROUTER_CONFIG_PATH = os.getenv(
    "ROUTER_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "router.yaml")
)


def load_router_config() -> dict[str, Any]:
    try:
        with open(ROUTER_CONFIG_PATH, encoding="utf-8") as source:
            config = yaml.safe_load(source) or {}
        if not isinstance(config, dict):
            raise ValueError("router configuration must be a mapping")
        return config
    except Exception as exc:
        raise RuntimeError(f"Cannot load router configuration {ROUTER_CONFIG_PATH}: {exc}") from exc


ROUTER_CONFIG = load_router_config()
ROUTE_CONFIG = ROUTER_CONFIG.get("routes") or {}
ROUTING_CONFIG = ROUTER_CONFIG.get("routing") or {}

HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PORT = int(os.getenv("PROXY_PORT", "9000"))
MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", "128000"))

OMLX_UPSTREAM = os.getenv("OMLX_UPSTREAM", "http://127.0.0.1:8000")
OMLX_API_KEY = os.getenv("OMLX_API_KEY", "")


def derive_route_state(config: dict[str, Any]) -> dict[str, Any]:
    """Derive route state from a parsed router config (pure, testable).

    This is the model-swappability seam: changing models or backends in
    router.yaml flows through here with no code changes. Env overrides:
    MOE_MODEL / DENSE_MODEL (model IDs), OMLX_UPSTREAM (default upstream),
    OMLX_API_KEY (default key), MOE_UPSTREAM / DENSE_UPSTREAM (per-route
    upstream). Routes with identical backend specs share one instance
    (one residency domain).
    """
    route_config = config.get("routes") or {}
    backends_cfg = config.get("backends") or {}
    default_upstream = os.getenv("OMLX_UPSTREAM", "http://127.0.0.1:8000")
    api_key = os.getenv("OMLX_API_KEY", "")

    model_ids = {
        route: os.getenv(
            f"{route.upper()}_MODEL",
            str((route_config.get(route) or {}).get("model", "")),
        )
        for route in ("moe", "dense")
    }

    shared: dict[tuple, BackendInterface] = {}
    backends: dict[str, BackendInterface] = {}
    for route in ("moe", "dense"):
        spec = dict(backends_cfg.get(route) or {})
        env_upstream = os.getenv(f"{route.upper()}_UPSTREAM", "")
        if env_upstream:
            spec["upstream"] = env_upstream
        spec.setdefault("upstream", default_upstream)
        if api_key and not spec.get("api_key"):
            spec["api_key"] = api_key
        key = (
            str(spec.get("type", "omlx")).lower(),
            str(spec.get("upstream", "")).rstrip("/"),
            str(spec.get("api_key", "") or ""),
        )
        if key not in shared:
            shared[key] = build_backend(spec)
        backends[route] = shared[key]

    model_capabilities = {
        route: {
            "context_window": int(((route_config.get(route) or {}).get("capabilities") or {}).get("context_window") or 0),
            "max_output": int(((route_config.get(route) or {}).get("capabilities") or {}).get("max_output") or 0),
            "vision": bool(((route_config.get(route) or {}).get("capabilities") or {}).get("vision", False)),
        }
        for route in ("moe", "dense")
    }
    return {
        "model_ids": model_ids,
        "model_routes": {value: key for key, value in model_ids.items()},
        "model_capabilities": model_capabilities,
        "backends": backends,
        "endpoints": {route: backend.upstream for route, backend in backends.items()},
    }


_ROUTE_STATE = derive_route_state(ROUTER_CONFIG)
MOE_MODEL = _ROUTE_STATE["model_ids"]["moe"]
DENSE_MODEL = _ROUTE_STATE["model_ids"]["dense"]
MODEL_IDS = _ROUTE_STATE["model_ids"]
ROUTE_BACKENDS = _ROUTE_STATE["backends"]
MODEL_ENDPOINTS = _ROUTE_STATE["endpoints"]
RUNTIME_MODEL_IDS = dict(MODEL_IDS)
MODEL_ROUTES = _ROUTE_STATE["model_routes"]
BENCHMARK_MODEL_ALIASES = {
    "benchmark-moe": "moe",
    "benchmark-dense": "dense",
}
GATEWAY_MODEL_ALIASES = {"gateway-moe": "moe", "gateway-dense": "dense"}
MODEL_ROUTES.update(BENCHMARK_MODEL_ALIASES)
MODEL_ROUTES.update(GATEWAY_MODEL_ALIASES)
AUTO_MODEL_ALIASES = {
    value.strip() for value in os.getenv(
        "AUTO_MODEL_ALIASES", "auto,local-model,inference-gateway,gateway-auto"
    ).split(",") if value.strip()
}
STATE_DIR = os.getenv("INFERENCE_STACK_STATE", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".inference-stack"))
ROUTING_ENABLED = os.getenv("ROUTING_ENABLED", "true").lower() == "true"
KEEP_MODELS_LOADED = os.getenv("KEEP_MODELS_LOADED", "true").lower() == "true"
SWAP_TIMEOUT_SEC = float(os.getenv("SWAP_TIMEOUT_SEC", "240"))

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "300"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "128"))
BENCHMARK_HISTORY_MAX = int(os.getenv("BENCHMARK_HISTORY_MAX", "100"))
BENCHMARK_FILE = os.path.join(STATE_DIR, "benchmarks.json") if "STATE_DIR" in globals() else ""
BENCHMARK_WORK_DIR = os.path.join(STATE_DIR, "benchmark-workspaces")
PI_EXECUTABLE = os.getenv("PI_EXECUTABLE", shutil.which("pi") or "/opt/homebrew/bin/pi")
PI_BENCHMARK_TIMEOUT_SEC = int(os.getenv("PI_BENCHMARK_TIMEOUT_SEC", "1800"))
PI_EVENT_STREAM_LIMIT_BYTES = int(os.getenv("PI_EVENT_STREAM_LIMIT_BYTES", str(32 * 1024 * 1024)))

# Keep enough recent context for the one routing decision made at a task boundary.
ROUTER_CONTEXT_CHARS = int(os.getenv("ROUTER_CONTEXT_CHARS", str(ROUTING_CONFIG.get("context_characters", 24000))))
FALLBACK_ROUTE = str(ROUTING_CONFIG.get("fallback_route", "moe"))
if FALLBACK_ROUTE not in ("moe", "dense"):
    raise RuntimeError("routing.fallback_route must be 'moe' or 'dense'")

# Per-route capability metadata (context window, max output, vision). Drives
# capability-aware routing adjustments and is reported in /metrics.
MODEL_CAPABILITIES = _ROUTE_STATE["model_capabilities"]

# Start protecting the machine when available memory falls below this.
# 3 GiB is the user's requested safety zone.
MEMORY_GUARD_GB = float(os.getenv("MEMORY_GUARD_GB", "3"))

# Below this level, do not admit a new model request.
MEMORY_HARD_GB = float(os.getenv("MEMORY_HARD_GB", "1.5"))

# Polling interval for memory watcher.
MEMORY_POLL_SEC = float(os.getenv("MEMORY_POLL_SEC", "1"))

# Keep only this many completed requests in RAM.
HISTORY_MAX = int(os.getenv("HISTORY_MAX", "500"))

# Match oMLX's single-user scheduler. Router and selected backend run
# sequentially, which also keeps MTP on its optimal batch=1 path.
MAX_ACTIVE_REQUESTS = int(os.getenv("MAX_ACTIVE_REQUESTS", "1"))

# If true, requests wait for memory to recover rather than immediately failing.
QUEUE_ON_MEMORY_PRESSURE = os.getenv("QUEUE_ON_MEMORY_PRESSURE", "true").lower() == "true"

app = FastAPI(title="Local Inference Gateway", version="2.1.1-phase2-omlx-ttft")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class Metric:
    id: str
    ts: float
    model: str = ""
    status: int = 0
    ttft_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    gen_tps: Optional[float] = None
    finish_reason: str = ""
    error: str = ""
    max_tokens: Optional[int] = None
    streaming: bool = False
    client_wait_ms: Optional[float] = None
    peak_available_gb: Optional[float] = None
    low_memory_events: int = 0
    requested_model: str = ""
    route: str = ""
    route_reason: str = ""
    router_confidence: Optional[float] = None
    router_ms: Optional[float] = None
    swap_ms: Optional[float] = None
    cache_hit: bool = False
    workload: str = "interactive"
    compute_score: Optional[float] = None
    task_type: str = ""
    effort: str = ""
    thinking: Optional[bool] = None
    effective_max_tokens: Optional[int] = None
    routing_level: Optional[int] = None
    routing_rules: str = ""


history = deque(maxlen=HISTORY_MAX)
active: dict[str, dict[str, Any]] = {}
lock = asyncio.Lock()
request_slots = asyncio.Semaphore(MAX_ACTIVE_REQUESTS)
swap_lock = asyncio.Lock()
backend_slot = asyncio.Semaphore(1)
# Serializes semantic routing transitions. Unlike backend_slot, this protects
# the pre-dispatch router call, which may need to restore the MoE before it can
# classify a genuinely new task.
router_transition_lock = asyncio.Lock()
response_cache: dict[str, dict[str, Any]] = {}
cache_stats = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}
benchmark_history = deque(maxlen=BENCHMARK_HISTORY_MAX)
benchmark_jobs: dict[str, dict[str, Any]] = {}
benchmark_lock = asyncio.Lock()
manual_pin_lock = asyncio.Lock()
last_activity_at = time.time()
backend_status_cache: dict[str, dict[str, Any]] = {
    route: {"updated_at": 0.0, "models": {}} for route in ("moe", "dense")
}

BENCHMARK_SUITES = {
    "quick": {
        "name": "Quick comparison",
        "description": "Short explanation completed by a fresh Pi coding-agent session.",
        "prompt": "Explain in practical terms how an inference gateway should choose between a fast MoE model and a dense model. Use five concise bullet points.",
        "max_tokens": 384,
        "thinking": False,
        "reasoning_effort": "low",
        "temperature": 0.2,
    },
    "coding_hitl": {
        "name": "Bill-splitter HITL",
        "description": "Pi builds a real HTML artifact in an isolated workspace for human testing.",
        "prompt": "Build a HTML file that is a bill-splitter that handles tax, tip, multiple people and rounding correctly.",
        "max_tokens": 32144,
        "thinking": True,
        "reasoning_effort": "medium",
        "temperature": 0.2,
    },
    "reasoning": {
        "name": "Systems reasoning",
        "description": "Architecture analysis completed by a fresh Pi coding-agent session.",
        "prompt": "A 64 GB Apple Silicon machine must switch between a fast 35B-A3B MoE workhorse and a dense 27B specialist. Design a safe switching policy that minimizes latency while preventing swap thrashing. State assumptions, thresholds, and failure recovery.",
        "max_tokens": 4096,
        "thinking": True,
        "reasoning_effort": "medium",
        "temperature": 0.2,
    },
}


BENCHMARK_SUITES["coding_hitl"]["artifact_extension"] = "html"
BENCHMARK_SUITES.update({'browser_tetris': {'name': 'Browser Tetris',
                    'description': 'Keyboard and touch controls; wall rotation; line clear and '
                                   'score; pause; restart after game over.',
                    'prompt': 'Build a polished browser-based Tetris game in one self-contained '
                              'artifact.html. Implement all seven tetrominoes, rotation with wall '
                              'kicks, collision detection, line clearing, score, increasing '
                              'levels, next-piece preview, pause/resume, restart, and game over. '
                              'Include keyboard controls, on-screen touch buttons, and a visible '
                              'controls guide. Use no external libraries or assets. Test rotations '
                              'near walls, line clearing, pause, and restarting after game over.',
                    'artifact_extension': 'html',
                    'review_checklist': 'Keyboard and touch controls; wall rotation; line clear '
                                        'and score; pause; restart after game over.',
                    'max_tokens': 32000,
                    'thinking': True,
                    'reasoning_effort': 'medium'},
 'svg_portrait': {'name': 'SVG historical portrait',
                  'description': 'Recognizable likeness without reading the caption; original '
                                 'vector shapes; clean scaling; valid SVG.',
                  'prompt': 'Create a recognizable, carefully composed vector portrait of Mahatma '
                            'Gandhi as a standalone artifact.svg. Draw the likeness using original '
                            'SVG paths and shapes; do not use embedded raster images, external '
                            'assets, scripts, or a generic face with only a name label. Include a '
                            'descriptive title and description, a scalable viewBox, and a '
                            'restrained color palette. Capture distinctive facial features and '
                            'clothing respectfully. Validate that the SVG parses and scales '
                            'correctly.',
                  'artifact_extension': 'svg',
                  'review_checklist': 'Recognizable likeness without reading the caption; original '
                                      'vector shapes; clean scaling; valid SVG.',
                  'max_tokens': 32000,
                  'thinking': True,
                  'reasoning_effort': 'medium',
                  'subjects': ['Mahatma Gandhi',
                               'Abraham Lincoln',
                               'Mother Teresa',
                               'Nelson Mandela'],
                  'prompt_template': 'Create a recognizable, carefully composed vector portrait of '
                                     '{figure} as a standalone artifact.svg. Draw the likeness '
                                     'using original SVG paths and shapes; do not use embedded '
                                     'raster images, external assets, scripts, or a generic face '
                                     'with only a name label. Include a descriptive title and '
                                     'description, a scalable viewBox, and a restrained color '
                                     'palette. Capture distinctive facial features and clothing '
                                     'respectfully. Validate that the SVG parses and scales '
                                     'correctly.'},
 'kanban_board': {'name': 'Interactive Kanban board',
                  'description': 'Create/edit/delete; drag and keyboard move; filtering; correct '
                                 'counts; valid export/import round trip.',
                  'prompt': 'Build a polished self-contained artifact.html for a Kanban board with '
                            'To Do, In Progress, and Done columns. Support adding, editing, '
                            'deleting, and moving cards; priorities; text search; and live column '
                            'counts. Provide both drag-and-drop and keyboard-accessible move '
                            'controls. Include JSON export/import with validation so data survives '
                            'through exported files, without relying on browser storage. Include '
                            'sample cards and empty states. Use no external dependencies. Test '
                            'card moves, filtering, and export/import round trips.',
                  'artifact_extension': 'html',
                  'review_checklist': 'Create/edit/delete; drag and keyboard move; filtering; '
                                      'correct counts; valid export/import round trip.',
                  'max_tokens': 32000,
                  'thinking': True,
                  'reasoning_effort': 'medium'},
 'csv_dashboard': {'name': 'CSV data dashboard',
                   'description': 'Quoted CSV fields; invalid rows; correct revenue totals; '
                                  'synchronized filters and charts; empty-state handling.',
                   'prompt': 'Build a polished self-contained artifact.html that lets a user paste '
                             'or upload CSV sales data and inspect a table, filters, and SVG '
                             'charts. Include sample data with date, product, region, quantity, '
                             'and unit_price. Correctly parse quoted commas and escaped quotes, '
                             'reject malformed rows clearly, compute revenue, and update totals '
                             'and charts together when filtered. Handle empty data, zero values, '
                             'and invalid numbers. Use no external libraries or network calls. '
                             'Test the parser and filtered totals.',
                   'artifact_extension': 'html',
                   'review_checklist': 'Quoted CSV fields; invalid rows; correct revenue totals; '
                                       'synchronized filters and charts; empty-state handling.',
                   'max_tokens': 32000,
                   'thinking': True,
                   'reasoning_effort': 'medium'},
 'pathfinding_visualizer': {'name': 'Pathfinding visualizer',
                            'description': 'Both algorithms find shortest paths; no-path state; '
                                           'pause/reset; accurate path length; responsive '
                                           'controls.',
                            'prompt': 'Build a polished self-contained artifact.html that '
                                      'visualizes breadth-first search and A* on an editable grid '
                                      'with unit movement cost and no diagonal steps. Let users '
                                      'position start and goal, draw and erase walls, choose '
                                      'algorithm and speed, run/pause/reset, and inspect visited '
                                      'nodes and final path length. Keep start/goal valid and show '
                                      'a clear no-path state. Use no external libraries. Test '
                                      'shortest-path length on a known grid, unreachable targets, '
                                      'and reset during animation.',
                            'artifact_extension': 'html',
                            'review_checklist': 'Both algorithms find shortest paths; no-path '
                                                'state; pause/reset; accurate path length; '
                                                'responsive controls.',
                            'max_tokens': 32000,
                            'thinking': True,
                            'reasoning_effort': 'medium'}})


def benchmark_busy() -> bool:
    return benchmark_lock.locked() or any(
        job.get("status") in ("queued", "running")
        for job in benchmark_jobs.values()
    )

memory_state = {
    "available_gb": 0.0,
    "used_gb": 0.0,
    "total_gb": 0.0,
    "percent": 0.0,
    "guard_gb": MEMORY_GUARD_GB,
    "hard_gb": MEMORY_HARD_GB,
    "pressure": False,
    "hard_pressure": False,
    "last_update": 0.0,
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_ms() -> float:
    return time.perf_counter() * 1000.0


def macos_memory_available_bytes() -> int:
    """
    macOS has no Linux-style /proc/meminfo. psutil's available memory is
    suitable for admission control and is based on VM statistics.
    """
    return int(psutil.virtual_memory().available)


def update_memory_state():
    vm = psutil.virtual_memory()
    available = vm.available / (1024 ** 3)
    total = vm.total / (1024 ** 3)
    used = total - available

    memory_state.update({
        "available_gb": round(available, 2),
        "used_gb": round(used, 2),
        "total_gb": round(total, 2),
        "percent": round(vm.percent, 1),
        "guard_gb": MEMORY_GUARD_GB,
        "hard_gb": MEMORY_HARD_GB,
        "pressure": available <= MEMORY_GUARD_GB,
        "hard_pressure": available <= MEMORY_HARD_GB,
        "last_update": time.time(),
    })


def extract_usage(obj: dict) -> dict:
    usage = obj.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": details.get("cached_tokens")
        if isinstance(details, dict) else None,
    }


def extract_finish_reason(obj: dict) -> str:
    try:
        return (obj.get("choices") or [{}])[0].get("finish_reason") or ""
    except Exception:
        return ""


def stream_delta_text(obj: dict) -> tuple[str, str]:
    """Extract actual generated answer/reasoning text from an SSE event.

    OpenAI-compatible servers commonly emit an immediate role-only delta such
    as {"role":"assistant","content":""}. oMLX can also emit keep-alive and
    usage events before generation. None of those count as time-to-first-token.
    """
    try:
        choice = (obj.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
    except (AttributeError, IndexError, TypeError):
        return "", ""

    def text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    candidate = item.get("text") or item.get("content")
                    if isinstance(candidate, str):
                        parts.append(candidate)
            return "".join(parts)
        return ""

    answer = text(delta.get("content")) or text(delta.get("text"))
    reasoning = (
        text(delta.get("reasoning_content"))
        or text(delta.get("reasoning"))
    )
    return answer, reasoning


def cache_key(body: dict) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cache_get(key: str) -> Optional[dict]:
    if not CACHE_ENABLED:
        return None
    item = response_cache.get(key)
    if not item:
        cache_stats["misses"] += 1
        return None
    if time.time() - item["stored_at"] > CACHE_TTL_SEC:
        response_cache.pop(key, None)
        cache_stats["misses"] += 1
        cache_stats["evictions"] += 1
        return None
    cache_stats["hits"] += 1
    return item


def cache_put(key: str, payload: dict, status: int):
    if not CACHE_ENABLED or status < 200 or status >= 300:
        return
    if len(response_cache) >= CACHE_MAX_ENTRIES:
        oldest = min(response_cache, key=lambda k: response_cache[k]["stored_at"])
        response_cache.pop(oldest, None)
        cache_stats["evictions"] += 1
    response_cache[key] = {
        "payload": payload,
        "status": status,
        "stored_at": time.time(),
    }
    cache_stats["stores"] += 1


def compute_score(latency_ms: Optional[float], prompt_tokens: Optional[int],
                  completion_tokens: Optional[int], swap_ms: Optional[float]) -> float:
    """Relative local-compute score; intentionally not a currency estimate."""
    latency = max(0.0, latency_ms or 0.0) / 1000.0
    swap = max(0.0, swap_ms or 0.0) / 1000.0
    prompt = max(0, prompt_tokens or 0)
    completion = max(0, completion_tokens or 0)
    return round(latency + (swap * 0.5) + (prompt / 2000.0) + (completion / 500.0), 2)


def swap_out_bytes() -> Optional[int]:
    """Return cumulative swap-out bytes when macOS exposes them to psutil."""
    try:
        return int(psutil.swap_memory().sout)
    except (OSError, AttributeError, NotImplementedError):
        try:
            result = subprocess.run(
                ["/usr/bin/vm_stat"], capture_output=True, text=True,
                check=True, timeout=3,
            )
            page_match = re.search(r"page size of (\d+) bytes", result.stdout)
            swap_match = re.search(r"^Swapouts:\s+(\d+)\.?", result.stdout, re.MULTILINE)
            if page_match and swap_match:
                return int(page_match.group(1)) * int(swap_match.group(1))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return None


def save_benchmarks():
    os.makedirs(STATE_DIR, exist_ok=True)
    temp_path = f"{BENCHMARK_FILE}.tmp"
    with open(temp_path, "w", encoding="utf-8") as output:
        json.dump(list(benchmark_history), output, indent=2)
    os.replace(temp_path, BENCHMARK_FILE)


def load_benchmarks():
    try:
        with open(BENCHMARK_FILE, encoding="utf-8") as source:
            items = json.load(source)
        if isinstance(items, list):
            supported = [item for item in items if item.get("suite") in BENCHMARK_SUITES]
            benchmark_history.extend(supported[-BENCHMARK_HISTORY_MAX:])
            if len(supported) != len(items):
                save_benchmarks()
    except (OSError, ValueError, TypeError):
        pass


def process_pid_running(name: str) -> bool:
    try:
        with open(os.path.join(STATE_DIR, "pids", f"{name}.pid"), encoding="utf-8") as pid_file:
            pid = int(pid_file.read())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


async def backend_models(route: str, force: bool = False) -> dict[str, dict[str, Any]]:
    """Return the route backend's model load state, cached briefly for polling."""
    cache = backend_status_cache[route]
    if not force and time.time() - float(cache["updated_at"]) < 0.5:
        return cache["models"]
    try:
        models = await ROUTE_BACKENDS[route].list_models()
        cache.update({"updated_at": time.time(), "models": models})
    except Exception:
        # A transient admin/status failure must not erase the last known state.
        pass
    return cache["models"]


def invalidate_backend_status(route: str) -> None:
    backend_status_cache[route]["updated_at"] = 0.0


async def pid_running(name: str) -> bool:
    """Compatibility name retained for routing code; models live in backends."""
    if name in RUNTIME_MODEL_IDS:
        states = await backend_models(name)
        return bool(states.get(RUNTIME_MODEL_IDS[name], {}).get("loaded"))
    return process_pid_running(name)


async def model_state() -> dict:
    states: dict[str, dict[str, Any]] = {}
    for route in ("moe", "dense"):
        states.update(await backend_models(route))
    return {
        route: {
            "model": RUNTIME_MODEL_IDS[route],
            "endpoint": MODEL_ENDPOINTS[route],
            "running": bool(states.get(RUNTIME_MODEL_IDS[route], {}).get("loaded")),
            "loading": bool(states.get(RUNTIME_MODEL_IDS[route], {}).get("is_loading")),
            "runtime": ROUTE_BACKENDS[route].name,
        }
        for route in ("moe", "dense")
    }


EFFORT_TOKENS = {"fast": 4096, "balanced": 16384, "high": 32144}

# Versioned guidance for coding-agent harnesses. The gateway injects this only
# for coding requests, after routing, so ordinary chat is unchanged. Keep the
# policy concise: every token is repeated throughout a long agent session.
AGENT_EFFICIENCY_POLICY_VERSION = "coding-agent-v1"
AGENT_EFFICIENCY_POLICY = (
    f"Inference Gateway execution policy ({AGENT_EFFICIENCY_POLICY_VERSION}): "
    "For agentic coding work, batch related read-only inspections and independent "
    "checks when practical. Run the project's canonical focused tests before "
    "inventing ad-hoc assertions. When an ad-hoc check is necessary, validate "
    "observable behavior rather than source-text spelling, quoting, or helper names. "
    "Do not retry substantially the same failed diagnostic more than twice: inspect "
    "the evidence, correct the check if it is invalid, otherwise implement the most "
    "likely fix and rerun the focused test. Validate focused behavior before expensive "
    "end-to-end evaluations. Preserve checkpoints after coherent milestones. Never "
    "skip required correctness, security, or user-requested validation merely to save turns."
)


def normalize_policy(candidate: dict, fallback: Optional[dict] = None) -> dict:
    base = dict(fallback or {
        "route": "moe", "confidence": 0.70, "reason": "safe default",
        "task_type": "general", "effort": "balanced", "thinking": True,
        "max_tokens": EFFORT_TOKENS["balanced"],
    })
    route = str(candidate.get("route", base["route"])).lower()
    effort = str(candidate.get("effort", base["effort"])).lower()
    if route not in ("moe", "dense"):
        route = base["route"]
    if effort not in EFFORT_TOKENS:
        effort = base["effort"]
    try:
        confidence = max(0.0, min(1.0, float(candidate.get("confidence", base["confidence"]))))
    except (TypeError, ValueError):
        confidence = base["confidence"]
    requested_tokens = candidate.get("max_tokens", EFFORT_TOKENS[effort])
    try:
        max_tokens = max(512, min(32144, int(requested_tokens)))
    except (TypeError, ValueError):
        max_tokens = EFFORT_TOKENS[effort]
    max_tokens = max(max_tokens, EFFORT_TOKENS[effort])
    return {
        "route": route,
        "confidence": confidence,
        "reason": str(candidate.get("reason", base["reason"]))[:240],
        "task_type": str(candidate.get("task_type", base["task_type"]))[:64],
        "effort": effort,
        "thinking": bool(candidate.get("thinking", effort != "fast")),
        "reasoning_effort": str(candidate.get(
            "reasoning_effort", "low" if effort == "fast" else "medium"
        )).lower() if str(candidate.get(
            "reasoning_effort", "low" if effort == "fast" else "medium"
        )).lower() in ("low", "medium", "high") else "medium",
        "max_tokens": max_tokens,
    }


async def wait_for_backend_drain_before_routing(req_id: Optional[str]) -> None:
    """Never let a routing-time model swap interrupt active inference.

    Streaming responses outlive the FastAPI handler that created them, so the
    request admission semaphore alone cannot protect this transition. Active
    entries remain present until their stream/non-stream request completes.
    """
    while True:
        async with lock:
            blockers = [
                item for item_id, item in active.items()
                if item_id != req_id and item.get("route") in ("moe", "dense")
            ]
        if not blockers:
            return
        await asyncio.sleep(0.1)


async def wait_for_model_transition(backend: BackendInterface, model_id: str, loaded: bool) -> dict:
    """Admin unload may return HTTP 202 while activity drains."""
    deadline = time.monotonic() + SWAP_TIMEOUT_SEC
    while True:
        states = await backend.list_models()
        for r, b in ROUTE_BACKENDS.items():
            if b is backend:
                backend_status_cache[r].update({"updated_at": time.time(), "models": states})
        entry = states.get(model_id)
        if entry is None:
            raise RuntimeError(f"backend model disappeared during transition: {model_id}")
        if bool(entry.get("loaded")) == loaded and not entry.get("is_loading"):
            return states
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {model_id} loaded={loaded}; "
                f"loaded={entry.get('loaded')}, is_loading={entry.get('is_loading')}"
            )
        await asyncio.sleep(0.25)


async def ensure_route(route: str) -> float:
    """Ensure the requested route's model is loaded on its backend.

    Query fresh state and fail closed on admin errors. When both routes share
    one backend instance and KEEP_MODELS_LOADED is false, the other model is
    unloaded first to release memory. Backends without a residency API
    (generic OpenAI servers) are assumed to keep their model loaded.
    """
    if route not in MODEL_IDS:
        raise ValueError(f"Unknown route: {route}")
    backend = ROUTE_BACKENDS[route]
    other_route = "dense" if route == "moe" else "moe"
    shared = ROUTE_BACKENDS[other_route] is backend
    started = now_ms()
    changed = False
    async with swap_lock:
        states = await backend.list_models()
        target = MODEL_IDS[route]
        if target not in states:
            raise RuntimeError(f"backend {backend.name} has not discovered model {target}")
        if any(state.get("is_loading") for state in states.values()):
            raise RuntimeError("A backend is already loading; retry after it finishes")
        if shared and not KEEP_MODELS_LOADED:
            other = MODEL_IDS[other_route]
            if states.get(other, {}).get("loaded"):
                await backend.unload_model(other)
                changed = True
                states = await wait_for_model_transition(backend, other, loaded=False)
        if not states[target].get("loaded"):
            if not backend.supports_residency:
                raise RuntimeError(
                    f"backend {backend.name} does not list model {target} and has no "
                    "admin load API; start the server with that model"
                )
            await backend.load_model(target)
            changed = True
        states = await wait_for_model_transition(backend, target, loaded=True)
        if shared and not KEEP_MODELS_LOADED and states.get(MODEL_IDS[other_route], {}).get("loaded"):
            raise RuntimeError(f"Other backend became loaded during transition: {MODEL_IDS[other_route]}")
    return now_ms() - started if changed else 0.0


async def prepare_backend(route: str, reason: str) -> tuple[str, str, float]:
    return route, reason, await ensure_route(route)


def capability_adjustment(route: str, body: dict) -> tuple[str, str]:
    """Capability-aware route adjustment: vision and context window.

    Returns (route, note). The note is appended to the route reason when the
    route was adjusted or a capability gap is expected.
    """
    caps = MODEL_CAPABILITIES.get(route, {})
    other = "dense" if route == "moe" else "moe"
    other_caps = MODEL_CAPABILITIES.get(other, {})
    notes = []
    has_image = any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for message in body.get("messages") or []
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
    )
    if has_image and not caps.get("vision", False):
        if other_caps.get("vision", False):
            notes.append(f"capability: image content requires vision; switched {route} -> {other}")
            route = other
        else:
            notes.append("capability: image content but no vision-capable model configured")
    est_tokens = sum(len(text_content(m)) for m in body.get("messages") or []) // 4
    window = caps.get("context_window") or 0
    if window and est_tokens > window:
        other_window = other_caps.get("context_window") or 0
        if other_window > window:
            notes.append(f"capability: ~{est_tokens} tokens exceeds {route} window {window}; switched to {other}")
            route = other
        else:
            notes.append(f"capability: ~{est_tokens} tokens exceeds {route} window {window}")
    return route, "; ".join(notes)


async def choose_policy(body: dict, req_id: Optional[str] = None) -> tuple[dict, float]:
    started = now_ms()
    explicit = MODEL_ROUTES.get(str(body.get("model", "")))
    if not ROUTING_ENABLED:
        # Routing disabled: pin to the explicit route or the fallback route.
        decision = {
            "route": explicit or FALLBACK_ROUTE, "complexity": None, "code_related": False,
            "matched_rules": [], "modifiers": [], "invalid_controls": [],
            "rules_version": ROUTING_ENGINE.rules["version"],
            "source": "routing disabled", "reason": "routing disabled; pinned to fallback route",
        }
    else:
        decision = ROUTING_ENGINE.route(body, explicit=explicit)
    policy = normalize_policy({**decision, "confidence": 1.0,
                               "task_type": "deterministic", "thinking": True})
    policy.update(decision)
    route, note = capability_adjustment(policy["route"], body)
    if note:
        policy["route"] = route
        policy["reason"] = f"{policy['reason']}; {note}"
    return policy, now_ms() - started


def apply_request_policy(body: dict, policy: dict, automatic: bool) -> dict:
    """Forward the client request unchanged except for the selected backend ID."""
    routed = strip_routing_controls(body)
    routed["model"] = MODEL_IDS[policy["route"]]
    return routed


def manual_pin_status() -> dict:
    return {"active": False, "route": "", "remaining_sec": 0.0, "duration_sec": 0}


async def wait_for_memory(req_id: str, planned_route: str = ""):
    """
    Admission control:
    - hard pressure: wait until memory is above hard threshold
    - guard pressure: normally wait until memory rises above guard
    This does not interrupt an active generation. It protects the next request.
    """
    waited_start = now_ms()

    while True:
        update_memory_state()

        # Either transition first unloads the other large backend, releasing memory.
        # Do not deadlock by waiting for that memory before allowing the swap.
        if not KEEP_MODELS_LOADED and planned_route in MODEL_IDS and await pid_running("dense" if planned_route == "moe" else "moe"):
            return now_ms() - waited_start

        async with lock:
            if req_id in active:
                active[req_id]["memory_available_gb"] = memory_state["available_gb"]
                active[req_id]["memory_pressure"] = memory_state["pressure"]

        if not memory_state["hard_pressure"] and not memory_state["pressure"]:
            return now_ms() - waited_start

        if not QUEUE_ON_MEMORY_PRESSURE:
            raise RuntimeError(
                f"Memory pressure: only {memory_state['available_gb']:.2f} GiB available"
            )

        await asyncio.sleep(MEMORY_POLL_SEC)


async def record(metric: Metric):
    global last_activity_at
    async with lock:
        history.append(asdict(metric))
        active.pop(metric.id, None)
        last_activity_at = time.time()


async def memory_watcher():
    while True:
        update_memory_state()

        async with lock:
            for item in active.values():
                item["memory_available_gb"] = memory_state["available_gb"]
                item["memory_pressure"] = memory_state["pressure"]
                if memory_state["pressure"]:
                    item["low_memory_events"] = item.get("low_memory_events", 0) + 1

        await asyncio.sleep(MEMORY_POLL_SEC)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    update_memory_state()
    load_benchmarks()
    if KEEP_MODELS_LOADED:
        for route in MODEL_IDS:
            await ensure_route(route)
        states: dict[str, dict[str, Any]] = {}
        for route in MODEL_IDS:
            states.update(await backend_models(route, force=True))
        if not all(states.get(mid, {}).get("loaded") for mid in MODEL_IDS.values()):
            raise RuntimeError("backends could not retain both models; check their memory limits")
    asyncio.create_task(memory_watcher())


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    update_memory_state()
    return {
        "ok": True,
        "gateway": "phase2",
        "ttft_definition": "upstream dispatch to first generated content/reasoning token",
        "models": await model_state(),
        "active_requests": len(active),
        "max_active_requests": MAX_ACTIVE_REQUESTS,
        "keep_models_loaded": KEEP_MODELS_LOADED,
        "memory": memory_state,
        "routing_mode": "deterministic complexity lookup; code above level 4 uses dense",
        "router_config": ROUTER_CONFIG_PATH,
    }


@app.get("/metrics")
async def metrics():
    update_memory_state()
    async with lock:
        return {
            "memory": memory_state,
            "active": list(active.values()),
            "history": list(history),
            "models": await model_state(),
            "cache": {**cache_stats, "entries": len(response_cache), "ttl_sec": CACHE_TTL_SEC},
            "route_affinity": {
                "task_routes": 0,
                "dense_pins": 0,
                "ttl_sec": None,
                "enabled": False,
            },
            "manual_pin": manual_pin_status(),
            "config": {
                "routing_enabled": ROUTING_ENABLED,
                "auto_model_aliases": sorted(AUTO_MODEL_ALIASES),
                "max_active_requests": MAX_ACTIVE_REQUESTS,
        "keep_models_loaded": KEEP_MODELS_LOADED,
                "memory_guard_gb": MEMORY_GUARD_GB,
                "memory_hard_gb": MEMORY_HARD_GB,
                "queue_on_memory_pressure": QUEUE_ON_MEMORY_PRESSURE,
                "route_affinity_ttl_sec": None,
                "router_context_chars": ROUTER_CONTEXT_CHARS,
                "manual_dense_pin_sec": 0,
                "idle_baseline_quiet_sec": None,
                "routing_mode": "deterministic complexity lookup; code above level 4 uses dense",
                "router_config": ROUTER_CONFIG_PATH,
                "fallback_route": FALLBACK_ROUTE,
                "model_capabilities": MODEL_CAPABILITIES,
                "routing_rules_version": ROUTING_ENGINE.rules["version"],
                "routing_rules_error": ROUTING_ENGINE.last_error,
            },
            "benchmarks": {
                "jobs": list(benchmark_jobs.values()),
                "history": list(benchmark_history),
                "suites": BENCHMARK_SUITES,
            },
        }


@app.post("/control/model/{route}")
async def control_model(route: str):
    if route not in ("moe", "dense"):
        return JSONResponse({"error": "route must be moe or dense"}, status_code=400)
    if benchmark_busy():
        return JSONResponse(
            {"error": "manual model switch refused while a benchmark is active"}, status_code=409
        )
    if active:
        return JSONResponse(
            {"error": "model switch refused while requests are active"}, status_code=409
        )
    try:
        async with manual_pin_lock:
            elapsed = await ensure_route(route)
        return {
            "ok": True, "route": route, "swap_ms": round(elapsed, 1),
            "models": await model_state(), "manual_pin": manual_pin_status(),
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.post("/control/cache/clear")
async def clear_cache():
    removed = len(response_cache)
    response_cache.clear()
    return {"ok": True, "removed": removed}


@app.post("/control/affinity/clear")
async def clear_affinity():
    async with lock:
        if active or benchmark_busy():
            return JSONResponse({"error": "gateway is busy"}, status_code=409)
        removed = len(ROUTING_ENGINE.tasks)
        ROUTING_ENGINE.tasks.clear()
    return {"ok": True, "removed": removed}


@app.get("/routing/rules")
async def routing_rules():
    return {"rules": ROUTING_ENGINE.rules, "error": ROUTING_ENGINE.last_error}


@app.put("/routing/rules")
async def update_routing_rules(request: Request):
    if active or benchmark_busy():
        return JSONResponse({"error": "gateway is busy"}, status_code=409)
    try:
        ROUTING_ENGINE.update_rules(await request.json())
    except (ValueError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    response_cache.clear()
    return {"ok": True, "version": ROUTING_ENGINE.rules["version"]}


@app.get("/api/routing-rationale")
async def routing_rationale(limit: int = 100):
    """Export of recent routing decisions: level, matched rules, source, version."""
    async with lock:
        entries = list(history)
    entries = [m for m in reversed(entries) if m.route and m.route != "cache"]
    return {
        "rules_version": ROUTING_ENGINE.rules["version"],
        "dense_above": ROUTING_ENGINE.rules["dense_above"],
        "decisions": [
            {
                "ts": m.ts,
                "route": m.route,
                "level": m.routing_level,
                "matched_rules": m.routing_rules.split(",") if m.routing_rules else [],
                "reason": m.route_reason,
                "requested_model": m.requested_model,
                "model": m.model,
            }
            for m in entries[: max(1, min(limit, 500))]
        ],
    }


@app.get("/benchmarks")
async def benchmarks():
    return {
        "suites": BENCHMARK_SUITES,
        "jobs": list(benchmark_jobs.values()),
        "history": list(benchmark_history),
    }


def benchmark_result(result_id: str) -> Optional[dict]:
    return next((item for item in benchmark_history if item.get("id") == result_id), None)


def extract_html_artifact(response: str) -> str:
    fenced = re.search(r"```(?:html|svg)?\s*(.*?)```", response, re.IGNORECASE | re.DOTALL)
    html = fenced.group(1).strip() if fenced else response.strip()
    if not re.search(r"<!doctype\s+html|<html\b", html, re.IGNORECASE):
        raise ValueError("The model response does not contain an HTML document")
    return html


@app.get("/benchmarks/artifact/{result_id}", response_class=HTMLResponse)
async def benchmark_artifact(result_id: str):
    result = benchmark_result(result_id)
    if not result or not BENCHMARK_SUITES.get(result.get("suite"), {}).get("artifact_extension"):
        return HTMLResponse("Benchmark artifact not found", status_code=404)
    try:
        if BENCHMARK_SUITES[result["suite"]].get("artifact_extension") == "svg":
            import xml.etree.ElementTree as ET
            html = str(result.get("response", ""))
            try:
                element = ET.fromstring(html)
                if element.tag.split("}")[-1] != "svg":
                    raise ValueError("Expected an SVG document")
            except ET.ParseError as exc:
                raise ValueError("Invalid SVG document") from exc
        else:
            html = extract_html_artifact(str(result.get("response", "")))
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    return HTMLResponse(
        html,
        media_type="image/svg+xml" if BENCHMARK_SUITES[result["suite"]].get("artifact_extension") == "svg" else "text/html",
        headers={
            "Content-Security-Policy": "sandbox allow-scripts; default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; font-src data:",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


@app.post("/benchmarks/result/{result_id}/verdict")
async def save_benchmark_verdict(result_id: str, request: Request):
    result = benchmark_result(result_id)
    if not result:
        return JSONResponse({"error": "benchmark result not found"}, status_code=404)
    body = await request.json()
    verdict = str(body.get("verdict", "")).lower()
    comparison = str(body.get("comparison", "")).lower()
    if verdict not in {"pass", "partial", "fail"}:
        return JSONResponse({"error": "verdict must be pass, partial, or fail"}, status_code=400)
    if comparison not in {"", "better", "similar", "worse"}:
        return JSONResponse(
            {"error": "comparison must be better, similar, or worse"}, status_code=400
        )
    result["human_verdict"] = {
        "verdict": verdict,
        "comparison": comparison,
        "notes": str(body.get("notes", ""))[:2000],
        "recorded_at": time.time(),
    }
    save_benchmarks()
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, "routing-feedback.jsonl"), "a") as journal:
        journal.write(json.dumps({"result_id": result_id, "suite": result.get("suite"),
            "route": result.get("route"), "prompt_hash": result.get("prompt_hash"),
            "rules_version": ROUTING_ENGINE.rules["version"],
            "verdict": result["human_verdict"]}) + "\n")
    return {"ok": True, "human_verdict": result["human_verdict"]}


@app.post("/benchmarks/clear")
async def clear_benchmarks():
    if benchmark_busy():
        return JSONResponse(
            {"error": "benchmark history cannot be cleared while a benchmark is active"},
            status_code=409,
        )
    removed_results = len(benchmark_history)
    removed_jobs = len(benchmark_jobs)
    benchmark_history.clear()
    benchmark_jobs.clear()
    save_benchmarks()
    return {
        "ok": True, "removed_results": removed_results,
        "removed_jobs": removed_jobs,
    }


# ---------------------------------------------------------------------------
# Multi-axis benchmark lab (model x framework x harness)
# ---------------------------------------------------------------------------

lab_jobs: dict[str, dict] = {}
lab_history: list[dict] = []
_lab_manager = benchmark_lab.FrameworkManager()


def _lab_busy() -> bool:
    return any(j.get("status") in ("queued", "running") for j in lab_jobs.values())


async def _lab_run(job_id: str, framework: str, model_key: str, harness: str,
                   iterations: int, max_tokens: int, prompt: str) -> None:
    job = lab_jobs[job_id]
    job["status"] = "running"
    job["stage"] = "starting"

    def _progress(stage: str) -> None:
        # Called from the benchmark worker thread; update the job's stage so the
        # dashboard can show live progress.
        job["stage"] = stage
        job["stage_at"] = time.time()

    try:
        result = await asyncio.to_thread(
            benchmark_lab.run_benchmark,
            framework, model_key, harness,
            iterations, max_tokens, prompt,
            True, 1, _lab_manager, None, _progress,
        )
        job["result"] = result
        job["status"] = "done" if result.get("status") == "success" else "failed"
        job["stage"] = "complete"
        record = {"id": job_id, "created_at": job["created_at"], **result}
        lab_history.insert(0, record)
        del lab_history[200:]
    except Exception as e:  # noqa: BLE001
        job["status"] = "failed"
        job["stage"] = "error"
        job["result"] = {"status": "failed", "error": str(e)}


@app.get("/lab/options")
async def lab_options():
    return benchmark_lab.available_options()


@app.post("/lab/run")
async def lab_run(request: Request):
    body = await request.json()
    framework = str(body.get("framework", ""))
    model_key = str(body.get("model", ""))
    harness = str(body.get("harness", ""))
    iterations = int(body.get("iterations", 3))
    max_tokens = int(body.get("max_tokens", 128))
    prompt = str(body.get("prompt") or benchmark_lab.BENCH_PROMPT).strip()
    if framework not in benchmark_lab.FRAMEWORKS:
        return JSONResponse({"error": f"unknown framework: {framework}"}, status_code=400)
    if model_key not in benchmark_lab.MODELS:
        return JSONResponse({"error": f"unknown model: {model_key}"}, status_code=400)
    if not benchmark_lab.framework_available_for(framework, model_key):
        return JSONResponse(
            {"error": f"model '{model_key}' has no '{framework}' artifact — "
                      f"pick a framework that supports this model"},
            status_code=400)
    if harness not in benchmark_lab.HARNESS_RUNNERS:
        return JSONResponse({"error": f"unknown harness: {harness}"}, status_code=400)
    if not 1 <= iterations <= 20:
        return JSONResponse({"error": "iterations must be 1-20"}, status_code=400)
    if not 16 <= max_tokens <= 4096:
        return JSONResponse({"error": "max_tokens must be 16-4096"}, status_code=400)
    if _lab_busy():
        return JSONResponse({"error": "a benchmark lab job is already running"}, status_code=409)
    job_id = str(uuid.uuid4())
    lab_jobs[job_id] = {
        "id": job_id, "framework": framework, "model": model_key, "harness": harness,
        "iterations": iterations, "max_tokens": max_tokens, "prompt": prompt,
        "status": "queued", "stage": "queued", "created_at": time.time(),
        "result": None,
    }
    asyncio.create_task(_lab_run(job_id, framework, model_key, harness,
                                 iterations, max_tokens, prompt))
    return JSONResponse(lab_jobs[job_id], status_code=202)


@app.get("/lab/status/{job_id}")
async def lab_status(job_id: str):
    job = lab_jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return job


@app.get("/lab/results")
async def lab_results():
    return {"jobs": list(lab_jobs.values()), "history": lab_history}


@app.post("/lab/clear")
async def lab_clear():
    if _lab_busy():
        return JSONResponse({"error": "cannot clear while a lab job is running"}, status_code=409)
    lab_jobs.clear()
    lab_history.clear()
    return {"ok": True}


def rubric_check(label: str, passed: bool, detail: str, weight: int = 1) -> dict:
    return {"label": label, "passed": bool(passed), "detail": detail, "weight": weight}


def category_score(checks: list[dict]) -> int:
    total = sum(max(1, int(item.get("weight", 1))) for item in checks)
    earned = sum(
        max(1, int(item.get("weight", 1)))
        for item in checks if item.get("passed")
    )
    return round(100 * earned / total) if total else 0


def evaluate_benchmark_output(suite_id: str, response: str, finish_reason: str,
                              status: int, usage: dict) -> dict:
    """Static, non-executing evidence rubric. It never runs model-generated code."""
    text = response.strip()
    lowered = text.lower()
    categories: dict[str, dict] = {}

    completeness = [
        rubric_check("Successful model response", 200 <= status < 300, f"HTTP status {status}", 3),
        rubric_check("Substantive final answer", len(text) >= 180, f"Captured {len(text)} final-answer characters", 3),
        rubric_check("Not truncated by token limit", finish_reason != "length", f"finish_reason={finish_reason or 'not reported'}", 2),
        rubric_check("No obvious placeholder", not re.search(r"\b(todo|placeholder|implement later)\b", lowered), "Searched for TODO/placeholder language", 1),
    ]
    categories["completeness"] = {"weight": 15, "checks": completeness}

    security_patterns = [
        (r"\beval\s*\(", "Uses eval()"),
        (r"\bnew\s+function\s*\(", "Uses dynamic Function construction"),
        (r"document\.write\s*\(", "Uses document.write()"),
        (r"\.innerhtml\s*=", "Assigns untrusted-capable HTML sink"),
        (r"(?:api[_-]?key|password|secret)\s*[:=]\s*['\"][^'\"]+", "Appears to hard-code a credential"),
        (r"child_process|exec\s*\(|spawn\s*\(", "Invokes an operating-system process"),
    ]
    findings = [label for pattern, label in security_patterns if re.search(pattern, lowered)]
    security = [
        rubric_check("No high-risk static patterns", not findings,
                     "; ".join(findings) if findings else "No configured high-risk patterns found", 4),
        rubric_check("No external script dependency", not re.search(r"<script[^>]+src\s*=", lowered),
                     "Checked for externally loaded scripts", 1),
    ]
    categories["security"] = {"weight": 15, "checks": security, "findings": findings}

    if suite_id == "quick":
        bullet_count = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text))
        instruction = [
            rubric_check("Five concise bullet points", bullet_count == 5, f"Detected {bullet_count} bullet points", 4),
            rubric_check("Discusses model choice", "moe" in lowered and "dense" in lowered, "Looked for both MoE and dense", 3),
            rubric_check("Explains practical decision factors", any(x in lowered for x in ("latency", "complex", "quality", "memory", "cost")), "Looked for practical routing factors", 2),
        ]
        correctness = [
            rubric_check("States a usable routing approach", any(x in lowered for x in ("route", "routing", "choose", "select")), "Looked for an actionable selection approach", 2),
            rubric_check("Avoids absolute model claims", not re.search(r"\b(always|never)\b.*\b(moe|dense)\b", lowered), "Checked for brittle always/never rules", 1),
        ]
    elif suite_id == "coding_hitl":
        instruction = [
            rubric_check("Contains an HTML document", bool(re.search(r"<!doctype\s+html|<html\b", text, re.IGNORECASE)), "Required for local browser testing", 3),
            rubric_check("Includes tax, tip, and people inputs", all(term in lowered for term in ("tax", "tip", "people")), "Checks task coverage", 3),
            rubric_check("Uses no external dependency", not re.search(r"<(?:script|link)[^>]+(?:src|href)\s*=", lowered), "Keeps the artifact portable", 2),
        ]
        correctness = [
            rubric_check("Contains calculation logic", any(term in lowered for term in ("calculate", "total", "split", "perperson", "per-person")), "Static evidence only; human testing decides functionality", 2),
            rubric_check("Contains input validation", any(term in lowered for term in ("isnan", "isfinite", "invalid", "required", "min=")), "Static validation evidence", 2),
            rubric_check("Addresses currency rounding", any(term in lowered for term in ("math.round", "tofixed", "cents", "round")), "Static rounding evidence", 2),
        ]
    elif BENCHMARK_SUITES.get(suite_id, {}).get("artifact_extension"):
        extension = BENCHMARK_SUITES[suite_id]["artifact_extension"]
        instruction = [rubric_check("Expected artifact document", bool(re.search(
            r"<svg\b" if extension == "svg" else r"<!doctype\s+html|<html\b", text, re.I)),
            "Structural evidence only; functional behavior and likeness require human review", 3)]
        correctness = [rubric_check("Self-contained asset", not re.search(
            r'(?:src|href)=[\"\']https?://', text, re.I),
            "No external asset URLs detected; review against the task checklist", 1)]
    else:
        required_terms = {
            "States assumptions": "assumption" in lowered,
            "Defines thresholds": "threshold" in lowered or re.search(r"\d+\s*(?:gb|gib|%)", lowered) is not None,
            "Addresses swap thrashing": "swap" in lowered and any(x in lowered for x in ("thrash", "growth", "delta", "rate")),
            "Includes failure recovery": any(x in lowered for x in ("failure", "recover", "rollback", "fallback")),
            "Balances latency and memory": "latency" in lowered and "memory" in lowered,
        }
        instruction = [rubric_check(label, passed, "Required by the systems-reasoning benchmark", 2) for label, passed in required_terms.items()]
        correctness = [
            rubric_check("Provides an actionable policy", any(x in lowered for x in ("policy", "state machine", "if ", "when ")), "Looked for operational decision logic", 3),
            rubric_check("Includes hysteresis/cooldown", any(x in lowered for x in ("hysteresis", "cooldown", "idle timeout", "consecutive")), "Prevents oscillating model swaps", 2),
            rubric_check("Includes health/readiness checks", any(x in lowered for x in ("health", "ready", "readiness")), "Looks for safe transition validation", 2),
        ]

    categories["instruction_adherence"] = {"weight": 20, "checks": instruction}
    categories["static_correctness"] = {"weight": 30, "checks": correctness}

    line_count = max(1, len(text.splitlines()))
    maintainability = [
        rubric_check("Organized response", line_count >= 5, f"Captured {line_count} lines", 1),
        rubric_check("Avoids excessive repetition", len(set(text.split())) / max(1, len(text.split())) > 0.35, "Approximate vocabulary repetition check", 1),
        rubric_check("Provides a structured explanation", line_count >= 5, "Checks response structure", 1),
    ]
    categories["maintainability"] = {"weight": 10, "checks": maintainability}

    output_tokens = usage.get("completion_tokens")
    efficiency = [
        rubric_check("Stays within requested token budget", finish_reason != "length", f"completion_tokens={output_tokens}; finish_reason={finish_reason or 'not reported'}", 2),
        rubric_check("Final answer is not excessively verbose", len(text) <= 12000, f"Captured {len(text)} characters", 1),
    ]
    categories["output_efficiency"] = {"weight": 10, "checks": efficiency}

    for category in categories.values():
        category["score"] = category_score(category["checks"])
    readiness = round(sum(c["score"] * c["weight"] for c in categories.values()) /
                      max(1, sum(c["weight"] for c in categories.values())))
    return {
        "readiness_score": readiness,
        "method": "static non-executing rubric",
        "confidence": "medium",
        "categories": categories,
        "limitations": [
            "Security findings are pattern-based and are not a full audit.",
            "Scores measure this response, not the model globally.",
        ],
    }


async def benchmark_infer(route: str, body: dict) -> tuple[int, float, Optional[float], dict, str, str, str, int]:
    """Consume an MLX SSE response and return accurate first-token/final timing."""
    started = now_ms()
    ttft_ms: Optional[float] = None
    usage: dict[str, Any] = {}
    finish_reason = ""
    response_parts: list[str] = []
    reasoning_parts: list[str] = []
    stream_events = 0
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    async with httpx.AsyncClient(timeout=None, headers=ROUTE_BACKENDS[route].headers()) as client:
        async with client.stream(
            "POST", ROUTE_BACKENDS[route].chat_url(), json=body
        ) as response:
            if response.status_code >= 400:
                detail = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"upstream HTTP {response.status_code}: {detail[:500]}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                stream_events += 1
                content_text, reasoning_text = stream_delta_text(event)
                if content_text:
                    response_parts.append(content_text)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                if ttft_ms is None and (content_text or reasoning_text):
                    ttft_ms = now_ms() - started
                for key, value in extract_usage(event).items():
                    if value is not None:
                        usage[key] = value
                reason = extract_finish_reason(event)
                if reason:
                    finish_reason = reason
            return (
                response.status_code, now_ms() - started, ttft_ms, usage,
                finish_reason, "".join(response_parts), "".join(reasoning_parts),
                stream_events,
            )


def pi_message_text(message: dict) -> tuple[str, str]:
    """Return final-answer and reasoning text from a Pi JSON message."""
    content = message.get("content") or []
    if isinstance(content, str):
        return content, ""
    answer: list[str] = []
    reasoning: list[str] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        text = str(part.get("text") or part.get("content") or "")
        if part.get("type") in ("thinking", "reasoning"):
            reasoning.append(text)
        elif part.get("type") == "text":
            answer.append(text)
    return "".join(answer), "".join(reasoning)


def find_benchmark_artifact(workspace: str, suite_id: str) -> Optional[str]:
    extension = BENCHMARK_SUITES.get(suite_id, {}).get("artifact_extension")
    if not extension:
        return None
    preferred = os.path.join(workspace, f"artifact.{extension}")
    if os.path.isfile(preferred):
        return preferred
    candidates: list[str] = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in ("sessions", ".git", "node_modules")]
        candidates.extend(
            os.path.join(root, name) for name in files if name.lower().endswith("." + extension)
        )
    return max(candidates, key=os.path.getsize) if candidates else None


def benchmark_live(job: dict, route: str, kind: str, message: str):
    log = job.setdefault("live_log", [])
    log.append({
        "ts": time.time(), "route": route, "kind": kind,
        "message": re.sub(r"(?i)(api[_-]?key|authorization|token)\s*[:=]\s*\S+", r"\1=[redacted]", message)[:1200],
    })
    del log[:-250]


def describe_pi_event(event: dict) -> Optional[tuple[str, str]]:
    event_type = str(event.get("type") or "")
    if event_type == "session":
        return "session", f"Pi session {event.get('id', '')} started in {event.get('cwd', '')}"
    if event_type == "agent_start":
        return "agent", "Pi agent started"
    if event_type == "turn_start":
        return "turn", "Agent turn started"
    if event_type == "tool_execution_start":
        args = json.dumps(event.get("args") or {}, ensure_ascii=False)
        return "tool", f"Running {event.get('toolName', 'tool')}: {args[:700]}"
    if event_type == "tool_execution_end":
        state = "failed" if event.get("isError") else "completed"
        return "error" if event.get("isError") else "tool", f"{event.get('toolName', 'tool')} {state}"
    if event_type == "message_end":
        message = event.get("message") or {}
        if message.get("role") == "assistant":
            answer, _ = pi_message_text(message)
            return "answer", f"Assistant message completed: {answer[:500] or '[tool call / reasoning]'}"
    if event_type in ("agent_end", "agent_settled"):
        return "agent", "Pi agent finished"
    if event_type in ("compaction_start", "compaction_end"):
        return "context", event_type.replace("_", " ")
    return None


def benchmark_generation_stats(metrics: list[dict]) -> dict:
    """Time-weighted generation rate across every call, never a mean of rates."""
    if not metrics or any(
        m.get("completion_tokens") is None or m.get("generation_ms") is None
        or m["generation_ms"] <= 0 for m in metrics
    ):
        return {"average_tps": None, "generation_ms": None}
    tokens = sum(m["completion_tokens"] for m in metrics)
    duration = sum(m["generation_ms"] for m in metrics)
    return {"average_tps": round(tokens * 1000 / duration, 2),
            "generation_ms": round(duration, 1)}


async def run_pi_benchmark(route: str, suite_id: str, prompt: str, workspace: str,
                           reasoning_effort: str, job: dict) -> dict:
    """Run a complete isolated Pi agent session and capture its JSON event stream."""
    os.makedirs(workspace, exist_ok=True)
    transcript_path = os.path.join(workspace, "pi-events.jsonl")
    alias = f"benchmark-{route}"
    system_note = (
        "You are running an isolated, reproducible benchmark. Work only inside the current "
        "directory. Complete the user's task fully rather than merely describing a plan. "
        "Use your normal coding-agent tools when useful."
    )
    extension = BENCHMARK_SUITES.get(suite_id, {}).get("artifact_extension")
    if extension:
        system_note += (
            f" Create the complete, self-contained solution as artifact.{extension}. "
            "Validate the artifact locally before finishing."
        )
    command = [
        PI_EXECUTABLE, "--mode", "json", "--print", "--no-session", "--approve",
        "--provider", "mlx-proxy", "--model", alias,
        "--thinking", reasoning_effort if reasoning_effort in
        ("off", "minimal", "low", "medium", "high", "xhigh", "max") else "medium",
        "--append-system-prompt", system_note, prompt,
    ]
    started_perf = now_ms()
    started_wall = time.time()
    process = await asyncio.create_subprocess_exec(
        *command, cwd=workspace,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        limit=PI_EVENT_STREAM_LIMIT_BYTES,
    )
    benchmark_live(job, route, "request", f"Exact prompt sent to Pi: {prompt}")
    benchmark_live(job, route, "process", f"Pi launched with model={alias}, thinking={reasoning_effort}")
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    async def consume_stdout():
        assert process.stdout is not None
        pending = b""

        def consume_line(raw: bytes):
            line = raw.decode("utf-8", errors="replace")
            stdout_lines.append(line + "\n")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return
            described = describe_pi_event(event)
            if described:
                benchmark_live(job, route, described[0], described[1])

        while True:
            chunk = await process.stdout.read(64 * 1024)
            if not chunk:
                break
            pending += chunk
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                if raw.strip():
                    consume_line(raw)
        if pending.strip():
            consume_line(pending)

    async def consume_stderr():
        assert process.stderr is not None
        while True:
            raw = await process.stderr.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace")
            stderr_lines.append(line)
            if line.strip():
                benchmark_live(job, route, "stderr", line.strip())

    readers = [asyncio.create_task(consume_stdout()), asyncio.create_task(consume_stderr())]
    try:
        await asyncio.wait_for(
            asyncio.gather(process.wait(), *readers), timeout=PI_BENCHMARK_TIMEOUT_SEC
        )
    except asyncio.TimeoutError:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        for reader in readers:
            reader.cancel()
        benchmark_live(job, route, "error", f"Pi timed out after {PI_BENCHMARK_TIMEOUT_SEC}s")
        raise RuntimeError(f"Pi benchmark timed out after {PI_BENCHMARK_TIMEOUT_SEC}s")
    except Exception as exc:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        for reader in readers:
            if not reader.done():
                reader.cancel()
        benchmark_live(job, route, "error", f"Pi event capture failed: {type(exc).__name__}: {exc}")
        raise
    latency_ms = now_ms() - started_perf
    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)
    with open(transcript_path, "w", encoding="utf-8") as output:
        output.write(stdout_text)

    final_answer = ""
    reasoning = ""
    stop_reason = ""
    pi_input_tokens = 0
    pi_output_tokens = 0
    pi_cache_read = 0
    tool_calls = 0
    tool_errors = 0
    events = 0
    for line in stdout_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events += 1
        if event.get("type") == "tool_execution_start":
            tool_calls += 1
        elif event.get("type") == "tool_execution_end" and event.get("isError"):
            tool_errors += 1
        elif event.get("type") == "message_end":
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            answer, thought = pi_message_text(message)
            if answer:
                final_answer = answer
            if thought:
                reasoning += thought
            usage = message.get("usage") or {}
            pi_input_tokens += int(usage.get("input") or 0)
            pi_output_tokens += int(usage.get("output") or 0)
            pi_cache_read += int(usage.get("cacheRead") or 0)
            stop_reason = str(message.get("stopReason") or stop_reason)

    matching_metrics: list[dict[str, Any]] = []
    for item in history:
        if isinstance(item, Metric):
            metric = asdict(item)
        elif isinstance(item, dict):
            metric = item
        else:
            continue
        if (
            float(metric.get("ts") or 0) >= started_wall
            and metric.get("requested_model") == alias
        ):
            matching_metrics.append(metric)
    model_prompt_tokens = sum(int(x.get("prompt_tokens") or 0) for x in matching_metrics)
    model_completion_tokens = sum(int(x.get("completion_tokens") or 0) for x in matching_metrics)
    model_latency_ms = sum(float(x.get("latency_ms") or 0) for x in matching_metrics)
    first_ttft = next((x.get("ttft_ms") for x in matching_metrics if x.get("ttft_ms") is not None), None)
    artifact_path = find_benchmark_artifact(workspace, suite_id)
    artifact_html = ""
    if artifact_path:
        with open(artifact_path, "r", encoding="utf-8", errors="replace") as artifact:
            artifact_html = artifact.read()
    if process.returncode != 0:
        detail = stderr_text.strip()[-1200:] or final_answer or "Pi exited without an error message"
        raise RuntimeError(f"Pi exited with status {process.returncode}: {detail}")
    benchmark_live(job, route, "complete", f"Pi completed in {latency_ms / 1000:.1f}s with {tool_calls} tool calls")
    return {
        "status": 200,
        "latency_ms": latency_ms,
        "ttft_ms": first_ttft,
        "prompt_tokens": model_prompt_tokens or pi_input_tokens,
        "completion_tokens": model_completion_tokens or pi_output_tokens,
        "total_tokens": (model_prompt_tokens + model_completion_tokens) or
                        (pi_input_tokens + pi_output_tokens),
        "cached_tokens": pi_cache_read,
        **benchmark_generation_stats(matching_metrics),
        "model_latency_ms": model_latency_ms,
        "model_requests": len(matching_metrics),
        "response": artifact_html or final_answer,
        "pi_response": final_answer,
        "reasoning": reasoning,
        "finish_reason": stop_reason or "agent_settled",
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "event_count": events,
        "workspace": workspace,
        "artifact_path": artifact_path or "",
        "transcript_path": transcript_path,
        "stderr": stderr_text[-4000:],
    }


async def run_benchmark(job_id: str, suite_id: str, routes: list[str], max_tokens: int,
                        benchmark_prompt: str):
    job = benchmark_jobs[job_id]
    original_route = "dense" if await pid_running("dense") else "moe"
    suite = BENCHMARK_SUITES[suite_id]
    try:
        async with benchmark_lock:
            job.update({"status": "running", "started_at": time.time()})
            for route in routes:
                    workspace = os.path.join(BENCHMARK_WORK_DIR, job_id, route)
                    job.update({"current_route": route, "stage": f"loading {route}"})
                    logger.info(
                        "Benchmark %s loading route=%s model=%s suite=%s max_tokens=%s prompt=%r",
                        job_id, route, MODEL_IDS[route], suite_id, max_tokens, benchmark_prompt,
                    )
                    before = psutil.virtual_memory()
                    swap_before = swap_out_bytes()
                    switch_started = now_ms()
                    await ensure_route(route)
                    swap_ms = now_ms() - switch_started
                    ready_at = time.time()
                    job.update({"stage": f"Pi agent running on {route}", "model_ready_at": ready_at})
                    logger.info(
                        "Benchmark %s model endpoint ready route=%s readiness_ms=%.1f; starting inference",
                        job_id, route, swap_ms,
                    )
                    inference_started_at = time.time()
                    pi_run = await run_pi_benchmark(
                        route, suite_id, benchmark_prompt, workspace,
                        suite["reasoning_effort"] if suite["thinking"] else "off",
                        job,
                    )
                    inference_finished_at = time.time()
                    after = psutil.virtual_memory()
                    swap_after = swap_out_bytes()
                    completion = pi_run["completion_tokens"]
                    latency_ms = pi_run["latency_ms"]
                    ttft_ms = pi_run["ttft_ms"]
                    response_text = pi_run["response"]
                    reasoning_text = pi_run["reasoning"]
                    finish_reason = pi_run["finish_reason"]
                    status = pi_run["status"]
                    result = {
                        "id": str(uuid.uuid4()), "job_id": job_id,
                        "ts": time.time(), "suite": suite_id, "route": route,
                        "requested_routes": list(routes),
                        "model": MODEL_IDS[route], "status": status,
                        "prompt": benchmark_prompt,
                        "prompt_hash": hashlib.sha256(benchmark_prompt.encode("utf-8")).hexdigest(),
                        "prompt_mode": "standard" if benchmark_prompt == suite["prompt"] else "custom",
                        "parameters": {
                            "executor": "pi", "mode": "json", "isolated_workspace": True,
                            "max_tokens": max_tokens,
                            "enable_thinking": suite["thinking"],
                            "reasoning_effort": suite["reasoning_effort"],
                        },
                        "pi_response": pi_run["pi_response"],
                        "response": response_text,
                        "reasoning": reasoning_text,
                        "response_chars": len(response_text),
                        "reasoning_chars": len(reasoning_text),
                        "answer_state": "final_answer" if response_text else
                        ("reasoning_only" if reasoning_text else "empty"),
                        "stream_events": pi_run["event_count"],
                        "pi_tool_calls": pi_run["tool_calls"],
                        "pi_tool_errors": pi_run["tool_errors"],
                        "pi_model_requests": pi_run["model_requests"],
                        "model_latency_ms": round(pi_run["model_latency_ms"], 1),
                        "workspace": pi_run["workspace"],
                        "artifact_path": pi_run["artifact_path"],
                        "transcript_path": pi_run["transcript_path"],
                        "model_ready_at": ready_at,
                        "inference_started_at": inference_started_at,
                        "inference_finished_at": inference_finished_at,
                        "total_time_ms": round(swap_ms + latency_ms, 1),
                        "average_tps": pi_run["average_tps"],
                        "generation_ms": pi_run["generation_ms"],
                        "end_to_end_tps": round(completion * 1000 / latency_ms, 2) if latency_ms > 0 else None,
                        "swap_ms": round(swap_ms, 1), "latency_ms": round(latency_ms, 1),
                        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                        "prompt_tokens": pi_run["prompt_tokens"],
                        "completion_tokens": completion, "total_tokens": pi_run["total_tokens"],
                        "gen_tps": pi_run["average_tps"],
                        "finish_reason": finish_reason,
                        "available_before_gb": round(before.available / 1024 ** 3, 2),
                        "available_after_gb": round(after.available / 1024 ** 3, 2),
                        "swap_delta_mb": round(max(0, swap_after - swap_before) / 1024 ** 2, 2)
                        if swap_before is not None and swap_after is not None else None,
                        "compute_score": compute_score(latency_ms, pi_run["prompt_tokens"], completion, swap_ms),
                    }
                    result["evaluation"] = evaluate_benchmark_output(
                        suite_id, response_text, finish_reason, status, {
                            "prompt_tokens": pi_run["prompt_tokens"],
                            "completion_tokens": completion,
                            "total_tokens": pi_run["total_tokens"],
                        }
                    )
                    benchmark_history.append(result)
                    job["results"].append(result)
                    save_benchmarks()
                    logger.info(
                        "Benchmark %s completed route=%s status=%s ttft_ms=%s latency_ms=%.1f completion_tokens=%s",
                        job_id, route, status, ttft_ms, latency_ms, completion,
                    )
            job.update({"stage": "restoring original backend", "current_route": ""})
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc) or 'no error details supplied'}"
        logger.exception("Benchmark job %s failed", job_id)
        job.update({"status": "failed", "stage": "failed", "finished_at": time.time(), "error": error, "current_route": ""})
    finally:
        try:
            async with benchmark_lock:
                async with backend_slot:
                    if original_route in ("moe", "dense") and not active:
                        await ensure_route(original_route)
                        if job.get("status") != "failed":
                            job.update({
                                "status": "complete", "stage": f"complete; restored {original_route}",
                                "finished_at": time.time(),
                            })
                            logger.info("Benchmark %s restored original route=%s", job_id, original_route)
        except Exception as exc:
            job["restore_error"] = f"{type(exc).__name__}: {str(exc) or 'no error details supplied'}"


@app.post("/benchmarks/run")
async def start_benchmark(request: Request):
    body = await request.json()
    suite_id = str(body.get("suite", "quick"))
    routes = body.get("routes", ["moe", "dense"])
    if suite_id not in BENCHMARK_SUITES:
        return JSONResponse({"error": "unknown benchmark suite"}, status_code=400)
    if not isinstance(routes, list) or not routes or any(r not in ("moe", "dense") for r in routes):
        return JSONResponse({"error": "routes must contain moe and/or dense"}, status_code=400)
    if active or benchmark_busy():
        return JSONResponse({"error": "gateway is busy"}, status_code=409)
    # Pi's two benchmark aliases use the same real-world 132K context / 32,144
    # output profile. The model may stop earlier naturally.
    max_tokens = 32144
    benchmark_prompt = str(body.get("prompt") or BENCHMARK_SUITES[suite_id]["prompt"]).strip()
    if not benchmark_prompt or len(benchmark_prompt) > 20000:
        return JSONResponse({"error": "prompt must contain 1 to 20,000 characters"}, status_code=400)
    job_id = str(uuid.uuid4())
    benchmark_jobs[job_id] = {
        "id": job_id, "suite": suite_id, "routes": routes, "max_tokens": max_tokens,
        "status": "queued", "stage": "queued", "created_at": time.time(),
        "current_route": "", "prompt": benchmark_prompt,
        "prompt_hash": hashlib.sha256(benchmark_prompt.encode("utf-8")).hexdigest(),
        "prompt_mode": "standard" if benchmark_prompt == BENCHMARK_SUITES[suite_id]["prompt"] else "custom",
        "parameters": {
            "max_tokens": max_tokens,
            "executor": "pi", "mode": "json", "isolated_workspace": True,
            "enable_thinking": BENCHMARK_SUITES[suite_id]["thinking"],
            "reasoning_effort": BENCHMARK_SUITES[suite_id]["reasoning_effort"],
        },
        "executor": "pi-agent",
        "live_log": [{
            "ts": time.time(), "route": "all", "kind": "request",
            "message": f"Benchmark queued. Exact shared prompt: {benchmark_prompt}",
        }],
        "results": [],
    }
    asyncio.create_task(run_benchmark(job_id, suite_id, routes, max_tokens, benchmark_prompt))
    return JSONResponse(benchmark_jobs[job_id], status_code=202)


@app.get("/v1/models")
async def models():
    state = await model_state()
    return {
        "object": "list",
        "data": [
            {
                "id": "gateway-auto",
                "object": "model",
                "created": int(time.time()),
                "route": "auto",
                "running": True,
                "context_window": MODEL_CONTEXT_WINDOW,
            },
            {
                "id": "gateway-moe", "object": "model", "created": int(time.time()),
                "route": "moe", "running": state["moe"]["running"],
                "context_window": MODEL_CONTEXT_WINDOW,
            },
            {
                "id": "gateway-dense", "object": "model", "created": int(time.time()),
                "route": "dense", "running": state["dense"]["running"],
                "context_window": MODEL_CONTEXT_WINDOW,
            },
            *[
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "route": route,
                "running": state[route]["running"],
                "context_window": MODEL_CONTEXT_WINDOW,
            }
            for route, model in MODEL_IDS.items()
            ],
        ],
    }


# ---------------------------------------------------------------------------
# Streaming chat proxy
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat(request: Request):
    global last_activity_at
    body = await request.json()
    requested_model = str(body.get("model", ""))
    if benchmark_busy() and requested_model not in BENCHMARK_MODEL_ALIASES:
        return JSONResponse(
            {"error": "Gateway benchmark in progress; retry when it completes."},
            status_code=409,
            headers={"Retry-After": "5"},
        )

    req_id = str(uuid.uuid4())
    started = now_ms()
    last_activity_at = time.time()

    streaming = bool(body.get("stream"))
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))

    async with lock:
        active[req_id] = {
            "id": req_id,
            "model": requested_model,
            "requested_model": requested_model,
            "route": "routing",
            "started": time.time(),
            "max_tokens": max_tokens,
            "streaming": streaming,
            "ttft_ms": None,
            "chunks": 0,
            "memory_available_gb": memory_state["available_gb"],
            "memory_pressure": memory_state["pressure"],
            "low_memory_events": 0,
        }

    try:
        cache_id = cache_key(body)
        if not streaming:
            cached = cache_get(cache_id)
            if cached:
                usage = extract_usage(cached["payload"])
                await record(Metric(
                    id=req_id, ts=time.time(), model=requested_model,
                    requested_model=requested_model, route="cache",
                    route_reason="exact request cache", status=cached["status"],
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"], cache_hit=True,
                    max_tokens=max_tokens, streaming=False, latency_ms=now_ms() - started,
                ))
                return JSONResponse(
                    cached["payload"], status_code=cached["status"],
                    headers={"X-Gateway-Cache": "HIT", "X-Gateway-Request-ID": req_id},
                )

        policy, router_ms = await choose_policy(body, req_id)
        route = policy["route"]
        confidence = policy["confidence"]
        route_reason = policy["reason"]
        automatic = MODEL_ROUTES.get(requested_model) is None
        async with lock:
            if req_id in active:
                active[req_id].update({
                    "route": route,
                    "route_reason": route_reason, "router_confidence": confidence,
                    "routing_level": policy.get("complexity"),
                    "router_ms": router_ms,
                    "task_type": policy["task_type"], "effort": policy["effort"],
                    "thinking": policy["thinking"], "effective_max_tokens": policy["max_tokens"],
                })

        # Protect admission to the model.
        async with request_slots:
            wait_ms = await wait_for_memory(req_id, route)

            if not streaming:
                return await proxy_nonstream(
                    body=body,
                    requested_model=requested_model,
                    route=route,
                    route_reason=route_reason,
                    router_confidence=confidence,
                    router_ms=router_ms,
                    cache_id=cache_id,
                    req_id=req_id,
                    request_started=started,
                    client_wait_ms=wait_ms,
                    policy=policy,
                    automatic=automatic,
                )

            return StreamingResponse(
                stream_request(
                    body=body,
                    requested_model=requested_model,
                    route=route,
                    route_reason=route_reason,
                    router_confidence=confidence,
                    router_ms=router_ms,
                    req_id=req_id,
                    request_started=started,
                    client_wait_ms=wait_ms,
                    policy=policy,
                    automatic=automatic,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Proxy-Request-ID": req_id,
                },
            )

    except Exception:
        async with lock:
            active.pop(req_id, None)
        raise


async def proxy_nonstream(
    body: dict,
    requested_model: str,
    route: str,
    route_reason: str,
    router_confidence: float,
    router_ms: float,
    cache_id: str,
    req_id: str,
    request_started: float,
    client_wait_ms: float,
    policy: dict,
    automatic: bool,
):
    t0 = now_ms()
    swap_ms = 0.0

    try:
        async with backend_slot:
            route, route_reason, swap_ms = await prepare_backend(route, route_reason)
            policy = dict(policy, route=route)
            routed_body = apply_request_policy(body, policy, automatic)
            async with lock:
                if req_id in active:
                    active[req_id].update({"model": MODEL_IDS[route], "route": route, "route_reason": route_reason, "swap_ms": swap_ms})
            async with httpx.AsyncClient(timeout=None, headers=ROUTE_BACKENDS[route].headers()) as client:
                r = await client.post(
                    ROUTE_BACKENDS[route].chat_url(),
                    json=routed_body,
                )

        elapsed = now_ms() - t0
        payload = r.json()
        cache_put(cache_id, payload, r.status_code)
        usage = extract_usage(payload)

        completion = usage["completion_tokens"]
        gen_tps = (
            completion / (elapsed / 1000)
            if completion and elapsed > 0 else None
        )

        metric = Metric(
            id=req_id,
            ts=time.time(),
            model=MODEL_IDS[route],
            requested_model=requested_model,
            route=route,
            route_reason=route_reason,
            router_confidence=router_confidence,
            router_ms=router_ms,
            swap_ms=swap_ms,
            status=r.status_code,
            latency_ms=elapsed,
            generation_ms=elapsed,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=completion,
            total_tokens=usage["total_tokens"],
            cached_tokens=usage["cached_tokens"],
            gen_tps=gen_tps,
            finish_reason=extract_finish_reason(payload),
            max_tokens=body.get("max_tokens", body.get("max_completion_tokens")),
            streaming=False,
            client_wait_ms=client_wait_ms,
            peak_available_gb=memory_state["available_gb"],
            compute_score=compute_score(elapsed, usage["prompt_tokens"], completion, swap_ms),
            task_type=policy["task_type"], effort=policy["effort"],
            thinking=policy["thinking"], effective_max_tokens=policy["max_tokens"],
            routing_level=policy.get("complexity"),
            routing_rules=",".join(policy.get("matched_rules", []) + policy.get("modifiers", [])),
        )

        await record(metric)
        return JSONResponse(payload, status_code=r.status_code)

    except Exception as exc:
        await record(
            Metric(
                id=req_id,
                ts=time.time(),
                model=MODEL_IDS.get(route, body.get("model", "")),
                requested_model=requested_model,
                route=route,
                route_reason=route_reason,
                router_confidence=router_confidence,
                router_ms=router_ms,
                swap_ms=swap_ms,
                status=502,
                error=str(exc),
                max_tokens=body.get("max_tokens", body.get("max_completion_tokens")),
                streaming=False,
                client_wait_ms=client_wait_ms,
                task_type=policy.get("task_type", ""), effort=policy.get("effort", ""),
                thinking=policy.get("thinking"), effective_max_tokens=policy.get("max_tokens"),
                routing_level=policy.get("complexity"),
                routing_rules=",".join(policy.get("matched_rules", []) + policy.get("modifiers", [])),
            )
        )
        return JSONResponse({"error": str(exc)}, status_code=502)


async def stream_request(
    body: dict,
    requested_model: str,
    route: str,
    route_reason: str,
    router_confidence: float,
    router_ms: float,
    req_id: str,
    request_started: float,
    client_wait_ms: float,
    policy: dict,
    automatic: bool,
):
    request_t0 = now_ms()
    ttft_ms = None
    finish_reason = ""
    usage: dict = {}
    upstream_status = 200
    stream_events = 0
    low_memory_events = 0
    swap_ms = 0.0

    try:
        async with backend_slot:
            route, route_reason, swap_ms = await prepare_backend(route, route_reason)
            policy = dict(policy, route=route)
            body = apply_request_policy(body, policy, automatic)
            async with lock:
                if req_id in active:
                    active[req_id].update({"model": MODEL_IDS[route], "route": route, "route_reason": route_reason, "swap_ms": swap_ms})
            generation_dispatch = now_ms()
            async for chunk in stream_upstream(body, req_id, route):
                kind, value = chunk
                if kind == "meta":
                    upstream_status, ttft_ms, finish_reason, usage, stream_events, low_memory_events = value
                    continue
                yield value

        elapsed = now_ms() - request_t0

        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")
        cached_tokens = usage.get("cached_tokens")

        dispatch_ms = now_ms() - generation_dispatch
        generation_ms = max(0.0, dispatch_ms - ttft_ms) if ttft_ms is not None else None
        gen_tps = completion_tokens / (generation_ms / 1000) if completion_tokens and generation_ms is not None and generation_ms > 0 else None
        update_memory_state()
        await record(Metric(
            id=req_id, ts=time.time(), model=MODEL_IDS[route], requested_model=requested_model,
            route=route, route_reason=route_reason, router_confidence=router_confidence,
            router_ms=router_ms, swap_ms=swap_ms, status=upstream_status, ttft_ms=ttft_ms,
            latency_ms=elapsed, generation_ms=generation_ms, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, total_tokens=total_tokens, cached_tokens=cached_tokens,
            gen_tps=gen_tps, finish_reason=finish_reason,
            max_tokens=body.get("max_tokens", body.get("max_completion_tokens")), streaming=True,
            client_wait_ms=client_wait_ms, peak_available_gb=memory_state["available_gb"],
            low_memory_events=low_memory_events,
            compute_score=compute_score(elapsed, prompt_tokens, completion_tokens, swap_ms),
            task_type=policy["task_type"], effort=policy["effort"],
            thinking=policy["thinking"], effective_max_tokens=policy["max_tokens"],
            routing_level=policy.get("complexity"),
            routing_rules=",".join(policy.get("matched_rules", []) + policy.get("modifiers", [])),
        ))

    except asyncio.CancelledError:
        async with lock:
            info = active.get(req_id, {})
        await record(Metric(id=req_id, ts=time.time(), model=MODEL_IDS.get(route, ""), requested_model=requested_model,
            route=route, route_reason=route_reason, router_confidence=router_confidence, router_ms=router_ms,
            swap_ms=swap_ms, status=499, ttft_ms=info.get("ttft_ms"), latency_ms=now_ms()-request_t0,
            error="client disconnected/cancelled", max_tokens=body.get("max_tokens", body.get("max_completion_tokens")),
            streaming=True, client_wait_ms=client_wait_ms, low_memory_events=info.get("low_memory_events", 0),
            task_type=policy.get("task_type", ""), effort=policy.get("effort", ""),
            thinking=policy.get("thinking"), effective_max_tokens=policy.get("max_tokens"),
            routing_level=policy.get("complexity"),
            routing_rules=",".join(policy.get("matched_rules", []) + policy.get("modifiers", []))))
        raise
    except Exception as exc:
        await record(Metric(id=req_id, ts=time.time(), model=MODEL_IDS.get(route, ""), requested_model=requested_model,
            route=route, route_reason=route_reason, router_confidence=router_confidence, router_ms=router_ms,
            swap_ms=swap_ms, status=502, ttft_ms=ttft_ms, latency_ms=now_ms()-request_t0,
            error=str(exc), max_tokens=body.get("max_tokens", body.get("max_completion_tokens")), streaming=True,
            client_wait_ms=client_wait_ms, low_memory_events=low_memory_events,
            task_type=policy.get("task_type", ""), effort=policy.get("effort", ""),
            thinking=policy.get("thinking"), effective_max_tokens=policy.get("max_tokens"),
            routing_level=policy.get("complexity"),
            routing_rules=",".join(policy.get("matched_rules", []) + policy.get("modifiers", []))))
        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()


async def stream_upstream(body: dict, req_id: str, route: str):
    request_t0 = now_ms()
    ttft_ms = None
    finish_reason = ""
    usage: dict = {}
    upstream_status = 200
    stream_events = 0
    low_memory_events = 0
    timeout = httpx.Timeout(None)

    async with httpx.AsyncClient(timeout=timeout, headers=ROUTE_BACKENDS[route].headers()) as client:
        async with client.stream(
            "POST",
            ROUTE_BACKENDS[route].chat_url(),
            json=body,
        ) as response:

            upstream_status = response.status_code

            if response.status_code >= 400:
                raw = await response.aread()
                text = raw.decode("utf-8", errors="replace")
                yield "data", f"data: {json.dumps({'error': text})}\n\n".encode()
                raise RuntimeError(
                    f"upstream HTTP {response.status_code}: {text[:500]}"
                )

            async for line in response.aiter_lines():
                if not line:
                    yield "data", b"\n"
                    continue

                if not line.startswith("data:"):
                    yield "data", (line + "\n").encode()
                    continue

                payload = line[5:].strip()

                if payload == "[DONE]":
                    yield "data", b"data: [DONE]\n\n"
                    continue

                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    yield "data", (line + "\n\n").encode()
                    continue

                content_text, reasoning_text = stream_delta_text(obj)
                now = now_ms()
                stream_events += 1

                # Role-only deltas, keep-alives and usage metadata are not
                # generated tokens. Start TTFT on real answer/reasoning text.
                if ttft_ms is None and (content_text or reasoning_text):
                    ttft_ms = now - request_t0

                u = extract_usage(obj)
                for k, v in u.items():
                    if v is not None:
                        usage[k] = v

                reason = extract_finish_reason(obj)
                if reason:
                    finish_reason = reason

                update_memory_state()

                if memory_state["pressure"]:
                    low_memory_events += 1

                async with lock:
                    if req_id in active:
                        update = {
                            "chunks": stream_events,
                            "last_event": time.time(),
                            "memory_available_gb": memory_state["available_gb"],
                            "memory_pressure": memory_state["pressure"],
                            "low_memory_events": low_memory_events,
                        }
                        if ttft_ms is not None:
                            update["ttft_ms"] = round(ttft_ms, 1)
                        active[req_id].update(update)

                # Do NOT buffer or modify the SSE response.
                yield "data", (line + "\n\n").encode()

    yield "meta", (upstream_status, ttft_ms, finish_reason, usage, stream_events, low_memory_events)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

DASHBOARD = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLX Inference Gateway</title>
<style>
:root{color-scheme:dark}
body{font:14px -apple-system,BlinkMacSystemFont,"SF Pro Text",sans-serif;background:#0d0f12;color:#eee;margin:24px}
h1{margin:0 0 4px}.sub{color:#8b949e;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:10px;margin:18px 0}
.card{background:#16191d;border:1px solid #30363d;border-radius:10px;padding:14px}
.k{color:#8b949e;font-size:12px}.v{font-size:24px;font-weight:700;margin-top:6px}
.warn{border-color:#d29922}.danger{border-color:#f85149}.good{color:#7ee787}.bad{color:#ff7b72}.blue{color:#79c0ff}
table{width:100%;border-collapse:collapse;background:#111418;border:1px solid #30363d}
th,td{padding:8px;border-bottom:1px solid #242a31;text-align:left;white-space:nowrap}
th{color:#8b949e;font-weight:500}
section{margin-top:26px;overflow:auto}
.bar{height:8px;background:#242a31;border-radius:99px;overflow:hidden;margin-top:9px}
.fill{height:100%;width:0%;background:#7ee787;transition:width .3s}
@media(max-width:1000px){.grid{grid-template-columns:repeat(3,1fr)}}
</style>
</head>
<body>
<h1>MLX Inference Gateway</h1>
<div class=sub id=status>Connecting...</div>

<div class=grid>
<div class=card><div class=k>Memory available</div><div class=v id=mem>—</div><div class=bar><div class=fill id=membar></div></div></div>
<div class=card><div class=k>Memory pressure</div><div class=v id=pressure>—</div></div>
<div class=card><div class=k>Active</div><div class=v id=active>0</div></div>
<div class=card><div class=k>Requests</div><div class=v id=req>0</div></div>
<div class=card><div class=k>Avg TTFT</div><div class=v id=ttft>—</div></div>
<div class=card><div class=k>Avg generation</div><div class=v id=tps>—</div></div>
<div class=card><div class=k>Total tokens</div><div class=v id=tokens>0</div></div>
<div class=card><div class=k>Exact cache</div><div class=v id=cache>—</div></div>
<div class=card><div class=k>Active backend</div><div class=v id=backend>—</div></div>
</div>

<section>
<h2>Model state</h2>
<table><thead><tr><th>Role</th><th>Model</th><th>State</th><th>Endpoint</th></tr></thead>
<tbody id=modelRows></tbody></table>
</section>

<section>
<h2>Active requests</h2>
<table>
<thead><tr>
<th>Route</th><th>Model</th><th>Level</th><th>TTFT</th><th>Max tokens</th><th>Memory</th><th>Chunks</th><th>Age</th>
</tr></thead>
<tbody id=activeRows></tbody>
</table>
</section>

<section>
<h2>Request history</h2>
<table>
<thead><tr>
<th>Time</th><th>Route</th><th>Model</th><th>Level</th><th>Router</th><th>Swap</th><th>Cache</th>
<th>TTFT</th><th>Prompt</th><th>Completion</th><th>Total</th><th>Tok/s</th><th>Finish</th><th>Memory</th><th>Status</th>
</tr></thead>
<tbody id=historyRows></tbody>
</table>
</section>

<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const val=(x,d='—')=>x==null?d:x;
const sec=x=>x==null?'—':(x/1000).toFixed(2)+'s';

async function tick(){
  try{
    const d=await (await fetch('/metrics',{cache:'no-store'})).json();
    const h=d.history||[], a=d.active||[], m=d.memory||{};

    document.getElementById('status').textContent =
      `Phase 1 · local-first routing · ${new Date().toLocaleTimeString()}`;

    document.getElementById('mem').textContent =
      `${m.available_gb?.toFixed(2)} GiB`;

    const total=m.total_gb||1;
    const availablePct=Math.max(0,Math.min(100,(m.available_gb/total)*100));
    document.getElementById('membar').style.width=availablePct+'%';

    const p=document.getElementById('pressure');
    p.textContent=m.hard_pressure?'HARD STOP':m.pressure?'GUARD':'NORMAL';
    p.className='v '+(m.hard_pressure?'bad':m.pressure?'warn':'good');

    document.getElementById('active').textContent=a.length;
    document.getElementById('req').textContent=h.length;

    const tt=h.filter(x=>x.ttft_ms!=null).map(x=>x.ttft_ms);
    const tp=h.filter(x=>x.gen_tps!=null).map(x=>x.gen_tps);

    document.getElementById('ttft').textContent=tt.length?
      (tt.reduce((s,x)=>s+x,0)/tt.length/1000).toFixed(2)+'s':'—';

    document.getElementById('tps').textContent=tp.length?
      (tp.reduce((s,x)=>s+x,0)/tp.length).toFixed(1)+' tok/s':'—';

    document.getElementById('tokens').textContent=
      h.reduce((s,x)=>s+(x.total_tokens||0),0).toLocaleString();

    const c=d.cache||{};
    document.getElementById('cache').textContent=`${c.hits||0} hits`;
    const ms=d.models||{};
    const resident=['moe','dense'].filter(k=>ms[k]?.running).join(' + ')||'none';
    document.getElementById('backend').textContent=resident.toUpperCase();
    document.getElementById('modelRows').innerHTML=Object.entries(ms).map(([role,x])=>`
      <tr><td>${esc(role)}</td><td>${esc(x.model)}</td>
      <td class="${x.running?'good':'bad'}">${x.running?'RUNNING':'STOPPED'}</td>
      <td>${esc(x.endpoint)}</td></tr>`).join('');

    document.getElementById('activeRows').innerHTML=a.map(x=>`
      <tr>
        <td>${esc(x.route)}</td>
        <td>${esc(x.model)}</td>
        <td>${x.routing_level==null?'—':'L'+x.routing_level}</td>
        <td>${sec(x.ttft_ms)}</td>
        <td>${val(x.max_tokens)}</td>
        <td>${val(x.memory_available_gb)} GiB</td>
        <td>${val(x.chunks,0)}</td>
        <td>${Math.round(Date.now()/1000-(x.started||Date.now()/1000))}s</td>
      </tr>`).join('') ||
      '<tr><td colspan="8">No active requests</td></tr>';

    document.getElementById('historyRows').innerHTML=
      h.slice().reverse().map(x=>`
      <tr>
        <td>${new Date(x.ts*1000).toLocaleTimeString()}</td>
        <td title="${esc(x.route_reason)}">${esc(x.route||'—')}</td>
        <td>${esc(x.model)}</td>
        <td>${x.routing_level==null?'—':'L'+x.routing_level}</td>
        <td>${sec(x.router_ms)}</td>
        <td>${sec(x.swap_ms)}</td>
        <td>${x.cache_hit?'HIT':'—'}</td>
        <td>${sec(x.ttft_ms)}</td>
        <td>${val(x.prompt_tokens)}</td>
        <td>${val(x.completion_tokens)}</td>
        <td>${val(x.total_tokens)}</td>
        <td>${x.gen_tps?x.gen_tps.toFixed(1):'—'}</td>
        <td>${esc(x.finish_reason||'—')}</td>
        <td>${val(x.peak_available_gb)} GiB</td>
        <td class="${x.status>=400?'bad':'good'}">${x.status}</td>
      </tr>`).join('') ||
      '<tr><td colspan="15">No completed requests</td></tr>';

  }catch(e){
    document.getElementById('status').textContent='Dashboard error: '+e;
  }
}
setInterval(tick,1000); tick();
</script>
</body>
</html>
"""

DASHBOARD_V2 = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLX Inference Gateway</title><style>
:root{color-scheme:dark;--bg:#0b0e12;--panel:#151a21;--line:#2d3540;--muted:#8b98a8;--green:#65d58b;--blue:#68a8ff;--red:#ff7474;--amber:#e8b85b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2f7;font:14px -apple-system,BlinkMacSystemFont,"SF Pro Text",sans-serif}
header{padding:24px 28px 12px;display:flex;justify-content:space-between;align-items:end}h1{margin:0;font-size:25px}.sub,.muted{color:var(--muted)}
nav{display:flex;gap:6px;padding:0 28px;border-bottom:1px solid var(--line)}nav button{border:0;border-bottom:2px solid transparent;background:none;color:var(--muted);padding:13px 16px;cursor:pointer;font-weight:600}nav button.on{color:#fff;border-color:var(--blue)}
main{padding:22px 28px}.tab{display:none}.tab.on{display:block}.grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px}.k{font-size:12px;color:var(--muted)}.v{font-size:23px;font-weight:750;margin-top:7px}.good{color:var(--green)}.bad{color:var(--red)}.warn{color:var(--amber)}
.actions{display:flex;gap:9px;flex-wrap:wrap;margin:14px 0}button.action,select,input{background:#202731;color:#fff;border:1px solid #3a4553;border-radius:8px;padding:9px 12px}button.action{cursor:pointer}button.primary{background:#2463b5;border-color:#3475cb}button.danger{border-color:#8f4545}button:disabled{opacity:.45;cursor:not-allowed}textarea{width:100%;height:52px;min-height:52px;margin:8px 0 10px;background:#0c1015;color:#edf2f7;border:1px solid var(--line);border-radius:8px;padding:8px 11px;font:13px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;resize:vertical}
section{margin-top:23px;overflow:auto}h2{font-size:17px;margin:0 0 11px}table{width:100%;border-collapse:collapse;background:#101419;border:1px solid var(--line)}th,td{padding:9px;border-bottom:1px solid #242c35;text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600}.reason{white-space:normal;min-width:260px}.bar{height:7px;background:#29313b;border-radius:9px;overflow:hidden;margin-top:9px}.fill{height:100%;background:var(--green)}
.suite{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center}.notice{padding:10px 12px;border:1px solid var(--line);border-radius:8px;margin:12px 0;color:var(--muted)}
pre{white-space:pre-wrap;word-break:break-word;background:#0c1015;border:1px solid var(--line);border-radius:8px;padding:10px;max-width:780px;max-height:260px;overflow:auto}details summary{cursor:pointer;color:var(--blue)}
.comparegrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.output{max-width:none;min-height:220px;max-height:520px}.meta{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);margin:8px 0}.comparehead{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:11px}
.score{font-size:34px;font-weight:800;margin:8px 0}.rubric{display:grid;grid-template-columns:1fr auto;gap:7px;border-top:1px solid var(--line);padding:7px 0}.checks{margin:5px 0 12px;padding-left:18px;color:var(--muted)}.checks .pass{color:var(--green)}.checks .fail{color:var(--red)}
.verdicts{display:flex;gap:12px;flex-wrap:wrap;margin:7px 0 12px}.verdicts label{display:flex;gap:5px;align-items:center}.verdicts input[type=radio]{padding:0;margin:0}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}header{align-items:start;flex-direction:column;gap:6px}main,header{padding-left:15px;padding-right:15px}nav{padding-left:15px}}
@media(max-width:760px){.comparegrid{grid-template-columns:1fr}}
</style></head><body>
<header><div><h1>MLX Inference Gateway</h1><div class=sub id=status>Connecting…</div></div><div class=muted>Phase 2 · local-first control plane</div></header>
<nav><button class=on data-tab=overview>Overview</button><button data-tab=routing>Routing & cache</button><button data-tab=benchmarks>Benchmark Lab</button><button data-tab=lab>3-Axis Lab</button></nav>
<main>
<div class="tab on" id=overview>
 <div class=grid><div class=card><div class=k>Memory available</div><div class=v id=mem>—</div><div class=bar><div class=fill id=membar></div></div></div><div class=card><div class=k>Pressure</div><div class=v id=pressure>—</div></div><div class=card><div class=k>Active backend</div><div class=v id=backend>—</div></div><div class=card><div class=k>Active requests</div><div class=v id=active>0</div></div><div class=card><div class=k>Completed</div><div class=v id=req>0</div></div><div class=card><div class=k>Average TTFT</div><div class=v id=ttft>—</div></div><div class=card><div class=k>Average generation</div><div class=v id=tps>—</div></div><div class=card><div class=k>Relative compute</div><div class=v id=cost>—</div></div></div>
 <section><h2>Model state</h2><table><thead><tr><th>Role</th><th>Model</th><th>State</th><th>Endpoint</th></tr></thead><tbody id=modelRows></tbody></table></section>
 <section><h2>Active requests</h2><table><thead><tr><th>Route</th><th>Model</th><th>Level</th><th>TTFT</th><th>Memory</th><th>Age</th></tr></thead><tbody id=activeRows></tbody></table></section>
 <section><h2>Recent requests</h2><table><thead><tr><th>Time</th><th>Route</th><th>Reason</th><th>Level</th><th>Swap</th><th>TTFT</th><th>Tokens</th><th>Tok/s</th><th>Compute</th><th>Status</th></tr></thead><tbody id=historyRows></tbody></table></section>
</div>
<div class=tab id=routing>
 <div class=grid><div class=card><div class=k>Exact-cache entries</div><div class=v id=cacheEntries>0</div></div><div class=card><div class=k>Cache hits</div><div class=v id=cacheHits>0</div></div><div class=card><div class=k>Route affinity</div><div class=v id=pins>OFF</div></div><div class=card><div class=k>Idle model swapping</div><div class=v id=manualPin>OFF</div></div><div class=card><div class=k>Routing</div><div class=v id=routingState>—</div></div></div>
 <section><h2>Backend controls</h2><div class=notice>Every automatic request is scored by the deterministic routing engine (routing_rules.json): code above complexity level 4 uses dense, everything else uses the MoE workhorse. No client identity, session affinity, route timers, or idle switching are used. Explicit model selections and [model:...] / [complexity:...] controls override the lookup.</div><div class=actions><button class="action primary" onclick="switchModel('moe')">Activate MoE</button><button class="action primary" onclick="switchModel('dense')">Activate dense</button><button class="action danger" onclick="post('/control/cache/clear','Cache cleared')">Clear exact cache</button></div><div id=controlMsg class=muted></div></section>
 <section><h2>Routing decisions</h2><table><thead><tr><th>Time</th><th>Route</th><th>Profile</th><th>Level</th><th>Reason</th><th>Router time</th><th>Cache</th></tr></thead><tbody id=routeRows></tbody></table></section>
</div>
<div class=tab id=benchmarks>
 <div class=card><div class=suite><div><h2>Run a real Pi agent comparison</h2><div class=muted>Each model receives the same prompt in a fresh isolated Pi coding-agent session. Pi may reason, use tools, create files, and validate its work. Models run sequentially and the previous backend is restored.</div></div><button class="action primary" id=runBench onclick=runBenchmark()>Run benchmark</button></div><div class=actions><select id=suite onchange=showPrompt()></select><label id=portraitLabel style="display:none">Portrait subject <select id=portraitSubject onchange=showPrompt()></select></label><label><input type=checkbox id=bmoe checked> MoE</label><label><input type=checkbox id=bdense checked> Dense</label><span class=muted>128K context · up to 32,000 output tokens</span><button class=action onclick=resetPrompt()>Reset standard prompt</button><button class="action danger" onclick=clearBenchmarkHistory()>Clear benchmark history</button></div><div class=muted id=profileHint>Editable prompt sent identically to each selected model through Pi.</div><textarea id=promptEditor aria-label="Benchmark prompt"></textarea><div id=benchMsg class=muted></div></div>
 <section><h2>Live Pi activity</h2><div class=muted>The exact prompt and Pi's agent/tool activity appear here while the benchmark runs. Secrets are redacted and long payloads are shortened.</div><pre class=output id=liveBenchLog style="max-height:360px">No benchmark activity yet.</pre></section>
 <section><h2>Current and recent jobs</h2><table><thead><tr><th>Created</th><th>Suite</th><th>Routes</th><th>Status</th><th>Lifecycle stage</th><th>Error</th></tr></thead><tbody id=jobRows></tbody></table></section>
 <section><div class=comparehead><div><h2>Output Comparison Workspace</h2><div class=muted>Persistent side-by-side evidence for one benchmark run.</div></div><select id=compareJob onchange="selectComparison(this.value)"><option value="">Choose a completed run</option></select></div><div id=compareEmpty class=notice>Select a completed benchmark run to compare its model outputs.</div><div id=compareGrid class=comparegrid style="display:none"><div class=card><div class=k>Output A</div><div class=v id=compareATitle>—</div><div class=meta id=compareAMeta></div><div id=compareAEval></div><pre class=output id=compareA></pre><div id=compareAHitl></div></div><div class=card><div class=k>Output B</div><div class=v id=compareBTitle>—</div><div class=meta id=compareBMeta></div><div id=compareBEval></div><pre class=output id=compareB></pre><div id=compareBHitl></div></div></div><div id=comparePromptWrap style="display:none"><h3>Exact shared prompt</h3><pre id=comparePrompt></pre><div class=notice>Readiness is an evidence-backed rubric for this response, not a claim of global model accuracy.</div></div></section>
 <section><h2>Results</h2><p class=muted>Total time includes loading and the full Pi run. Tool calls count all executed tools, including retries. Average generation TPS = total output tokens / summed generation time across model calls; excludes tool execution and per-call time to first token. Older runs without generation timing show —.</p><table><thead><tr><th>Time</th><th>Suite</th><th>Route</th><th>Model</th><th>Readiness</th><th>Evidence</th><th>Load/readiness</th><th>TTFT</th><th>Total time incl. load</th><th>Pi run time</th><th>Tool calls</th><th>Model calls</th><th>Prompt tokens</th><th>Completion</th><th>Average generation TPS</th><th>Available before → after</th><th>Swap written</th><th>Compute</th></tr></thead><tbody id=benchRows></tbody></table></section>
</div>
<div class=tab id=lab>
 <div class=card><div class=suite><div><h2>Multi-axis benchmark lab</h2><div class=muted>Measure one (model × framework × harness) cell. The framework loads the model, runs the harness, then unloads it (load-run-unload residency). Jobs run sequentially. Incompatible model×framework combos are greyed out, not errored.</div></div><button class="action primary" id=runLab onclick=runLabCell()>Run cell</button></div>
 <div class=actions>
  <label>Model <select id=labModel onchange=applyLabAvailability()></select></label>
  <label>Framework <select id=labFramework onchange=applyLabAvailability()></select></label>
  <label>Harness <select id=labHarness></select></label>
  <label>Iterations <input id=labIters type=number value=3 min=1 max=20 style=width:70px></label>
  <label>Max tokens <input id=labMaxTok type=number value=128 min=16 max=4096 style=width:90px></label>
  <button class="action danger" onclick=clearLabHistory()>Clear lab history</button>
 </div>
 <label style="display:block;margin-top:10px"><b>Prompt sent to the model</b> <span class=muted>(edit to benchmark a different prompt; a unique suffix is appended per iteration to avoid cache hits)</span><textarea id=labPrompt rows=4 style="width:100%;margin-top:4px;font-family:ui-monospace,Menlo,monospace;font-size:12px"></textarea></label>
 <div class=muted id=labMsg>Configure a cell and run it. Results appear below.</div>
 </div>
 <section><h2>Current lab job</h2><div id=labJob class=notice>No lab job running.</div></section>
 <section><h2>Lab results</h2><p class=muted>TTFT and TPS are decode metrics for the raw harness (streaming). For agent harnesses (pi, omp) TTFT is null and TPS is output tokens over total task time. Token counts are exact for raw/omp (from usage) and estimated (~4 chars/token) for pi.</p><table><thead><tr><th>Time</th><th>Model</th><th>Framework</th><th>Harness</th><th>Status</th><th>TTFT</th><th>Total</th><th>Tokens</th><th>TPS</th><th>Iters</th><th>Error</th></tr></thead><tbody id=labRows></tbody></table></section>
</div></main>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),sec=x=>x==null?'—':(x/1000).toFixed(2)+'s',val=(x,d='—')=>x==null?d:x;
let suiteData={},benchmarkResults=[],benchmarkJobs=[],selectedCompareJob='',verdictDrafts={};function showPrompt(){const x=suiteData[$('suite').value];if(x){const subjects=x.subjects||[];$('portraitLabel').style.display=subjects.length?'':'none';if(subjects.length&&!$('portraitSubject').options.length){$('portraitSubject').innerHTML=subjects.map(name=>`<option>${esc(name)}</option>`).join('')}$('promptEditor').value=x.prompt_template?x.prompt_template.replace('{figure}',$('portraitSubject').value):x.prompt;$('profileHint').textContent=`Pi agent profile: thinking ${x.thinking?'on':'off'} · reasoning ${x.reasoning_effort} · 128K context · up to 32,000 output tokens. The prompt and environment are identical for each selected model.`}}function resetPrompt(){showPrompt();$('benchMsg').textContent='Standard prompt restored.'}
function comparisonReady(jobId){const rows=benchmarkResults.filter(x=>x.job_id===jobId),job=benchmarkJobs.find(x=>x.id===jobId),expected=job?.routes||rows[0]?.requested_routes||[];return expected.length>=2&&expected.every(route=>rows.some(x=>x.route===route&&x.status>=200&&x.status<300))&&(!job||job.status==='complete')}
function renderLivePi(jobs){const job=[...jobs].reverse().find(j=>['queued','running'].includes(j.status))||jobs[jobs.length-1],lines=job?.live_log||[];if(!lines.length){$('liveBenchLog').textContent='No benchmark activity yet.';return}const wasNearBottom=$('liveBenchLog').scrollHeight-$('liveBenchLog').scrollTop-$('liveBenchLog').clientHeight<45;$('liveBenchLog').textContent=lines.map(x=>`[${new Date(x.ts*1000).toLocaleTimeString()}] [${(x.route||'all').toUpperCase()}] ${x.kind}: ${x.message}`).join('\n');if(wasNearBottom)$('liveBenchLog').scrollTop=$('liveBenchLog').scrollHeight}
function enforceComparisonLocks(){const ordered=benchmarkResults.slice().reverse();document.querySelectorAll('#benchRows button').forEach((button,index)=>{const ready=comparisonReady(ordered[index]?.job_id);button.disabled=!ready;button.textContent=ready?'Compare output':'Waiting for both models'});[...$('compareJob').options].slice(1).forEach(option=>{if(!comparisonReady(option.value))option.remove()});if(selectedCompareJob&&!comparisonReady(selectedCompareJob)){selectedCompareJob='';$('compareGrid').style.display='none';$('comparePromptWrap').style.display='none';$('compareEmpty').style.display='block'}}
async function clearBenchmarkHistory(){if(!confirm('Clear all stored benchmark jobs and results?'))return;try{const r=await fetch('/benchmarks/clear',{method:'POST'}),d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);benchmarkResults=[];benchmarkJobs=[];selectedCompareJob='';$('benchMsg').textContent=`Cleared ${d.removed_results} results.`;await tick()}catch(e){$('benchMsg').textContent='Error: '+e.message}}
function compareMeta(x){return x?`<span>TTFT ${sec(x.ttft_ms)}</span><span><b>Total time ${sec(x.total_time_ms??(x.latency_ms==null?null:x.latency_ms+(x.swap_ms||0)))}</b></span><span>Pi run time ${sec(x.latency_ms)}</span><span>Model runtime ${sec(x.model_latency_ms)}</span><span>${val(x.pi_model_requests)} model calls</span><span>${val(x.pi_tool_calls)} tool calls · ${val(x.pi_tool_errors)} errors</span><span>${val(x.completion_tokens)} tokens</span><span><b>${val(x.average_tps)} average generation TPS</b></span><span>Compute ${val(x.compute_score)}</span>${x.validation?`<span>Functional validation ${esc(x.validation.runtime_status||x.validation.status)} · ${x.validation.passed}/${x.validation.total}</span>`:''}`:''}
function capturedOutput(x){if(!x)return 'No model result.';if(x.response)return x.response;const reasoningChars=x.reasoning_chars??(x.reasoning?.length||0);if(reasoningChars)return `NO FINAL ANSWER EMITTED\n\nThe model used its output budget for ${reasoningChars.toLocaleString()} characters of reasoning and reached ${x.finish_reason||'an unknown finish state'} before producing final content. This is a benchmark failure, not a capture failure.`;if(x.parameters)return 'The model returned no final-answer content.';return 'Response was not captured by the gateway version used for this historical run.'}
function evaluationHtml(x){const e=x?.evaluation;if(!e)return '<div class=notice>No evaluation was recorded for this earlier run.</div>';const cats=Object.entries(e.categories||{}).map(([name,c])=>`<div class=rubric><span>${esc(name.replaceAll('_',' '))}</span><b>${c.score}/100</b></div><ul class=checks>${(c.checks||[]).map(k=>`<li class=${k.passed?'pass':'fail'}>${k.passed?'PASS':'FAIL'} · ${esc(k.label)} — ${esc(k.detail)}</li>`).join('')}</ul>`).join('');const qualified=x.qualification?.qualified!==false;const headline=qualified?`${e.readiness_score}/100`:'DISQUALIFIED';const why=qualified?'':`<div class="notice bad">${(x.qualification?.reasons||[]).map(esc).join(' · ')}</div>`;return `<div class="score ${qualified?'':'bad'}">${headline}</div>${why}<div class=muted>Readiness · ${esc(e.method)} · confidence ${esc(e.confidence)}</div>${cats}`}
function htmlFromResult(x){const raw=x?.response||'',m=raw.match(/```(?:html|svg)?\s*([\s\S]*?)```/i);return (m?m[1]:raw).trim()}
async function copyArtifact(id){const x=benchmarkResults.find(r=>r.id===id);if(!x)return;await navigator.clipboard.writeText(htmlFromResult(x));$('benchMsg').textContent='Artifact copied to clipboard.'}
function openArtifact(id){window.open('/benchmarks/artifact/'+encodeURIComponent(id),'_blank','noopener,noreferrer')}
function rememberVerdict(id,field,value){verdictDrafts[id]={...(verdictDrafts[id]||{}),[field]:value}}
function rememberNotes(id,value){verdictDrafts[id]={...(verdictDrafts[id]||{}),notes:value}}
function hitlHtml(x){if(!x||!suiteData[x.suite]?.artifact_extension)return '';const h=x.human_verdict||{},d=verdictDrafts[x.id]||{},v=d.verdict??h.verdict??'',c=d.comparison??h.comparison??'',notes=d.notes??h.notes??'';const radio=(field,value,label,selected)=>`<label><input type=radio name="${field}-${x.id}" value="${value}" ${selected===value?'checked':''} onchange="rememberVerdict('${x.id}','${field}',this.value)"> ${label}</label>`;return `<div class=notice><b>Human evaluation</b><p>${esc(suiteData[x.suite]?.review_checklist||'Test the generated artifact against the prompt.')}</p><div class=actions><button class="action primary" onclick="openArtifact('${x.id}')">Open artifact</button><button class=action onclick="copyArtifact('${x.id}')">Copy artifact</button></div><div class=muted>Functional outcome</div><div class=verdicts>${radio('verdict','pass','Pass',v)}${radio('verdict','partial','Partial',v)}${radio('verdict','fail','Fail',v)}</div><div class=muted>Final product compared with the other model</div><div class=verdicts>${radio('comparison','better','Better than the other model',c)}${radio('comparison','similar','Similar',c)}${radio('comparison','worse','Worse than the other model',c)}</div><input id="notes-${x.id}" value="${esc(notes)}" oninput="rememberNotes('${x.id}',this.value)" placeholder="Optional test notes" style="width:100%"><div class=actions><button class=action onclick="saveVerdict('${x.id}')">Save evaluation</button></div><div class=muted>${h.recorded_at?'Saved '+new Date(h.recorded_at*1000).toLocaleString():'Open and test the exact generated artifact, then record the result.'}</div></div>`}
async function saveVerdict(id){const verdict=document.querySelector(`input[name="verdict-${id}"]:checked`)?.value||'',comparison=document.querySelector(`input[name="comparison-${id}"]:checked`)?.value||'',notes=$('notes-'+id).value;if(!verdict){$('benchMsg').textContent='Choose Pass, Partial, or Fail first.';return}const r=await fetch('/benchmarks/result/'+encodeURIComponent(id)+'/verdict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({verdict,comparison,notes})}),d=await r.json();if(!r.ok){$('benchMsg').textContent='Error: '+(d.error||r.statusText);return}delete verdictDrafts[id];$('benchMsg').textContent='Human evaluation saved.';await tick()}
function selectComparison(jobId){if(jobId&&!comparisonReady(jobId))return;selectedCompareJob=jobId;$('compareJob').value=jobId;const rows=benchmarkResults.filter(x=>x.job_id===jobId);const a=rows.find(x=>x.route==='moe')||rows[0],b=rows.find(x=>x.route==='dense')||rows[1];if(!a){$('compareGrid').style.display='none';$('comparePromptWrap').style.display='none';$('compareEmpty').style.display='block';return}$('compareEmpty').style.display='none';$('compareGrid').style.display='grid';$('comparePromptWrap').style.display='block';$('compareATitle').textContent=(a.route||'A').toUpperCase();$('compareAMeta').innerHTML=compareMeta(a);$('compareAEval').innerHTML=evaluationHtml(a);$('compareA').textContent=capturedOutput(a);$('compareAHitl').innerHTML=hitlHtml(a);$('compareBTitle').textContent=b?(b.route||'B').toUpperCase():'No second model';$('compareBMeta').innerHTML=compareMeta(b);$('compareBEval').innerHTML=b?evaluationHtml(b):'<div class=notice>No second model result.</div>';$('compareB').textContent=b?capturedOutput(b):'Run the benchmark with both MoE and dense selected.';$('compareBHitl').innerHTML=b?hitlHtml(b):'';$('comparePrompt').textContent=a.prompt||b?.prompt||'Prompt was not captured by the gateway version used for this run.'}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button,.tab').forEach(x=>x.classList.remove('on'));b.classList.add('on');$(b.dataset.tab).classList.add('on')});
async function post(url,msg,body){try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);$('controlMsg').textContent=msg||'Done';await tick();return d}catch(e){$('controlMsg').textContent='Error: '+e.message;throw e}}
async function switchModel(route){await post('/control/model/'+route,route.toUpperCase()+' is active and remains resident until another route is needed.')}
async function runBenchmark(){const routes=[];if($('bmoe').checked)routes.push('moe');if($('bdense').checked)routes.push('dense');if(!routes.length){$('benchMsg').textContent='Select at least one model.';return}const prompt=$('promptEditor').value.trim();if(!prompt){$('benchMsg').textContent='Prompt cannot be empty.';return}$('runBench').disabled=true;try{const r=await fetch('/benchmarks/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({suite:$('suite').value,routes,prompt})});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);$('benchMsg').textContent=`Pi benchmark queued using ${d.prompt_mode} prompt ${d.prompt_hash.slice(0,8)}. Other gateway clients will be held out until both agent runs finish.`}catch(e){$('benchMsg').textContent='Error: '+e.message}finally{$('runBench').disabled=false}}
function render(d){const h=d.history||[],a=d.active||[],m=d.memory||{},ms=d.models||{},c=d.cache||{},aff=d.route_affinity||{},pin=d.manual_pin||{},b=d.benchmarks||{};$('status').textContent='Updated '+new Date().toLocaleTimeString();$('mem').textContent=val(m.available_gb)+' GiB';$('membar').style.width=Math.max(0,Math.min(100,(m.available_gb/(m.total_gb||1))*100))+'%';$('pressure').textContent=m.hard_pressure?'HARD':m.pressure?'GUARD':'NORMAL';$('pressure').className='v '+(m.hard_pressure?'bad':m.pressure?'warn':'good');$('active').textContent=a.length;$('req').textContent=h.length;const resident=['moe','dense'].filter(k=>ms[k]?.running).join(' + ')||'none';$('backend').textContent=resident.toUpperCase();const tt=h.filter(x=>x.ttft_ms!=null),tp=h.filter(x=>x.gen_tps!=null);$('ttft').textContent=tt.length?sec(tt.reduce((s,x)=>s+x.ttft_ms,0)/tt.length):'—';$('tps').textContent=tp.length?(tp.reduce((s,x)=>s+x.gen_tps,0)/tp.length).toFixed(1)+' t/s':'—';$('cost').textContent=h.reduce((s,x)=>s+(x.compute_score||0),0).toFixed(1);$('cacheEntries').textContent=c.entries||0;$('cacheHits').textContent=c.hits||0;$('pins').textContent='OFF';$('manualPin').textContent='OFF';$('manualPin').className='v good';$('routingState').textContent=d.config?.routing_enabled?'DETERMINISTIC':'DISABLED';
$('modelRows').innerHTML=Object.entries(ms).map(([k,x])=>`<tr><td>${esc(k)}</td><td>${esc(x.model)}</td><td class=${x.running?'good':'bad'}>${x.running?'RUNNING':'STOPPED'}</td><td>${esc(x.endpoint)}</td></tr>`).join('');$('activeRows').innerHTML=a.map(x=>`<tr><td>${esc(x.route)}</td><td>${esc(x.model)}</td><td>${x.routing_level==null?'—':'L'+x.routing_level}</td><td>${sec(x.ttft_ms)}</td><td>${val(x.memory_available_gb)} GiB</td><td>${Math.round(Date.now()/1000-x.started)}s</td></tr>`).join('')||'<tr><td colspan=6>No active requests</td></tr>';
$('historyRows').innerHTML=h.slice().reverse().slice(0,100).map(x=>`<tr><td>${new Date(x.ts*1000).toLocaleTimeString()}</td><td>${esc(x.route)}</td><td class=reason>${esc(x.route_reason)}</td><td>${x.routing_level==null?'—':'L'+x.routing_level}</td><td>${sec(x.swap_ms)}</td><td>${sec(x.ttft_ms)}</td><td>${val(x.total_tokens)}</td><td>${x.gen_tps?x.gen_tps.toFixed(1):'—'}</td><td>${val(x.compute_score)}</td><td class=${x.status>=400?'bad':'good'}>${x.status}</td></tr>`).join('')||'<tr><td colspan=10>No requests</td></tr>';$('routeRows').innerHTML=h.slice().reverse().slice(0,100).map(x=>`<tr><td>${new Date(x.ts*1000).toLocaleTimeString()}</td><td>${esc(x.route)}</td><td>${esc(x.task_type||'—')} · ${esc(x.effort||'—')} · thinking ${x.thinking==null?'—':x.thinking?'on':'off'} · ${val(x.effective_max_tokens)} tokens</td><td>${x.routing_level==null?'—':'L'+x.routing_level}</td><td class=reason>${esc(x.route_reason)}</td><td>${sec(x.router_ms)}</td><td>${x.cache_hit?'HIT':'—'}</td></tr>`).join('');
const suites=b.suites||{};suiteData=suites;if(!$('suite').options.length){$('suite').innerHTML=Object.entries(suites).map(([id,x])=>`<option value=${esc(id)}>${esc(x.name)} — ${esc(x.description)}</option>`).join('');showPrompt()}const jobs=b.jobs||[];renderLivePi(jobs);$('jobRows').innerHTML=jobs.slice().reverse().map(j=>`<tr><td>${new Date(j.created_at*1000).toLocaleTimeString()}</td><td>${esc(j.suite)}</td><td>${esc(j.routes.join(', '))}</td><td class=${j.status==='failed'?'bad':j.status==='complete'?'good':'warn'}>${esc(j.status)}</td><td>${esc(j.stage||j.current_route||'—')}</td><td>${esc(j.error||j.restore_error||'—')}</td></tr>`).join('')||'<tr><td colspan=6>No benchmark jobs</td></tr>';$('runBench').disabled=a.length>0||jobs.some(j=>['queued','running'].includes(j.status));benchmarkResults=b.history||[];const groups=[...new Set(benchmarkResults.map(x=>x.job_id).filter(Boolean))].reverse();const prior=$('compareJob').value;$('compareJob').innerHTML='<option value="">Choose a completed run</option>'+groups.map(id=>{const x=benchmarkResults.find(r=>r.job_id===id);return `<option value="${esc(id)}">${esc(x?.suite||'benchmark')} · ${new Date((x?.ts||0)*1000).toLocaleString()}</option>`}).join('');if(selectedCompareJob&&groups.includes(selectedCompareJob)){selectComparison(selectedCompareJob)}else if(prior&&groups.includes(prior)){selectComparison(prior)}$('benchRows').innerHTML=benchmarkResults.slice().reverse().map(x=>`<tr><td>${new Date(x.ts*1000).toLocaleTimeString()}</td><td>${esc(x.suite)}</td><td>${esc(x.route)}</td><td>${esc(x.model)}</td><td><b>${x.evaluation?.readiness_score??'—'}</b>${x.evaluation?' / 100':''}</td><td><button class=action onclick="selectComparison('${esc(x.job_id||'')}')">Compare output</button></td><td>${sec(x.swap_ms)}</td><td>${sec(x.ttft_ms)}</td><td>${sec(x.total_time_ms??(x.latency_ms==null?null:x.latency_ms+(x.swap_ms||0)))}</td><td>${sec(x.latency_ms)}</td><td>${val(x.pi_tool_calls)}</td><td>${val(x.pi_model_requests)}</td><td>${val(x.prompt_tokens)}</td><td>${val(x.completion_tokens)}</td><td>${val(x.average_tps)}</td><td>${val(x.available_before_gb)} → ${val(x.available_after_gb)} GiB</td><td>${x.swap_delta_mb==null?'Unavailable':x.swap_delta_mb+' MB'}</td><td>${val(x.compute_score)}</td></tr>`).join('')||'<tr><td colspan=18>No benchmark results</td></tr>'}
async function tick(){try{const r=await fetch('/metrics',{cache:'no-store'}),d=await r.json();benchmarkJobs=d.benchmarks?.jobs||[];render(d);enforceComparisonLocks()}catch(e){$('status').textContent='Dashboard error: '+e.message}}setInterval(tick,1200);tick();
// ---- 3-Axis Lab ----
let labJobId=null,labPollTimer=null,labOptions=null;
async function loadLabOptions(){try{const r=await fetch('/lab/options',{cache:'no-store'}),o=await r.json();const fill=(id,items)=>{const el=$(id);if(!el)return;const cur=el.value;el.innerHTML=items.map(x=>`<option value="${esc(x.key)}">${esc(x.label)}</option>`).join('');if([...el.options].some(x=>x.value===cur))el.value=cur};fill('labModel',o.models);fill('labFramework',o.frameworks);fill('labHarness',o.harnesses);labOptions=o;if($('labPrompt')&&!$('labPrompt').value)$('labPrompt').value=o.default_prompt||'';applyLabAvailability()}catch(e){$('labMsg').textContent='Failed to load lab options: '+e.message}}
function applyLabAvailability(){if(!labOptions)return;const model=$('labModel'),framework=$('labFramework');if(!model||!framework)return;const mSpec=labOptions.models.find(m=>m.key===model.value);for(const opt of framework.options){const ok=!mSpec||!!mSpec.frameworks[opt.value];opt.disabled=!ok}if(framework.options[framework.selectedIndex]&&framework.options[framework.selectedIndex].disabled){const first=[...framework.options].find(o=>!o.disabled);if(first)framework.value=first.value}}
async function renderLabResults(){try{const r=await fetch('/lab/results',{cache:'no-store'}),d=await r.json();const rows=(d.history||[]).map(x=>`<tr><td>${new Date((x.created_at||0)*1000).toLocaleTimeString()}</td><td>${esc(x.model)}</td><td>${esc(x.framework)}</td><td>${esc(x.harness)}</td><td class=${x.status==='success'?'good':'bad'}>${esc(x.status)}</td><td>${x.avg_ttft_ms==null?'—':sec(x.avg_ttft_ms)}</td><td>${x.avg_total_ms==null?'—':sec(x.avg_total_ms)}</td><td>${val(x.avg_tokens)}</td><td>${x.avg_tps?x.avg_tps.toFixed(1):'—'}</td><td>${val(x.iterations)}</td><td class=reason>${esc(x.error||'')}</td></tr>`).join('');$('labRows').innerHTML=rows||'<tr><td colspan=11>No lab results yet</td></tr>'}catch(e){}}
async function runLabCell(){const framework=$('labFramework').value,model=$('labModel').value,harness=$('labHarness').value,iterations=parseInt($('labIters').value,10)||3,max_tokens=parseInt($('labMaxTok').value,10)||128,prompt=($('labPrompt')?$('labPrompt').value:'');$('runLab').disabled=true;$('labMsg').textContent='Queuing lab cell…';try{const r=await fetch('/lab/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({framework,model,harness,iterations,max_tokens,prompt})});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);labJobId=d.id;pollLabJob(d.id)}catch(e){$('labMsg').textContent='Error: '+e.message;$('runLab').disabled=false}}
async function pollLabJob(jobId){clearInterval(labPollTimer);labPollTimer=setInterval(async()=>{try{const r=await fetch('/lab/status/'+encodeURIComponent(jobId),{cache:'no-store'}),j=await r.json();const wd=j.result&&j.result.work_dir?` · output: <code>${esc(j.result.work_dir)}</code>`:'';$('labJob').innerHTML=`<b>${esc(j.framework)} × ${esc(j.model)} × ${esc(j.harness)}</b> — status: <span class=${j.status==='failed'?'bad':j.status==='done'?'good':'warn'}>${esc(j.status)}</span> · stage: ${esc(j.stage||'—')}${j.result&&j.result.status==='success'?` · TTFT ${j.result.avg_ttft_ms==null?'—':sec(j.result.avg_ttft_ms)} · total ${sec(j.result.avg_total_ms)} · ${val(j.result.avg_tokens)} tok · ${j.result.avg_tps?j.result.avg_tps.toFixed(1):'—'} t/s${wd}`:''}${j.status==='running'&&j.stage_at?` <span class=muted>(updated ${Math.round((Date.now()/1000-j.stage_at))}s ago)</span>`:''}`;if(j.status==='done'||j.status==='failed'){clearInterval(labPollTimer);labPollTimer=null;$('runLab').disabled=false;$('labMsg').textContent=j.status==='done'?'Lab cell complete.':'Lab cell failed: '+(j.result?.error||'unknown');renderLabResults()}}catch(e){}},1500)}
async function clearLabHistory(){try{const r=await fetch('/lab/clear',{method:'POST'});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);renderLabResults()}catch(e){$('labMsg').textContent='Error: '+e.message}}
loadLabOptions();renderLabResults();
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_V2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
