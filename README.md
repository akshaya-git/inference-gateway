# Local Inference Gateway

A transparent, OpenAI-compatible inference gateway for local LLM inference on Apple Silicon. It sits between your application (e.g., Pi coding agent) and local model servers, intelligently routing each request to the best-suited model based on task complexity.

## Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────────────────────┐
│   Client    │     │  Inference Gateway  │     │         Model Servers           │
│  (Pi, etc)  │────▶│   :9000 (FastAPI)   │────▶│  oMLX :8080                     │
│             │     │                     │     │  ┌─────────────────────────────┐ │
│  /v1/chat/  │     │  ┌───────────────┐  │     │  │  Arch-Router-1.5B (judge)  │ │
│  completions│     │  │  Judge Router │  │     │  │  (small, always loaded)    │ │
│             │     │  │  (1.5B MoE)   │  │     │  ├─────────────────────────────┤ │
│  /v1/models │     │  └───────┬───────┘  │     │  │  Qwen3.6-35B-A3B (MoE)     │ │
└─────────────┘     │          │          │     │  │  (fast workhorse)          │ │
                    │          ▼          │     │  ├─────────────────────────────┤ │
                    │  ┌───────────────┐  │     │  │  Qwen3.8-27B (dense)        │ │
                    │  │  Route Select │  │     │  │  (quality specialist)       │ │
                    │  └───────┬───────┘  │     │  └─────────────────────────────┘ │
                    │          │          │     └─────────────────────────────────┘
                    │          ▼          │
                    │  ┌───────────────┐  │
                    │  │  Memory Guard │  │
                    │  │  Admission    │  │
                    │  │  Control      │  │
                    │  └───────────────┘  │
                    └─────────────────────┘
```

## How It Works

### 1. Request Flow

1. **Client sends request** to `POST /v1/chat/completions` with `model: "gateway-auto"` (or `auto`, `inference-gateway`, etc.)
2. **Gateway checks cache** — exact-match, non-streaming responses are cached (TTL: 300s, max 128 entries)
3. **Judge model evaluates** — the small Arch-Router-1.5B model analyzes the request context and selects the best route
4. **Memory guard** — if available memory is below threshold, request waits in queue
5. **Model swap** (if needed) — if the selected model isn't loaded, gateway swaps it in via oMLX admin API
6. **Request forwarded** to the selected model server
7. **Response streamed** back to client with full observability

### 2. Routing Logic

The judge model receives:
- Recent conversation context (configurable, default 24K chars)
- Current runtime state (which models are loaded)
- Route descriptions (what each model is good at)

It returns a route decision: `moe_workhorse` or `dense_specialist`

**Route profiles:**
- **MoE (Qwen3.6-35B-A3B)**: Fast workhorse for structured output, tool use, data analysis, summarization, general reasoning, routine coding
- **Dense (Qwen3.8-27B)**: High-quality specialist for complex code review, architecture redesign, difficult debugging, demanding code generation

**Explicit routing:** Clients can bypass the judge by requesting specific model IDs:
- `gateway-moe` → always MoE
- `gateway-dense` → always dense

### 3. Memory Protection

The gateway uses **admission control** (not throttling):

| Threshold | Value | Action |
|-----------|-------|--------|
| `MEMORY_GUARD_GB` | 3 GiB | New requests wait in queue |
| `MEMORY_HARD_GB` | 1.5 GiB | Hard safety threshold |
| `MAX_ACTIVE_REQUESTS` | 1 | Matches MLX single-user scheduler |

When memory is low, requests queue until memory recovers. This is safer than throttling active generation.

### 4. Model Swapping

Models are swapped via the oMLX admin API:
1. Unload current model
2. Load target model
3. Warm up with a small test request
4. Proceed with inference

Swaps are serialized with a lock to prevent concurrent transitions.

## Components

### proxy.py (main gateway)

- **FastAPI app** with OpenAI-compatible endpoints
- **Judge router** — calls the small model to classify requests
- **Memory watcher** — polls available memory, updates admission state
- **Response cache** — exact-match, non-streaming only
- **Benchmark lab** — runs Pi agent sessions against both models
- **Dashboard** — real-time browser UI at `http://127.0.0.1:9000/`

### router.yaml (configuration)

```yaml
judge:
  model: Arch-Router-1.5B-mlx-8Bit
  max_tokens: 64
  temperature: 0.0
  timeout_seconds: 30

routes:
  moe:
    model: Qwen3.6-35B-A3B-oQ5e-mtp
    judge_name: moe_workhorse
    description: "Fast workhorse for..."
  dense:
    model: Qwen3.8-27B-oQ4e-mtp
    judge_name: dense_specialist
    description: "High-quality specialist for..."

routing:
  evaluate_every_request: true
  context_characters: 24000
  fallback_route: moe
```

### inference-stack.sh (stack control)

Bash script that manages the full stack:
- `start` — start oMLX + gateway, load MoE
- `dense-on` — swap to dense model
- `moe-on` — swap back to MoE
- `status` — show model states
- `logs` — tail logs
- `stop` — stop everything

## API Endpoints

### OpenAI-Compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/models` | GET | List available models |

### Gateway Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model states |
| `/metrics` | GET | Full metrics: active, history, cache, benchmarks |
| `/` | GET | Browser dashboard |
| `/control/model/moe` | POST | Force switch to MoE |
| `/control/model/dense` | POST | Force switch to dense |
| `/control/cache/clear` | POST | Clear response cache |
| `/benchmarks` | GET | List benchmark suites + results |
| `/benchmarks/run` | POST | Start a benchmark job |
| `/benchmarks/clear` | POST | Clear benchmark history |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_HOST` | `127.0.0.1` | Gateway bind address |
| `PROXY_PORT` | `9000` | Gateway port |
| `OMLX_UPSTREAM` | `http://127.0.0.1:8080` | oMLX server URL |
| `ROUTER_MODEL` | `Arch-Router-1.5B-mlx-8Bit` | Judge model ID |
| `MOE_MODEL` | `Qwen3.6-35B-A3B-oQ5e-mtp` | MoE model ID |
| `DENSE_MODEL` | `Qwen3.8-27B-oQ4e-mtp` | Dense model ID |
| `ROUTING_ENABLED` | `true` | Enable judge routing |
| `CACHE_ENABLED` | `true` | Enable response cache |
| `CACHE_TTL_SEC` | `300` | Cache TTL (seconds) |
| `CACHE_MAX_ENTRIES` | `128` | Max cache entries |
| `MEMORY_GUARD_GB` | `3` | Memory guard threshold (GiB) |
| `MEMORY_HARD_GB` | `1.5` | Hard memory threshold (GiB) |
| `MAX_ACTIVE_REQUESTS` | `1` | Max concurrent requests |
| `QUEUE_ON_MEMORY_PRESSURE` | `true` | Queue instead of fail on low memory |
| `ROUTER_CONTEXT_CHARS` | `24000` | Context chars sent to judge |
| `ROUTER_TIMEOUT_SEC` | `30` | Judge timeout (seconds) |
| `SWAP_TIMEOUT_SEC` | `240` | Model swap timeout (seconds) |

## Installation

```bash
cd mlx_proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Start the full stack (oMLX + gateway + MoE)
~/Documents/inference-stack.sh start

# Open the dashboard
open http://127.0.0.1:9000/
```

## Pointing Pi at the Gateway

Change Pi's local provider base URL from:
```
http://127.0.0.1:8080/v1
```
to:
```
http://127.0.0.1:9000/v1
```

Use `gateway-auto` as the model ID to enable judge routing.

## Benchmark Lab

The gateway includes a benchmark lab that runs **real Pi agent sessions** against both models:

- **Quick comparison** — short explanation task
- **Bill-splitter HITL** — builds a real HTML artifact
- **Systems reasoning** — architecture analysis

Each benchmark:
1. Loads the model (measures swap time)
2. Runs a fresh Pi agent session
3. Captures full event stream (tool calls, reasoning, output)
4. Evaluates output with a static rubric
5. Restores the original backend

Results are persisted and viewable in the dashboard.

## Design Principles

1. **Transparent proxy** — clients see standard OpenAI API, no special SDK needed
2. **Local-first** — all inference happens on-device, no cloud dependency
3. **Memory-safe** — admission control prevents OOM crashes
4. **Observable** — every request is tracked with full metrics
5. **Extensible** — clean separation of routing, caching, and model management

## Roadmap

- [ ] CI/CD pipeline with GitHub Actions
- [ ] Additional benchmark suites (5+ new tests)
- [ ] Routing rationale capture (why did judge choose this model?)
- [ ] Instruction refinement loop (learn from routing outcomes)
- [ ] Generic model backend support (MLX, OMLX, LLaMA, etc.)
- [ ] Open-source release with community documentation
