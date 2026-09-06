# Task 10 — Multi-axis benchmark lab

Measure one **(model × framework × harness)** cell at a time, with
**load → run → unload** residency so the benchmark framework's copy of the
model is transient on top of the always-resident oMLX routing models.

## Axes

### Models (logical)
| key | label | oMLX id | MLX path | llama.cpp GGUF | Ollama name |
|-----|-------|---------|----------|----------------|-------------|
| `qwen3.6-35b-a3b` | Qwen3.6 35B-A3B (6-bit) | `mlx-community--Qwen3.6-35B-A3B-6bit` | HF snapshot | `models/gguf/Qwen3.6-35B-A3B-UD-Q6_K.gguf` | `bench/qwen36-35b-a3b-q6` |
| `qwen3.8-27b` | Qwen3.8 27B (6-bit) | `scottlowry--Qwen3.8-27B-oQ6e-mtp` | HF snapshot | `models/gguf/Qwen3.8-27B-UD-Q6_K.gguf` | `bench/qwen38-27b-q6` |

The registry in `benchmark_lab.py` (`MODELS`) separates the **server-start
path** (`mlx_path`, `llamacpp_path`) from the **request id** (`mlx_id`,
`llamacpp_id`, `omlx`, `ollama`). Swapping a model = edit the registry (and
`router.yaml` for the routing models).

### Frameworks
| key | what | port | residency |
|-----|------|------|-----------|
| `omlx` | oMLX (resident, OpenAI-compatible) | 8000 | always resident; admin load/unload |
| `mlx` | `mlx_lm.server` (bench venv) | 8200 | started per cell, killed after |
| `llamacpp` | `llama-server` (brew) | 8300 | started per cell, killed after |
| `ollama` | Ollama (brew) | 11434 | model created from GGUF, deleted after |

`FrameworkManager.ensure(framework, model)` returns `(endpoint, model_id)`;
`release()` tears the framework's copy down. oMLX uses the admin API
(`/admin/api/models/{id}/load|unload`); the others are subprocess servers.

### Harnesses
| key | what | metrics |
|-----|------|---------|
| `raw` | streaming HTTP (baseline) | TTFT, total, exact tokens (usage), decode TPS |
| `pi` | pi coding agent (`PI_CODING_AGENT_DIR` isolated) | total task time, est. tokens |
| `omp` | omp built-in `bench` (`PI_CODING_AGENT_DIR` isolated) | TTFT, total, exact tokens, generation TPS |
| `sisyphus` | opencode + Sisyphus agent (`OPENCODE_CONFIG` isolated) | total task time, exact tokens |
| `dsh` | DeepSeek Harness headless (`DSH_HOME` isolated) | total task time, est. tokens |

Agent harnesses (pi, sisyphus, dsh) do not stream, so TTFT is null and TPS is
output tokens over total task time. omp's built-in bench streams, so it reports
real TTFT + generation TPS.

## Residency model (two layers)
- **Resident:** oMLX + the two routing models (serves the gateway, normal
  usage, and the agent's own connection). Never unloaded by benchmarks.
- **Transient:** the benchmark framework's copy of the model-under-test,
  loaded for the cell and unloaded after. Keeps RAM bounded (~39 GB resident +
  ~17–21 GB transient on the M5 Max 128 GB).

Benchmarks run **strictly sequentially** (one cell at a time) for RAM and
measurement integrity. The gateway refuses a second lab job while one is active.

## Metrics
- **raw / omp:** TTFT (time to first token), total time, output tokens (exact,
  from `usage`), decode TPS = tokens / (total − TTFT).
- **pi / sisyphus / dsh:** total task time, output tokens (sisyphus exact from
  `step_finish.tokens`; pi/dsh estimated ~4 chars/token), TPS = tokens / total.

A `warmup` (default 1) uncounted iteration absorbs server/model warmup so the
first measured iteration is not penalised.

## Dashboard
The **3-Axis Lab** tab (gateway `http://127.0.0.1:9000/`) has model /
framework / harness dropdowns, iteration + max-token inputs, a Run button, a
current-job panel, and a results table. Endpoints:
- `GET /lab/options` — available models/frameworks/harnesses
- `POST /lab/run` — queue a cell `{framework, model, harness, iterations, max_tokens, prompt}`
- `GET /lab/status/{job_id}` — job status + result
- `GET /lab/results` — jobs + history
- `POST /lab/clear` — clear history

## CLI
```bash
.venv/bin/python benchmark_lab.py options
.venv/bin/python benchmark_lab.py run --framework omlx --model qwen3.6-35b-a3b \
    --harness raw --iterations 3 --max-tokens 128
```

## Setup (one-time, already done on this machine)
- `brew install ollama` (0.33.x), `brew install llama.cpp` (0.4.x)
- `brew install can1357/tap/omp` (omp), `npm install -g opencode-ai` (opencode)
- dsh via `npx -y @deepseek-ai/dsh`
- oh-my-opencode plugin: `npx -y oh-my-opencode install --no-tui ...`
  (registers `oh-my-openagent@latest` for opencode's Sisyphus agent)
- MLX in `~/.omlx/bench-venv`
- GGUFs downloaded to `models/gguf/` (Q6_K from unsloth repos)

## Model-change procedure
1. Edit `MODELS` in `benchmark_lab.py` (paths + request ids).
2. If the routing models change, edit `router.yaml` (see task-6-backend.md).
3. Restart the gateway (`scripts/stack.py restart`) — routing/backends read at
   startup.
