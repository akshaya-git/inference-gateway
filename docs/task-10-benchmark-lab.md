# Task 10 — Multi-axis benchmark lab

Measure one **(model × framework × harness)** cell at a time, with
**load → run → unload** residency so the benchmark framework's copy of the
model is transient on top of the always-resident oMLX routing models. When a
cell finishes, the gateway **drops the previous framework's model copy**, so
the next framework never runs with two model copies resident at once.

## Axes

### Models (logical)
| key | label | oMLX id | MTPLX | llama.cpp GGUF |
|-----|-------|---------|-------|----------------|
| `qwen3.8-27b` | Qwen3.8 27B (6-bit, **MTP**) | `scottlowry--Qwen3.8-27B-oQ6e-mtp` | `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality` | `Qwen3.8-27B-MTP-Q6_K.gguf` |

> **Dropped:** the MoE model (`qwen3.6-35b-a3b`) is no longer in the lab —
> the only model tested across all frameworks is **Qwen3.8 27B with MTP**.
> Ollama was dropped earlier (redundant with llama.cpp for GGUF models).

The registry in `benchmark_lab.py` (`MODELS`) maps each logical model to a
**per-framework artifact** (`artifacts[framework]`). A (model, framework) combo
is only valid if an artifact exists — the dashboard greys out the rest, and the
backend rejects them. Swapping a model = edit the registry (and `router.yaml`
for the routing models).

> **MTP per framework:** each framework uses its own MTP build (they support
> different formats). The dense model is MTP everywhere:
> - **oMLX**: `scottlowry/Qwen3.8-27B-oQ6e-mtp` (OptiQ 6-bit MTP, MLX-format,
>   native 256K context, `mtp_num_hidden_layers=1`).
> - **MTPLX**: `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality` (native MTP spec decoding).
> - **llama.cpp**: `Qwen3.8-27B-MTP-Q6_K.gguf` (Q6_K with MTP tensors).
>
> **MTP speculative decoding is active in all three:** oMLX (built-in),
> MTPLX (native), llama.cpp (`--spec-type draft-mtp`).
>
> **`mlx_lm.server` was dropped from the lab** because it cannot do MTP spec
> decoding: its `qwen3_5.py` model loader explicitly strips `mtp.*` weights
> (`weights = {k: v for k, v in weights.items() if "mtp." not in k}`), so it
> serves the base model only. The standalone MTP drafter repos
> (`mlx-community/Qwen3.8-27B-MTP-4bit`/`-8bit`, ~0.25–0.48 GB, model type
> `qwen3_5_mtp`) are **sidecars, not full models** (31 tensors, only
> `layers.0`), and mlx-lm 0.31.3 (latest) has no `qwen3_5_mtp` model file, so
> they can't be used as a `--draft-model` either.

### Context / output budget
Every framework loads the model with a **240K context window**
(`CONTEXT_WINDOW=245760`) and **32K max output tokens**
(`MAX_OUTPUT_TOKENS=32768`), MTP enabled:
- llama.cpp: `--ctx-size 245760 --n-predict 32768 --spec-type draft-mtp`
- MTPLX: `--context-window 245760 --max-tokens 32768`
- oMLX: `PUT /admin/api/models/{id}/settings` with
  `{"max_context_window": 245760, "max_tokens": 32768}` (applied in place by
  `FrameworkManager.ensure()`; safe on a loaded model — no unload)
- Harness provider configs (pi/omp `models.json`, opencode `limit`) use the
  same 240K/32K budget.

### Frameworks
| key | what | port | residency |
|-----|------|------|-----------|
| `omlx` | oMLX (resident, OpenAI-compatible) | 8000 | always resident; admin load/unload |
| `mtplx` | `mtplx serve` (native MTP speculative decoding) | 8400 | started per cell, stopped after |
| `llamacpp` | `llama-server` (brew) | 8300 | started per cell, killed after |

`FrameworkManager.ensure(framework, model)` returns `(endpoint, model_id)`;
`release()` tears the framework's copy down (the gateway drops the previous
model before the next framework loads its own). oMLX uses the admin API
(`/admin/api/models/{id}/load|unload`); the others are subprocess servers.
MTPLX serves OpenAI-compatible on :8400 (`mtplx serve --model <id> --no-auth`).

### Harnesses
| key | what | metrics |
|-----|------|---------|
| `pi` | pi coding agent, **full agent (tools enabled)**, `--mode json` event stream, isolated work dir | TTFT (first streamed delta), total, **exact tokens** (usage), **agent iterations** (turns), decode TPS |
| `omp` | omp built-in `bench` (`PI_CODING_AGENT_DIR` isolated) | TTFT, total, exact tokens, generation TPS (agent iters = 1 per request) |
| `opencode` | opencode + Sisyphus agent (`OPENCODE_CONFIG` isolated, isolated work dir) | total, **exact tokens** (per-step usage), **agent iterations** (steps); TTFT not measurable (completed parts only) |

> **Dropped:** `raw` (direct HTTP baseline), `bionic` (LM Studio GUI),
> `dsh` (deepseek harness). The former `sisyphus` harness is now named
> **`opencode`**.

**pi** runs as a full agent (tools enabled) in an isolated work dir
(`.inference-stack/bench/work/<fw>-<model>-pi/pi-work`), so long-running
tasks (file creation, multi-step work) are exercised. Metrics come from the
`--mode json` event stream, parsed live: `turn_end` events = agent
iterations, per-assistant `message_end` usage = exact output tokens, first
streamed delta = TTFT.

**opencode** runs the Sisyphus agent in an isolated work dir
(`.inference-stack/bench/work/<fw>-<model>-opencode/opencode-work`), so any
files the agent creates (e.g. an HTML page) land there, not in the project.
The work dir is reported in the result (`work_dir`) and shown in the
dashboard. Agent iterations = `step_finish` events; tokens = per-step
`tokens.output`.

**No timeout limits:** benchmarks run long-running tasks, so no harness
subprocess has a timeout — cells run to completion.

### Live progress
Every lab job reports its stage live via a progress callback (the dashboard
polls `/lab/status/{id}` every 1.5 s). Stages: `starting <fw> server` →
`server ready` → `warmup` → `<harness> iteration N/M` → `releasing server` →
`complete`. The stage timestamp (`stage_at`) lets the UI show "updated Ns ago".

## Residency model (two layers)
- **Resident:** oMLX + the routing models (serves the gateway, normal
  usage, and the agent's own connection). Never unloaded by benchmarks.
- **Transient:** the benchmark framework's copy of the model-under-test,
  loaded for the cell and dropped after. Keeps RAM bounded so only one
  transient model copy exists at a time.

Benchmarks run **strictly sequentially** (one cell at a time) for RAM and
measurement integrity. The gateway refuses a second lab job while one is active.

## Metrics
Per cell the result includes:
- **TTFT** — pi: first streamed delta (wall clock); omp: per-request
  (provider-measured); opencode: not measurable (—).
- **Total time** — full task time per iteration (agent harnesses include all
  agent steps and tool execution).
- **Output tokens** — exact, from provider usage (pi `message_end` usage,
  omp bench output, opencode `step_finish.tokens`).
- **TPS** — tokens / (total − TTFT) when TTFT is available, else
  tokens / total.
- **Agent iterations** (`avg_agent_iters`) — how many agent turns/steps the
  harness performed: pi `turn_end` events, omp 1 per request, opencode
  `step_finish` events.

A `warmup` (default 1) uncounted iteration absorbs server/model warmup so the
first measured iteration is not penalised.

## Dashboard
The **3-Axis Lab** tab (gateway `http://127.0.0.1:9000/`) has model /
framework / harness dropdowns, iteration + max-token inputs (max tokens
defaults to 32768), a Run button, a current-job panel, and a results table
(including an **Agent iters** column). Endpoints:
- `GET /lab/options` — available models/frameworks/harnesses
- `POST /lab/run` — queue a cell `{framework, model, harness, iterations, max_tokens, prompt}`
- `GET /lab/status/{job_id}` — job status + result
- `GET /lab/results` — jobs + history
- `POST /lab/clear` — clear history

## CLI
```bash
.venv/bin/python benchmark_lab.py options
.venv/bin/python benchmark_lab.py run --framework omlx --model qwen3.8-27b \
    --harness pi --iterations 3 --max-tokens 32768
```

## Setup (one-time, already done on this machine)
- `brew install llama.cpp` (0.4.x)
- `brew install can1357/tap/omp` (omp), `npm install -g opencode-ai` (opencode)
- oh-my-opencode plugin: `npx -y oh-my-opencode install --no-tui ...`
  (registers `oh-my-openagent@latest` for opencode's Sisyphus agent)
- GGUFs downloaded to `models/gguf/` (Q6_K MTP build)

## Model-change procedure
1. Edit `MODELS` in `benchmark_lab.py` (paths + request ids).
2. If the routing models change, edit `router.yaml` (see task-6-backend.md).
3. Restart the gateway (`scripts/stack.py restart`) — routing/backends read at
   startup.

## Status

- [x] Frameworks installed + smoke-tested: oMLX, **MTPLX**, llama.cpp
      (MLX and Ollama dropped)
- [x] Model: **Qwen3.8 27B MTP across all 3 frameworks** (MoE dropped from lab)
- [x] Harnesses: **pi, omp, opencode** (raw, bionic, dsh dropped)
- [x] 240K context / 32K max tokens / MTP enabled on every framework
- [x] No timeout limits (long-running tasks)
- [x] Agent iterations + total time captured per harness
- [x] Gateway drops the previous framework's model copy between cells
- [x] Dashboard **3-Axis Lab** tab — dropdowns with grey-out, prompt window,
      job status, results table with Agent iters column
- [x] Unit tests green (137)
- [ ] Owner: run the full matrix from the dashboard (manual, sequential)

### Smoke results (Qwen3.8 27B MTP, omp harness, 256-token runs, 2026-09-07)
| framework | MTP spec decoding | TTFT (ms) | total (ms) | TPS |
|-----------|-------------------|-----------|------------|-----|
| oMLX | built-in | — (agent cell) | — | — |
| **MTPLX** | native | 3699.1 | 12615.3 | **28.71** |
| llama.cpp | `--spec-type draft-mtp` | 942.4 | 17796.7 | 15.19 |

Agent-harness smoke (oMLX, 1 iteration): pi — TTFT 5680.7 ms, total 17159.2 ms,
17 tokens, 1 agent iter; opencode — total 32780.1 ms, 189 tokens, 1 agent iter.
