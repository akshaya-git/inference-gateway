# Inference Gateway — Development Plan

> **Updated 2026-09-05 (rev 3)** — Checkpointed execution plan. Judge model dropped entirely;
> deterministic score/level routing; MoE workhorse = Qwen3.6-35B-A3B-6bit; dense = Qwen3.8-27B-oQ6e.
> Tasks run **strictly in sequence**, each ending in a checkpoint (commit + push + CI green).
> **Policy:** gateway code changes are handled by the dense model (`gateway-dense`).

## Architecture Change Log (2026-09-05)

| Area | Before | After |
|------|--------|-------|
| Routing | Arch-Router-1.5B judge model, semantic per-request inference | **Deterministic** `RoutingEngine` (`routing_logic.py` + versioned `routing_rules.json`): 6 complexity levels, code-term gate, base rules + modifiers; code above level 4 → dense, everything else → MoE |
| MoE workhorse | Qwen3.6-35B-A3B-oQ5e-mtp | **mlx-community--Qwen3.6-35B-A3B-6bit** (Kimi was tried, then dropped) |
| Dense specialist | Qwen3.8-27B-oQ4e-mtp | scottlowry--Qwen3.8-27B-oQ6e-mtp |
| Judge model | Loaded, reloaded if evicted, judged every auto request | **Dropped entirely** — no `judge:` section, no judge code; unloadable in oMLX, never reloaded |
| oMLX | Server on :8080, stack script swapped models | oMLX **app** on **:8000** (OpenAI-compatible + admin API); `scripts/stack.py` lifecycle |
| Model residency | One large model resident; swap on route change | **`KEEP_MODELS_LOADED=true`** default — both large models stay loaded; `false` restores swap behavior |
| Model switching | Subprocess call to `inference-stack.sh` | Gateway control API + oMLX admin load/unload with transition waits; fail-closed |
| Context/output | 131,072 ctx / 32,144 out | 128,000 ctx / 32,000 out |
| Benchmark suites | 3 Pi-session suites | +5 artifact suites (tetris, svg portrait, kanban, csv dashboard, pathfinding) + SVG endpoint + time-weighted TPS |
| Rules management | n/a | `GET/PUT /routing/rules` (validated, versioned, atomic); verdicts journal `.inference-stack/routing-feedback.jsonl` |
| Pi integration | `local-mlx` provider | `mlx-proxy` provider; aliases `gateway-auto` / `gateway-moe` / `gateway-dense` |
| Auth | none | Optional `OMLX_API_KEY` bearer header |

### Route naming (why results say "moe"/"dense")

`moe` and `dense` are **route names** — stable labels for the two backends — not model names.
`moe` = the MoE-architecture workhorse (currently Qwen3.6-35B-A3B-6bit, 3B active params);
`dense` = the dense-architecture specialist (currently Qwen3.8-27B-oQ6e). Names survive model
swaps; a request shows "moe" when its computed level ≤ `dense_above` (default 4).

## Known Routing Gaps (verified 2026-09-05, fixed in CP-2)

Live test of the current rules showed dense is almost unreachable:

| Prompt | Current | Problem |
|---|---|---|
| "Add authentication with JWT to my flask app" | L1, moe | code gate vetoed the level-5 "authentication" rule |
| "Fix the race condition in my worker pool" | L1, moe | "race conditions" (plural) ≠ "race condition"; no code term |
| "Redesign the system to handle 10x traffic" | L1, moe | "system redesign" ≠ "redesign the system" |
| "Migrate my database from sqlite to postgres" | L3, moe | "database migration" ≠ "migrate my database" |
| "Build a todo list app" | L1, moe | "app" not a code term |
| "Review my pull request for security issues" | L1, moe | "security repair" ≠ "security issues" |
| "What is the function of the liver?" | L3, code=True | "function" false-positives as code |
| "[complexity:7] write code" | control ignored, raw tag forwarded | invalid controls not handled |

**Fixes (R1–R8), scoped per owner decisions:**
- R1: expand `code_terms` (~60-100 real terms: languages, frameworks, artifacts, task verbs).
  **Excluded for now:** authentication terms (revisit later), postgres (sqlite is fine — `sqlite`
  is a code term).
- R2: robust matching — multiple phrase variants per rule + stemmed token-set matching
  ("migrat"+"databas" → migration rule).
- R3: decouple gate from rules — base rules match even without the code gate; the gate sets the
  *default* level, it does not veto rules.
- R4: `dense_above` + level→route map configurable in `routing_rules.json` (drop `==4` hardcode).
- R5: capability metadata in `router.yaml` (context window, vision, max output per model);
  image requests route to vision-capable models; context-window check.
- R6: bug fixes — `ROUTING_ENABLED=false` pins to `fallback_route` (or remove flag); invalid
  controls (`[complexity:7]`) are **stripped + warned** (surfaced in reason/metrics), never
  forwarded raw and never clamped; `strip_routing_controls` strips all text parts; dashboard
  shows "Level N · matched rules" instead of fake 100% confidence.
- R7: remove benchmark vocabulary (tetris/kanban/dashboard/visualizer) from production rules.
- R8: the 12 prompts above become regression tests in `test_routing_logic.py`.
  Expected after fixes: "review my pull request for security issues" → L5-6 dense (security
  terms + variants); "[complexity:7] write code" → control stripped, warned, scored normally.

## Checkpointed Task Sequence

All tasks run **in sequence**. Each checkpoint = work complete + tests pass + commit + push +
CI green. No task starts until the previous checkpoint is closed.

| CP | Task | Scope | Entry | Exit (checkpoint) |
|----|------|-------|-------|-------------------|
| **CP-1** ✅ | **Task 1: CI/CD re-run** | `ci.yml` covers `routing_logic.py`, `routing_rules.json` (validation test), `scripts/stack.py`; ruff clean; unit tests in CI; pre-commit verified | Baseline `d762373` pushed | **Closed**: ruff fixes + Python 3.12/3.13/3.14 matrix; CI green on `c500024` |
| **CP-2** ✅ | **Task 3 rework: routing fixes + rationale** | R1–R8 above; deterministic decision fields (level, matched rules, source, rules_version) in `Metric` + dashboard + `/api/routing-rationale` export | CP-1 closed | **Closed**: rules v2, 12-prompt regression suite, dead code removed, CI green (see CP-2 log below) |
| **CP-3** | **Task 2: integration re-baseline** | Live stack (oMLX running with both models), 10 integration tests, `benchmark_real.py`, M5 Max 128 GB baselines with Qwen3.6-6bit + Qwen3.8-oQ6e | CP-2 closed (fixed routing before generating data) | Integration tests pass, new baselines committed |
| **CP-4** | **Task 6 + Task 9: backend abstraction + model swappability** | `BackendInterface` ABC; `OMLXBackend` (OpenAI-compatible chat + admin residency); `OpenAIBackend` (generic — covers MLX `mlx_lm.server`, llama.cpp `llama-server`, Ollama, LM Studio); `backends:` config in `router.yaml`; documented model-change procedure + tests (config change, no code change → routes to new model) | CP-3 closed (live stack for validation) | Behavior unchanged via interface; model swap tested; CI green |
| **CP-5** | **Task 10 (new): multi-axis benchmark lab** | Model dropdown (all backend models, **load-run-unload** residency to preserve RAM); framework dropdown (backends from config); harness dropdown (installed detection) + adapters; 3-axis results (model × framework × harness) with backend/harness columns | CP-4 closed | Any model × framework × installed harness runnable; RAM preserved; CI green |
| **CP-6** | **Task 5: cache enhancement** | Semantic dedup + warming + `/api/cache/analytics` + dashboard | CP-5 closed | Tests pass, CI green |
| **CP-7** | **Task 7: refinement loop** | Refine `routing_rules.json` (terms/levels) from benchmark verdicts via `update_rules()` / `PUT /routing/rules`; analyzer + generator + endpoints | CP-6 closed (needs CP-2 decision data + CP-5 verdicts) | Loop demonstrated end-to-end, CI green |
| **CP-8** | **Task 8: open source release** | MIT LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, packaging-ready `pyproject.toml`, issue/PR templates, README polish, release checklist | CP-7 closed | Release candidate tagged |

**Total: ~3-4 weeks sequential.**

### CP-5 detail: frameworks and harnesses

**Frameworks (backends):**

| Framework | Install | Adapter | Notes |
|-----------|---------|---------|-------|
| oMLX | already installed/running | `OMLXBackend` (OpenAI-compatible + admin load/unload) | jundot/omlx; only backend with gateway-managed residency |
| Ollama | `brew install ollama` | `OpenAIBackend` | lowest-friction second backend; auto model lifecycle |
| llama.cpp | `brew install llama-brew` | `OpenAIBackend` | GGUF quantization axis |
| MLX | `pip install mlx-lm` | `OpenAIBackend` | raw MLX baseline (measures oMLX overhead) |

**Harnesses (agents):**

| Harness | Repo | Install | Adapter notes |
|---------|------|---------|---------------|
| Pi | (installed) | — | existing `run_pi_benchmark` |
| none (raw) | — | — | **control**: direct chat completion, no agent — isolates model quality from harness quality |
| omp (oh-my-pi) | can1357/oh-my-pi | `brew install can1357/tap/omp` | Pi fork — adapter closely mirrors the Pi adapter (same provider config style) |
| deepseek-harness (`dsh`) | deepseek-ai/deepseek-harness | `npx @deepseek-ai/dsh` (Node.js) | developer preview, breaking changes; Web UI on :3080 — headless/CLI mode must be investigated first |
| oh-my-opencode (Sisyphus) | rooftop-Owl/oh-my-opencode (upstream: code-yeongyu/oh-my-opencode) | opencode CLI + plugin | plugin layer for opencode; install opencode first, then the plugin |

Dropped: Cline, OpenHands (owner decision). "bionic" — dropped unless owner revives it.

**Residency & concurrency model (owner-confirmed 2026-09-05):**
- **Resident layer (always on):** oMLX on :8000 with the two routing models. Serves the gateway,
  normal usage, and the agent's own direct connection (`omlx-direct` provider → :8000, so the agent
  survives gateway restarts). `stack.py restart` never kills the oMLX app.
- **Transient layer (per benchmark job):** the benchmark backend loads the model-under-test, runs,
  unloads (load-run-unload). If the model-under-test is also resident in oMLX, that is a temporary
  second copy in unified memory (~17-21 GB extra — safe on 128 GB); different models (e.g. GGUF in
  Ollama) cause no duplication.
- **Benchmark runs are strictly sequential** — for RAM *and* measurement integrity (concurrent runs
  contend for CPU/GPU/bandwidth and corrupt the numbers). Operational rule: no benchmarks while
  actively using the gateway.
- Backend lifecycles: oMLX always running; mlx-lm/llama.cpp started per job and stopped after;
  Ollama daemon with on-demand loading. The gateway's backend abstraction orchestrates:
  start backend → load model → run harness → unload → record.

**Benchmark job schema:** `{suite, backend, harness, models: [...], residency: load_run_unload|keep, prompt}`.
Harness talks **directly to the selected framework endpoint** (gateway orchestrates + measures,
does not sit in the inference path, so routing can't interfere with explicit model selection).
Load-run-unload is the default for benchmark models; the two routing models stay resident.

**Installation policy:** the agent installs and configures frameworks/harnesses as needed
(brew/npm/pip/npx), verifies each with a smoke run against the local endpoint before enabling it
in the dropdown, and records the exact commands in the task doc for reproducibility. Anything
requiring GUI interaction or credentials is handed to the owner as explicit commands.

## Task Reference

| # | Task | File | Status |
|---|------|------|--------|
| 1 | CI/CD Pipeline | [docs/task-1-cicd.md](docs/task-1-cicd.md) | ✅ closed in CP-1 (`c500024`, CI green) |
| 2 | Testing Environment | [docs/task-2-testing.md](docs/task-2-testing.md) | 🔄 re-baseline in CP-3 |
| 3 | Routing Rationale | [docs/task-3-rationale.md](docs/task-3-rationale.md) | ✅ closed in CP-2 (rules v2, R1–R8, rationale capture) |
| 4 | Benchmark Suites | [docs/task-4-benchmarks.md](docs/task-4-benchmarks.md) | ✅ core done (5 artifact suites); live runs in CP-3/CP-5 |
| 5 | Cache Enhancement | [docs/task-5-cache.md](docs/task-5-cache.md) | ⬜ CP-6 |
| 6 | Backend Abstraction | [docs/task-6-backend.md](docs/task-6-backend.md) | 🔄 CP-4 (absorbs Task 9) |
| 7 | Instruction Refinement | [docs/task-7-refinement.md](docs/task-7-refinement.md) | 🔄 reworked in CP-7 (refine rules, not judge prompt) |
| 9 | Model Swappability | — | absorbed into Task 6 / CP-4 |
| 10 | Multi-axis Benchmark Lab | (new doc in CP-5) | ⬜ CP-5 |
| 8 | Open Source Release | [docs/task-8-release.md](docs/task-8-release.md) | ⬜ CP-8 |

## Notes

- **Restart the proxy after code upgrades** — routing behavior and model residency are read at startup
- **Gateway code changes are routed to the dense model** (`gateway-dense`)
- Dead judge-era code removed in CP-2: `router_request_context()`, `last_user_text()`,
  `message_text()`, `conversation_key()`, `routing_context()`, `task_routing_context()`,
  `task_state()`, `is_explicit_continuation()`, `is_coding_request()`,
  `enforce_capability_floors()`, `heuristic_policy()` in proxy.py; `judge_model: None`
  metrics field; `tests/test_routing.py` (tested only dead functions)
- Current environment: oMLX running on :8000 with both models loaded; M5 Max 128 GB;
  Python 3.14.7 in `.venv`; 113 unit tests passing (10 integration deselected)

## CP-2 completion log (2026-09-06)

- **Rules v2** (`routing_rules.json`): expanded `code_terms` (languages, frameworks,
  artifacts, technical verbs, infra); benchmark vocabulary removed (R7); no auth terms,
  no postgres terms (owner scope); `function` dropped from code terms (liver false
  positive); new base rules: `concurrency`, `data_migration`, `security`, `system`
  (L6), `integration` (L5), `stateful` (L4), `small` (L2), `mechanical` (L1).
- **Matching engine** (`routing_logic.py`): stemmed token-set containment with phrase
  variants (R2); code gate decoupled from rule matching (R3); `dense_above` configurable
  1-5 + optional `level_routes` map (R4); invalid controls stripped + reported, never
  clamped/forwarded (R6); `strip_routing_controls` strips all text parts.
- **Capability metadata** (R5): `capabilities` per route in `router.yaml` (context
  window, max output, vision); `capability_adjustment()` switches route for image
  content / oversized prompts when a capable model exists; gaps noted in reason.
- **Rationale capture**: `Metric.routing_level` + `Metric.routing_rules`; dashboard
  shows Level + reason (matched rules) instead of fake confidence; new
  `GET /api/routing-rationale` export endpoint; `judge_model: None` removed from metrics.
- **ROUTING_ENABLED** now actually pins to the fallback route (was reported-only).
- **12-prompt regression suite** in `tests/test_routing_logic.py` — all pass:
  race condition → L6 dense; system redesign → L6 dense; DB migration → L6 dense;
  security review → L6 dense; todo app → L3 moe; liver question → L1 moe (not code);
  `[complexity:7]` → stripped + reported; auth prompt → L3 moe (auth out of scope).
- **Documented behavior change**: "Build polished browser Tetris" is now L4 moe
  (was L5 dense via benchmark vocabulary). Benchmark runs use explicit aliases, so
  benchmark data is unaffected.
- Tests: 113 passed, 10 deselected (CI-equivalent). Ruff clean.
