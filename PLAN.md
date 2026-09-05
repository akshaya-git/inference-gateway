# Inference Gateway — Development Plan

> **Updated 2026-09-05 (rev 2)** — Architecture changed: the judge model is **dropped entirely**
> (no longer in `router.yaml` or `proxy.py`), routing is now **deterministic** (score/level lookup),
> and the MoE workhorse is **Qwen3.6-35B-A3B 6-bit** (Kimi was tried and dropped — the current
> oMLX setup keeps only the dense model resident, leaving more memory headroom).
> All tasks are re-baselined below. Task 1 is ready to re-run.
> **Policy:** gateway code changes are handled by the dense model (`gateway-dense`).

## Architecture Change Log (2026-09-05)

| Area | Before | After |
|------|--------|-------|
| Routing | Arch-Router-1.5B judge model, semantic per-request inference | **Deterministic** `RoutingEngine` (`routing_logic.py` + versioned `routing_rules.json`): 6 complexity levels, code-term gate, base rules + modifiers; code above level 4 → dense, everything else → MoE |
| MoE workhorse | Qwen3.6-35B-A3B-oQ5e-mtp | **mlx-community--Qwen3.6-35B-A3B-6bit** (Kimi Linear 48B was tried, then dropped) |
| Dense specialist | Qwen3.8-27B-oQ4e-mtp | scottlowry--Qwen3.8-27B-oQ6e-mtp |
| Judge model | Loaded, reloaded if evicted, judged every auto request | **Dropped entirely** — `judge:` section removed from `router.yaml`; all `JUDGE_CONFIG`/`ROUTER_MODEL` code removed from `proxy.py`; can be unloaded in oMLX and is never reloaded |
| oMLX | Server on :8080, stack script swapped models | oMLX **app** on **:8000** (admin API used directly); `scripts/stack.py` lifecycle |
| Model residency | One large model resident; swap on route change | **`KEEP_MODELS_LOADED=true`** default — both large models stay loaded; route changes never unload the other (set `false` for old behavior; current setup runs dense-only for memory headroom) |
| Model switching | Subprocess call to `inference-stack.sh` | Gateway control API (`/control/model/{route}`) + oMLX admin load/unload with transition waits; fail-closed on admin errors |
| Context/output | 131,072 ctx / 32,144 out | 128,000 ctx / 32,000 out (advertised on all aliases) |
| Benchmark suites | 3 Pi-session suites (quick, coding_hitl, reasoning) | +5 artifact suites: **browser_tetris, svg_portrait, kanban_board, csv_dashboard, pathfinding_visualizer** (HTML/SVG artifacts, human verdicts) |
| TPS metric | Mean of per-call rates | **Time-weighted** average generation TPS (tokens / summed generation ms) |
| Rules management | n/a | `GET/PUT /routing/rules` (validated, versioned, atomic); human verdicts journal to `.inference-stack/routing-feedback.jsonl` |
| Pi integration | `local-mlx` provider | `mlx-proxy` provider (`docs/pi-models.json`, `docs/pi-settings.json`); aliases `gateway-auto` / `gateway-moe` / `gateway-dense` |
| Auth | none | Optional `OMLX_API_KEY` bearer header for all upstream calls |

### Deterministic routing (replaces the judge)

- `routing_rules.json` (versioned lookup): `code_terms`, `base_rules` (level 1–6), `modifiers`
  (max 2 counted, level capped at 6), `dense_above: 4`.
- Decision = explicit alias > `[model:…]` control > `[complexity:N]` control > level lookup.
  Controls are stripped before the request is forwarded.
- Decisions are sticky per user-message history (survives tool turns and rule reloads);
  bounded 512-task memory; "continue"-style messages retain the prior decision.
- Invalid rule edits keep the last valid lookup; error surfaced in `/metrics`.
- **Model changes are a supported pipeline function**: route → model mapping lives only in
  `router.yaml` + env overrides (`MOE_MODEL`, `DENSE_MODEL`); see Task 9.

## Task Dependency Chart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Task Dependency Graph                                 │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  Task 1  │    │  Task 2  │    │  Task 3  │    │  Task 4  │         │
│  │  CI/CD   │    │  Testing │    │  Rationale│   │  Benchmarks│        │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│       │               │               │               │                │
│       ▼               ▼               ▼               ▼                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  Task 5  │    │  Task 6  │    │  Task 7  │    │  Task 9  │         │
│  │  Cache   │    │  Backend │    │  Refine  │    │  Model   │         │
│  └──────────┘    └──────────┘    └──────────┘    │  Swappab.│         │
│       │               │               │          └──────────┘         │
│       │               │               │               │                │
│       └───────────────┴───────────────┴───────────────┘                │
│                               │                                         │
│                               ▼                                         │
│                      ┌─────────────────┐                               │
│                      │  Task 8: Release│                               │
│                      └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Dependency Matrix

| Task | Depends On | Blocks | Can Start |
|------|-----------|--------|-----------|
| Task 1: CI/CD | None | Task 8 | Immediately |
| Task 2: Testing | None | Task 6, Task 8, Task 9 | Immediately |
| Task 3: Rationale | None | Task 7 | Immediately |
| Task 4: Benchmarks | None | Task 7, Task 8 | Immediately |
| Task 5: Cache | None | Task 8 | Immediately |
| Task 6: Backend | Task 2 | Task 8 | After Task 2 |
| Task 7: Refinement | Task 3, Task 4 | Task 8 | After Task 3 & 4 |
| Task 9: Model Swappability | Task 2 | Task 8 | After Task 2 |
| Task 8: Release | All tasks | None | After all complete |

## Task Summary

| # | Task | File | Effort | Status |
|---|------|------|--------|--------|
| 1 | CI/CD Pipeline (re-run on new code) | [docs/task-1-cicd.md](docs/task-1-cicd.md) | 1-2 days | 🔄 **Re-run needed** — CI must cover `routing_logic.py`, `routing_rules.json`, `scripts/stack.py`, 114 unit tests |
| 2 | Testing Environment | [docs/task-2-testing.md](docs/task-2-testing.md) | 1-2 days | 🔄 **Re-baseline** — new models (Qwen3.6-35B-A3B-6bit / Qwen3.8-27B-oQ6e), oMLX app :8000, M5 Max 128 GB; 10 integration tests + benchmarks to re-run |
| 3 | Routing Rationale | [docs/task-3-rationale.md](docs/task-3-rationale.md) | 1 day | 🔄 **Redesigned** — no judge raw output anymore; capture deterministic decision (level, matched rules, source, rules_version) + feedback journal; export endpoint |
| 4 | Benchmark Suites | [docs/task-4-benchmarks.md](docs/task-4-benchmarks.md) | 2-3 days | ✅ **Core done** — 5 new artifact suites implemented (tetris, svg portrait, kanban, csv dashboard, pathfinding) + SVG artifact endpoint + time-weighted TPS; remaining: run on live stack, human verdicts |
| 5 | Cache Enhancement | [docs/task-5-cache.md](docs/task-5-cache.md) | 1-2 days | ⬜ Not started — cache still exact-match only |
| 6 | Backend Abstraction | [docs/task-6-backend.md](docs/task-6-backend.md) | 2-3 days | 🔄 **Partially achieved** — oMLX admin API used directly, `OMLX_API_KEY`, config-driven model IDs; still no `BackendInterface` ABC/adapters |
| 7 | Instruction Refinement | [docs/task-7-refinement.md](docs/task-7-refinement.md) | 1-2 days | 🔄 **Redesigned** — no judge prompt to refine; now refine `routing_rules.json` (terms/levels) from benchmark verdicts via `RoutingEngine.update_rules()` / `PUT /routing/rules` |
| 9 | Model Swappability | (new — see below) | 1 day | ⬜ Not started — make model changes a first-class, tested, documented pipeline function |
| 8 | Open Source Release | [docs/task-8-release.md](docs/task-8-release.md) | 1-2 days | ⬜ Not started — depends on all of the above |

**Total: 9-15 days**

### Task 9: Model Swappability (new)

The MoE model already changed Qwen3.6-oQ5e → Kimi → Qwen3.6-6bit and will change again. Model
changes must be a **supported function of the pipeline**, not a code edit:

1. Route → model mapping only in `router.yaml` (+ `MOE_MODEL`/`DENSE_MODEL` env overrides); no
   model IDs hardcoded in logic, tests, or docs that drive behavior.
2. Documented change procedure: update `router.yaml` → restart gateway (or hot path via admin
   API) → verify with `/health`, `/metrics`, `scripts/stack.py status`.
3. Tests: alias/route resolution follows config; a config change with no code change routes to
   the new model (mock oMLX).
4. README "Changing models" section; release notes template includes model changes.

## Agentic Pipeline Overview

Each task is executed by a team of agents working in a pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agentic Development Pipeline                        │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  Developer  │───▶│  Tester     │───▶│  Engineer   │───▶┌────────┐ │
│  │  Agent      │    │  Agent      │    │  Agent      │    │  QA    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    │  Agent │ │
│        │                    │                    │          └────────┘ │
│        ▼                    ▼                    ▼               │     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │     │
│  │  Write code │    │  Run tests  │    │  Deploy &   │          │     │
│  │  + docs     │    │  locally    │    │  verify     │          │     │
│  └─────────────┘    └─────────────┘    └─────────────┘          │     │
│        │                    │                    │               │     │
│        └────────────────────┴────────────────────┘───────────────┘     │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  Final Review   │                                │
│                     │  + Merge        │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Write code, create docs, implement features | `write`, `edit` |
| **Tester** | Run tests, verify behavior, check edge cases | `bash` (pytest, curl) |
| **Engineer** | Deploy, verify integration, check CI | `bash` (git, curl) |
| **QA** | Final review, check for regressions, approve | `read`, `bash` |

## Execution Order

### Phase 1: Re-baseline (Week 1)
1. **Task 1: CI/CD (re-run)** — CI green on the new deterministic-routing codebase (114 unit tests)
2. **Task 2: Testing** — start oMLX app + gateway on this machine (dense model resident; more
   memory headroom than the two-large-model setup), run 10 integration tests, regenerate M5 Max
   baselines with Qwen3.6-35B-A3B-6bit + Qwen3.8-27B-oQ6e

### Phase 2: Core Features (Week 2)
3. **Task 3: Rationale** — deterministic decision capture + feedback journal export
4. **Task 4: Benchmarks** — run the 5 new artifact suites on the live stack, collect human verdicts
5. **Task 5: Cache** — semantic dedup + analytics

### Phase 3: Advanced Features (Week 3)
6. **Task 6: Backend** — `BackendInterface` ABC + adapters
7. **Task 7: Refinement** — rules refinement loop from verdicts
8. **Task 9: Model Swappability** — config-driven model changes, tested + documented

### Phase 4: Release (Week 4)
9. **Task 8: Release** — prepare for open source

## Notes

- Tasks 1, 2, 3, 4, 5 are independent and can be worked on in parallel
- Task 6 and Task 9 require Task 2 (live stack) for validation
- Task 7 requires Task 3 (decision data) and Task 4 (benchmark verdicts)
- Task 8 requires all other tasks to be complete
- Each task file is self-contained with architecture diagrams, step-by-step instructions, and
  agentic pipeline explanations
- **Restart the proxy after code upgrades** — routing behavior and model residency are read at
  startup; the old judge can be unloaded in oMLX and will not be reloaded
- **Gateway code changes are routed to the dense model** (`gateway-dense`) — keep this in mind
  when the gateway is used to develop the gateway itself
- Known cleanup candidates (dead code from the judge removal): `router_request_context()`,
  `last_user_text()` in proxy.py; `judge_model: None` metrics field
