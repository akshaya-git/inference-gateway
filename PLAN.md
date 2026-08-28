# Inference Gateway — Development Plan

## Current State

The gateway is a working Phase 2 system with:
- ✅ Judge-based routing (Arch-Router-1.5B)
- ✅ Memory protection with admission control
- ✅ Response cache (exact-match, non-streaming)
- ✅ 3 benchmark suites (quick, coding_hitl, reasoning)
- ✅ Real-time dashboard
- ✅ Stack control script

## Phase 3: Production-Ready Open Source

### 1. CI/CD Pipeline (GitHub Actions)

**Goal:** Automated testing, linting, and deployment on every push.

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest  # Apple Silicon runner
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
      - run: ruff check .
      - run: mypy proxy.py
```

**Deliverables:**
- [ ] `tests/` directory with unit + integration tests
- [ ] `pytest` test suite
- [ ] `ruff` linter config
- [ ] `mypy` type checking
- [ ] GitHub Actions workflow
- [ ] Pre-commit hooks

### 2. Controlled Testing Environment

**Goal:** Reproducible test environment that doesn't require actual models.

**Approach:**
- Mock oMLX server (FastAPI test app)
- Mock judge model responses
- Fixture-based test data
- Integration tests with real models (optional, tagged)

```
tests/
├── conftest.py          # Shared fixtures
├── mock_omlx.py         # Mock oMLX server
├── test_proxy.py        # Unit tests for proxy logic
├── test_routing.py      # Routing decision tests
├── test_cache.py        # Cache behavior tests
├── test_memory.py       # Memory guard tests
├── test_benchmarks.py   # Benchmark evaluation tests
└── integration/
    └── test_e2e.py      # End-to-end (requires real models)
```

### 3. Enhanced Cache Support

**Current:** Exact-match, non-streaming only, 128 entries, 300s TTL.

**Enhancements:**
- [ ] Semantic cache (embedding-based similarity)
- [ ] Streaming response caching (capture full stream, replay)
- [ ] Cache invalidation on model swap
- [ ] Cache metrics dashboard (hit rate, eviction rate)
- [ ] Configurable cache per route (moe vs dense)

### 4. Additional Benchmark Suites (5+ new)

**Goal:** Highlight differences and capabilities of the two models.

| Suite | Description | What It Tests |
|-------|-------------|---------------|
| `code_review` | Review a 500-line codebase for bugs | Deep code understanding |
| `refactoring` | Refactor a complex module | Multi-step editing |
| `debugging` | Find and fix a subtle concurrency bug | Root cause analysis |
| `architecture` | Design a distributed system | Systems reasoning |
| `security` | Identify and fix security vulnerabilities | Security awareness |
| `performance` | Optimize a slow algorithm | Performance tuning |
| `documentation` | Write comprehensive API docs | Technical writing |

Each suite:
- Has a static rubric for automated evaluation
- Captures full Pi agent session
- Measures: swap time, TTFT, latency, tokens, tool calls, errors
- Supports human verdict (pass/partial/fail)

### 5. Routing Rationale Capture

**Goal:** Understand *why* the judge chose a specific model.

**Implementation:**
```python
# In arch_router_policy(), capture the judge's reasoning
judge_response = {
    "route": "dense",
    "confidence": 0.85,
    "reason": "Complex multi-module refactoring requires deep code understanding",
    "task_type": "complex_analysis",
    "effort": "high",
    "thinking": True,
    "judge_raw": "The request involves refactoring a 500-line module...",
    "judge_tokens": 42,
    "judge_latency_ms": 1200,
}
```

**Storage:**
- Per-request: `route_reason`, `judge_raw`, `judge_latency_ms`
- Aggregated: routing accuracy metrics, false positive/negative tracking

**Dashboard:**
- "Routing Decisions" tab with expandable rationale
- Filter by route, confidence, task type
- Export to CSV for analysis

### 6. Instruction Refinement Loop

**Goal:** Incrementally improve routing accuracy based on outcomes.

**Approach:**
1. **Log outcomes** — track which route produced better results (human verdict, benchmark score)
2. **Analyze patterns** — identify task types where routing was suboptimal
3. **Update route descriptions** — refine `router.yaml` descriptions based on evidence
4. **A/B test** — run benchmarks with updated descriptions, compare accuracy
5. **Version control** — track description changes in git

```yaml
# router.yaml (versioned)
version: 3
routes:
  moe:
    description: >-
      Fast workhorse for structured output, tool use, data analysis,
      summarization, general reasoning, and routine coding.
      Best for: quick tasks, iterative agent work, moderate complexity.
      Avoid: complex refactoring, security audits, architecture design.
  dense:
    description: >-
      High-quality specialist for complex code review, architecture redesign,
      difficult debugging, and demanding code generation.
      Best for: multi-module refactoring, security audits, systems design.
      Avoid: simple transformations, quick summaries.
```

### 7. Generic Model Backend Support

**Goal:** Work with any model hosting framework, not just oMLX.

**Abstraction layer:**
```python
class ModelBackend(ABC):
    @abstractmethod
    async def load_model(self, model_id: str) -> None: ...
    @abstractmethod
    async def unload_model(self, model_id: str) -> None: ...
    @abstractmethod
    async def is_loaded(self, model_id: str) -> bool: ...
    @abstractmethod
    async def chat(self, model_id: str, messages: list) -> AsyncIterator: ...

class OMLXBackend(ModelBackend): ...
class MLXBackend(ModelBackend): ...
class LLAMABackend(ModelBackend): ...
class OpenAIBackend(ModelBackend): ...  # For cloud fallback
```

**Configuration:**
```yaml
backends:
  omlx:
    type: omlx
    url: http://127.0.0.1:8080
  mlx:
    type: mlx
    url: http://127.0.0.1:8081
  llama:
    type: llama
    url: http://127.0.0.1:8082
```

### 8. Open Source Release

**Deliverables:**
- [ ] `LICENSE` (MIT or Apache 2.0)
- [ ] `CONTRIBUTING.md`
- [ ] `CODE_OF_CONDUCT.md`
- [ ] `pyproject.toml` (proper packaging)
- [ ] `Makefile` (common tasks)
- [ ] `docker-compose.yml` (optional containerized setup)
- [ ] `docs/` directory with:
  - `installation.md`
  - `configuration.md`
  - `api-reference.md`
  - `benchmarks.md`
  - `troubleshooting.md`
- [ ] GitHub repository with:
  - Issues template
  - PR template
  - Release automation
  - Community discussion

## Execution Order

| Phase | Task | Effort | Priority |
|-------|------|--------|----------|
| 1 | CI/CD pipeline + tests | 2-3 days | High |
| 2 | Controlled testing environment | 1-2 days | High |
| 3 | Routing rationale capture | 1 day | High |
| 4 | 5+ new benchmark suites | 2-3 days | Medium |
| 5 | Enhanced cache support | 1-2 days | Medium |
| 6 | Generic backend abstraction | 2-3 days | Medium |
| 7 | Instruction refinement loop | 1-2 days | Low |
| 8 | Open source release | 1-2 days | High |

**Total estimated effort: 11-18 days**

## Success Metrics

- [ ] 100% test coverage on core routing logic
- [ ] 5+ new benchmark suites with automated evaluation
- [ ] Routing accuracy > 80% (measured by human verdict)
- [ ] Cache hit rate > 30% on repeat workloads
- [ ] Zero OOM crashes under sustained load
- [ ] Clean `ruff` + `mypy` pass
- [ ] Successful open-source release with community docs
