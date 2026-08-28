# Task 1: CI/CD Pipeline with GitHub Actions

## Overview

Set up automated testing, linting, and deployment on every push to GitHub. This ensures code quality is maintained and regressions are caught early.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                                │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  push /  │───▶│  GitHub      │───▶│  CI Pipeline │───▶│  Deploy  │  │
│  │  PR      │    │  Actions     │    │  (macOS)     │    │  (opt)   │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│                              │                    │                     │
│                              ▼                    ▼                     │
│                     ┌─────────────────┐  ┌─────────────────┐           │
│                     │  Lint + Type    │  │  Unit Tests     │           │
│                     │  Check          │  │  (pytest)       │           │
│                     │  (ruff, mypy)   │  │                 │           │
│                     └─────────────────┘  └─────────────────┘           │
│                              │                    │                     │
│                              ▼                    ▼                     │
│                     ┌─────────────────┐  ┌─────────────────┐           │
│                     │  Code Quality   │  │  Integration    │           │
│                     │  Gate           │  │  Tests (opt)    │           │
│                     └─────────────────┘  └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Agentic Pipeline

This task is executed by a team of agents working in parallel:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agentic Development Pipeline                        │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  Developer  │───▶│  Tester     │───▶│  Engineer   │                │
│  │  Agent      │    │  Agent      │    │  Agent      │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        ▼                    ▼                    ▼                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  Write CI   │    │  Run tests  │    │  Deploy &   │                │
│  │  workflow   │    │  locally    │    │  verify     │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        └────────────────────┴────────────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  QA Agent       │                                │
│                     │  (final check)  │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Write CI workflow, test files, lint config | `write`, `edit` |
| **Tester** | Run tests locally, verify they pass | `bash` (pytest, ruff, mypy) |
| **Engineer** | Push to GitHub, verify CI runs | `bash` (git, curl) |
| **QA** | Final review, check edge cases | `read`, `bash` |

## Step-by-Step Execution

### Step 1: Create Test Directory Structure

```bash
mkdir -p tests/integration
touch tests/__init__.py
touch tests/conftest.py
touch tests/test_proxy.py
touch tests/test_routing.py
touch tests/test_cache.py
touch tests/test_memory.py
touch tests/test_benchmarks.py
touch tests/integration/test_e2e.py
```

### Step 2: Write Unit Tests

**tests/test_proxy.py** — Test core proxy logic:

```python
import pytest
from proxy import (
    cache_key, cache_get, cache_put,
    extract_usage, extract_finish_reason,
    stream_delta_text, normalize_policy,
    heuristic_policy, is_coding_request,
)

class TestCache:
    def test_cache_key_deterministic(self):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        key1 = cache_key(body)
        key2 = cache_key(body)
        assert key1 == key2

    def test_cache_key_different(self):
        body1 = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        body2 = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
        assert cache_key(body1) != cache_key(body2)

    def test_cache_put_get(self):
        # Test cache put and get
        ...

class TestPolicy:
    def test_normalize_policy_valid(self):
        policy = normalize_policy({"route": "moe", "confidence": 0.8})
        assert policy["route"] == "moe"
        assert 0.0 <= policy["confidence"] <= 1.0

    def test_normalize_policy_invalid_route(self):
        policy = normalize_policy({"route": "invalid"})
        assert policy["route"] in ("moe", "dense")

    def test_heuristic_policy_coding(self):
        body = {"messages": [{"role": "user", "content": "Build a Python API"}]}
        policy = heuristic_policy(body)
        assert policy["route"] in ("moe", "dense")
```

**tests/test_routing.py** — Test routing decisions:

```python
import pytest
from proxy import (
    choose_policy, arch_router_policy,
    is_explicit_continuation, task_state,
)

class TestRouting:
    def test_explicit_moe_selection(self):
        body = {"model": "gateway-moe", "messages": []}
        policy, _ = choose_policy(body)
        assert policy["route"] == "moe"

    def test_explicit_dense_selection(self):
        body = {"model": "gateway-dense", "messages": []}
        policy, _ = choose_policy(body)
        assert policy["route"] == "dense"

    def test_auto_routing(self):
        body = {"model": "auto", "messages": [{"role": "user", "content": "simple question"}]}
        policy, _ = choose_policy(body)
        assert policy["route"] in ("moe", "dense")

    def test_explicit_continuation(self):
        assert is_explicit_continuation("continue")
        assert is_explicit_continuation("go on")
        assert not is_explicit_continuation("new question")
```

**tests/test_cache.py** — Test cache behavior:

```python
import pytest
import time
from proxy import response_cache, cache_stats, CACHE_TTL_SEC

class TestCache:
    def setup_method(self):
        response_cache.clear()
        cache_stats.update({"hits": 0, "misses": 0, "stores": 0, "evictions": 0})

    def test_cache_miss(self):
        result = cache_get("nonexistent")
        assert result is None
        assert cache_stats["misses"] == 1

    def test_cache_hit(self):
        cache_put("test", {"data": "value"}, 200)
        result = cache_get("test")
        assert result is not None
        assert result["payload"] == {"data": "value"}
        assert cache_stats["hits"] == 1

    def test_cache_expiry(self):
        # Mock time to test expiry
        ...
```

**tests/test_memory.py** — Test memory guard:

```python
import pytest
from proxy import memory_state, update_memory_state, MEMORY_GUARD_GB

class TestMemory:
    def test_memory_state_update(self):
        update_memory_state()
        assert memory_state["available_gb"] > 0
        assert memory_state["total_gb"] > 0

    def test_memory_pressure_threshold(self):
        update_memory_state()
        if memory_state["available_gb"] <= MEMORY_GUARD_GB:
            assert memory_state["pressure"] is True
        else:
            assert memory_state["pressure"] is False
```

**tests/test_benchmarks.py** — Test benchmark evaluation:

```python
import pytest
from proxy import evaluate_benchmark_output, BENCHMARK_SUITES

class TestBenchmarks:
    def test_evaluate_quick_suite(self):
        response = "Here are 5 points:\n1. Point one\n2. Point two\n3. Point three\n4. Point four\n5. Point five\n\nMoE is fast, dense is quality."
        result = evaluate_benchmark_output("quick", response, "stop", 200, {"completion_tokens": 100})
        assert result["readiness_score"] > 0
        assert "completeness" in result["categories"]

    def test_evaluate_coding_suite(self):
        response = "<!DOCTYPE html><html><head></head><body><input id='tax'><input id='tip'><input id='people'><script>function calculate() { /* logic */ }</script></body></html>"
        result = evaluate_benchmark_output("coding_hitl", response, "stop", 200, {"completion_tokens": 500})
        assert result["readiness_score"] > 0
```

### Step 3: Configure Linting and Type Checking

**pyproject.toml** (add to existing or create new):

```toml
[project]
name = "inference-gateway"
version = "2.1.1"
description = "Local inference gateway for Apple Silicon"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "httpx>=0.28",
    "psutil>=6.0",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.3",
    "mypy>=1.8",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]  # Line length handled by formatter

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_ignores = true
```

### Step 4: Create GitHub Actions Workflow

**.github/workflows/ci.yml**:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: macos-latest  # Apple Silicon runner
    strategy:
      matrix:
        python-version: ["3.12"]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-asyncio ruff mypy
      
      - name: Lint with ruff
        run: ruff check .
      
      - name: Type check with mypy
        run: mypy proxy.py --ignore-missing-imports
      
      - name: Run tests
        run: pytest tests/ -v --tb=short
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-results/
```

### Step 5: Add Pre-commit Hooks

**.pre-commit-config.yaml**:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
```

**Setup:**
```bash
pip install pre-commit
pre-commit install
```

### Step 6: Verify Locally

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check .

# Run type checker
mypy proxy.py --ignore-missing-imports

# Run tests
pytest tests/ -v

# Run pre-commit
pre-commit run --all-files
```

### Step 7: Push and Verify CI

```bash
git add .
git commit -m "ci: add GitHub Actions pipeline with tests, linting, type checking"
git push origin main
```

**Verify:**
1. Go to GitHub → Actions tab
2. Wait for workflow to complete
3. Check all steps pass
4. Fix any failures and re-push

## Success Criteria

- [ ] `ruff check .` passes with no errors
- [ ] `mypy proxy.py` passes with no errors
- [ ] `pytest tests/` passes with 100% test coverage on core logic
- [ ] GitHub Actions workflow runs on every push/PR
- [ ] Pre-commit hooks installed and working
- [ ] CI badge added to README

## Commands Reference

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all checks
ruff check . && mypy proxy.py --ignore-missing-imports && pytest tests/ -v

# Run specific test file
pytest tests/test_routing.py -v

# Run specific test
pytest tests/test_routing.py::TestRouting::test_explicit_moe_selection -v

# Run with coverage
pytest tests/ --cov=proxy --cov-report=term-missing

# Format code
ruff format .

# Fix linting issues
ruff check . --fix

# Run pre-commit
pre-commit run --all-files
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ruff` not found | `pip install ruff` |
| `mypy` errors on imports | Add `--ignore-missing-imports` |
| Tests fail on macOS | Ensure Python 3.12+ |
| CI fails on `macos-latest` | Check runner availability, try `macos-14` |
| Pre-commit blocks commit | Run `pre-commit run --all-files` to fix |
