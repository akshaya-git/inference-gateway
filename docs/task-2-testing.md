# Task 2: Controlled Testing Environment

## Overview

Create a reproducible test environment that doesn't require actual models. This allows developers to test the gateway logic without needing 64GB of RAM or actual MLX models loaded.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Controlled Testing Environment                        │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │  Test Client    │───▶│  Gateway (FastAPI) │──▶│  Mock oMLX     │    │
│  │  (pytest)       │    │  :9000           │    │  Server         │    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│         │                        │                        │             │
│         │                        ▼                        │             │
│         │               ┌─────────────────┐               │             │
│         │               │  Mock Judge     │               │             │
│         │               │  Model          │               │             │
│         │               │  (returns       │               │             │
│         │               │   fixed routes) │               │             │
│         │               └─────────────────┘               │             │
│         │                                                 │             │
│         └─────────────────────────────────────────────────┘             │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  Test Fixtures  │                                │
│                     │  (sample data)  │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Agentic Pipeline

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
│  │  Write mock │    │  Run tests  │    │  Verify CI  │                │
│  │  server +   │    │  with mocks │    │  passes     │                │
│  │  fixtures   │    │             │    │             │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        └────────────────────┴────────────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  QA Agent       │                                │
│                     │  (edge cases)   │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Write mock server, fixtures, test config | `write`, `edit` |
| **Tester** | Run tests with mocks, verify behavior | `bash` (pytest) |
| **Engineer** | Integrate with CI, verify no real models needed | `bash` (git, curl) |
| **QA** | Test edge cases, verify mock fidelity | `read`, `bash` |

## Step-by-Step Execution

### Step 1: Create Mock oMLX Server

**tests/mock_omlx.py**:

```python
"""
Mock oMLX server for testing.
Simulates the oMLX admin API and model endpoints without requiring actual models.
"""
import asyncio
import json
import time
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

app = FastAPI()

# Mock model states
model_states = {
    "Arch-Router-1.5B-mlx-8Bit": {"loaded": True, "is_loading": False},
    "Qwen3.6-35B-A3B-oQ5e-mtp": {"loaded": True, "is_loading": False},
    "Qwen3.8-27B-oQ4e-mtp": {"loaded": False, "is_loading": False},
}

# Mock judge responses (deterministic for testing)
judge_responses = {
    "simple": "moe_workhorse",
    "complex": "dense_specialist",
    "coding": "moe_workhorse",
    "default": "moe_workhorse",
}

def get_judge_response(prompt: str) -> str:
    """Return a deterministic judge response based on prompt content."""
    prompt_lower = prompt.lower()
    if "complex" in prompt_lower or "architecture" in prompt_lower:
        return "dense_specialist"
    if "code" in prompt_lower or "build" in prompt_lower:
        return "moe_workhorse"
    return judge_responses["default"]

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "loaded": state["loaded"]}
            for model, state in model_states.items()
        ]
    }

@app.get("/admin/api/models")
async def admin_list_models():
    return {
        "models": [
            {"id": model, "loaded": state["loaded"], "is_loading": state["is_loading"]}
            for model, state in model_states.items()
        ]
    }

@app.post("/admin/api/models/{model_id}/load")
async def load_model(model_id: str):
    if model_id in model_states:
        model_states[model_id]["loaded"] = True
        return {"status": "loaded"}
    return JSONResponse({"error": "model not found"}, status_code=404)

@app.post("/admin/api/models/{model_id}/unload")
async def unload_model(model_id: str):
    if model_id in model_states:
        model_states[model_id]["loaded"] = False
        return {"status": "unloaded"}
    return JSONResponse({"error": "model not found"}, status_code=404)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    
    # Determine response based on model
    if model == "Arch-Router-1.5B-mlx-8Bit":
        # Judge model: return route decision
        prompt = messages[0]["content"] if messages else ""
        route = get_judge_response(prompt)
        content = route
    else:
        # Backend model: return mock response
        content = "This is a mock response from the test model."
    
    if stream:
        async def stream_response():
            yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_response(), media_type="text/event-stream")
    
    return {
        "id": "mock-response",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

def start_mock_server(port: int = 8080):
    """Start the mock oMLX server."""
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    asyncio.get_event_loop().run_until_complete(server.serve())
    return server
```

### Step 2: Create Test Fixtures

**tests/conftest.py**:

```python
import pytest
import asyncio
import httpx
from tests.mock_omlx import app as mock_app

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def mock_omlx_client():
    """Provide a test client for the mock oMLX server."""
    from fastapi.testclient import TestClient
    client = TestClient(mock_app)
    yield client

@pytest.fixture
def sample_chat_request():
    """Sample chat completion request."""
    return {
        "model": "gateway-auto",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "max_tokens": 100,
    }

@pytest.fixture
def sample_coding_request():
    """Sample coding request."""
    return {
        "model": "gateway-auto",
        "messages": [{"role": "user", "content": "Build a Python API for user management"}],
        "stream": False,
        "max_tokens": 1000,
    }

@pytest.fixture
def sample_complex_request():
    """Sample complex request that should route to dense."""
    return {
        "model": "gateway-auto",
        "messages": [{"role": "user", "content": "Review the entire codebase for architectural issues and propose a redesign"}],
        "stream": False,
        "max_tokens": 2000,
    }

@pytest.fixture
def mock_memory_state():
    """Mock memory state for testing."""
    return {
        "available_gb": 32.0,
        "used_gb": 32.0,
        "total_gb": 64.0,
        "percent": 50.0,
        "guard_gb": 3.0,
        "hard_gb": 1.5,
        "pressure": False,
        "hard_pressure": False,
        "last_update": 0.0,
    }
```

### Step 3: Create Integration Test with Mock Server

**tests/integration/test_e2e.py**:

```python
import pytest
import asyncio
from fastapi.testclient import TestClient
from proxy import app as gateway_app

@pytest.fixture
def gateway_client():
    """Test client for the gateway."""
    return TestClient(gateway_app)

class TestEndToEnd:
    def test_health_endpoint(self, gateway_client):
        response = gateway_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "models" in data

    def test_models_endpoint(self, gateway_client):
        response = gateway_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) >= 3  # gateway-auto, gateway-moe, gateway-dense

    def test_chat_completion_non_streaming(self, gateway_client, sample_chat_request):
        response = gateway_client.post("/v1/chat/completions", json=sample_chat_request)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"]

    def test_chat_completion_streaming(self, gateway_client, sample_chat_request):
        sample_chat_request["stream"] = True
        response = gateway_client.post("/v1/chat/completions", json=sample_chat_request)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        # Verify SSE format
        lines = response.text.strip().split("\n")
        assert any(line.startswith("data:") for line in lines)

    def test_routing_decision_simple(self, gateway_client, sample_chat_request):
        """Simple request should route to MoE."""
        response = gateway_client.post("/v1/chat/completions", json=sample_chat_request)
        assert response.status_code == 200
        # Check metrics to verify routing
        metrics_response = gateway_client.get("/metrics")
        metrics = metrics_response.json()
        assert len(metrics["history"]) > 0
        last_request = metrics["history"][-1]
        assert last_request["route"] in ("moe", "dense")

    def test_cache_hit(self, gateway_client, sample_chat_request):
        """Second identical request should hit cache."""
        # First request (cache miss)
        response1 = gateway_client.post("/v1/chat/completions", json=sample_chat_request)
        assert response1.status_code == 200
        
        # Second request (cache hit)
        response2 = gateway_client.post("/v1/chat/completions", json=sample_chat_request)
        assert response2.status_code == 200
        assert response2.headers.get("X-Gateway-Cache") == "HIT"
```

### Step 4: Configure pytest

**pytest.ini**:

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
log_cli = true
log_cli_level = INFO
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    slow: marks tests as slow (deselect with '-m "not slow"')
```

### Step 5: Update CI to Use Mocks

**.github/workflows/ci.yml** (update test step):

```yaml
      - name: Run unit tests (no real models needed)
        run: pytest tests/ -v --tb=short -m "not integration"
      
      - name: Run integration tests (with mocks)
        run: pytest tests/integration/ -v --tb=short
```

### Step 6: Verify Locally

```bash
# Run unit tests (fast, no mocks needed)
pytest tests/ -v -m "not integration"

# Run integration tests (with mock server)
pytest tests/integration/ -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=proxy --cov-report=term-missing
```

## Success Criteria

- [ ] Mock oMLX server responds correctly to all admin API calls
- [ ] Mock judge model returns deterministic routes
- [ ] All unit tests pass without real models
- [ ] Integration tests pass with mock server
- [ ] CI runs tests without requiring 64GB RAM
- [ ] Test coverage > 80% on core logic

## Commands Reference

```bash
# Run unit tests only
pytest tests/ -v -m "not integration"

# Run integration tests with mocks
pytest tests/integration/ -v

# Run all tests with coverage
pytest tests/ --cov=proxy --cov-report=term-missing

# Run specific test
pytest tests/integration/test_e2e.py::TestEndToEnd::test_health_endpoint -v

# Run with verbose logging
pytest tests/ -v --log-cli-level=DEBUG

# Run tests in parallel
pytest tests/ -n auto  # requires pytest-xdist
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Mock server not responding | Check port conflict, use different port |
| Tests fail with real model errors | Ensure `OMLX_UPSTREAM` points to mock |
| Cache tests flaky | Clear cache in `setup_method` |
| Memory tests fail on CI | Mock `psutil.virtual_memory()` |
| Streaming tests timeout | Increase timeout in test config |
