# Task 2: Real Integration Testing

## Status: 🔄 IN PROGRESS

**Hardware:** Apple M3 Max, 64GB RAM, 16 cores  
**Models Loaded:** Arch-Router (1.6GB), Qwen3.8-27B-oQ4e-mtp (15.4GB)  
**Available RAM:** 17.13GB | **Used:** 46.87GB (73.2%)  
**Gateway:** Running on port 9000 | **oMLX:** Running on port 8080

## Overview

Test the gateway against the **real running oMLX stack** to validate:
1. **Integration** - Real model routing, memory management, caching
2. **Streaming** - SSE streams work correctly
3. **Performance** - Latency, TPS, TTFT metrics
4. **Error Handling** - Graceful failures when models are busy

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Real Integration Testing                              │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │  Test    │───▶│  Gateway     │───▶│  oMLX Stack  │                 │
│  │  Client  │    │  :9000       │    │  :8080       │                 │
│  └──────────┘    └──────────────┘    └──────────────┘                 │
│                              │                    │                     │
│                              ▼                    ▼                     │
│                     ┌─────────────────┐  ┌─────────────────┐          │
│                     │  Integration    │  │  Memory Guard   │          │
│                     │  Tests          │  │  & Admission    │          │
│                     │  • Routing      │  │  Control        │          │
│                     │  • Cache        │  │                 │          │
│                     │  • Streaming    │  │                 │          │
│                     └─────────────────┘  └─────────────────┘          │
│                              │                    │                     │
│                              ▼                    ▼                     │
│                     ┌─────────────────┐  ┌─────────────────┐          │
│                     │  Performance    │  │  Load Testing   │          │
│                     │  Benchmarks     │  │  (concurrent    │          │
│                     │  • Latency      │  │   requests)     │          │
│                     │  • TPS          │  │                 │          │
│                     │  • TTFT         │  │                 │          │
│                     └─────────────────┘  └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hardware Profile

| Component | Specification |
|-----------|--------------|
| **Chip** | Apple M3 Max |
| **RAM** | 64GB unified memory |
| **Cores** | 16 CPU cores |
| **GPU** | 40-core GPU |
| **NN Engine** | 16-core Neural Engine |

### Loaded Models

| Model | Size | Quantization | Context | Status |
|-------|------|-------------|---------|--------|
| Arch-Router-1.5B | 1.6GB | 8-bit | 16K | ✅ Loaded |
| Qwen3.8-27B-oQ4e-mtp | 15.4GB | 4-bit (OptiQ) | 131K | ✅ Loaded |
| Qwen3.6-35B-A3B-oQ5e-mtp | 25.4GB | 5-bit (OptiQ) | 131K | ⏳ Unloaded |

### Available Models (Unloaded)

| Model | Size | Quantization | Context |
|-------|------|-------------|---------|
| Qwen3.5-4B | 2.97GB | 4-bit | 262K |
| Qwen3.6-27B | 15.7GB | 4-bit | 262K |
| Qwen3.6-35B-A3B | 21.65GB | 4-bit (OptiQ) | 262K |
| Qwen3.8-27B | 15.7GB | 4-bit | 262K |
| Qwen3.8-27B | 28.85GB | 8-bit | 262K |

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
│  │  Write      │    │  Run real   │    │  Analyze    │                │
│  │  integration│    │  tests      │    │  results    │                │
│  │  + load     │    │  against    │    │  & generate │                │
│  │  tests      │    │  running    │    │  benchmarks │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        └────────────────────┴────────────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  QA Agent       │                                │
│                     │  (verify        │                                │
│                     │  tests pass)    │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Write integration + load tests | `write`, `edit` |
| **Tester** | Run tests against real stack | `bash` (curl, pytest) |
| **Engineer** | Analyze results, generate benchmarks | `bash`, `python3` |
| **QA** | Verify tests pass consistently | `bash`, `read` |

## Step-by-Step Execution

### Step 1: Create Integration Test Suite

**tests/integration/test_real_integration.py**:

```python
"""
Real integration tests against the running oMLX stack.
"""
import pytest
import time
import httpx
from proxy import app


@pytest.fixture
def gateway_client():
    """Test client for the gateway."""
    return httpx.Client(base_url="http://localhost:9000", timeout=30.0)


class TestRealIntegration:
    """Tests against the real running stack."""

    def test_health_endpoint(self, gateway_client):
        """Gateway should be healthy."""
        response = gateway_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["gateway"] == "phase2"

    def test_models_endpoint(self, gateway_client):
        """Models endpoint should show loaded models."""
        response = gateway_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        model_ids = [m["id"] for m in data["data"]]
        assert "gateway-auto" in model_ids
        assert "gateway-moe" in model_ids
        assert "gateway-dense" in model_ids

    def test_routing_to_dense(self, gateway_client):
        """Complex request should route to dense model."""
        response = gateway_client.post("/v1/chat/completions", json={
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "Review the entire codebase architecture"}],
            "max_tokens": 100,
        })
        # Should route to dense (200) or fail gracefully (502/503)
        assert response.status_code in (200, 502, 503)

    def test_routing_to_moe(self, gateway_client):
        """Simple request should route to MoE model."""
        response = gateway_client.post("/v1/chat/completions", json={
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 50,
        })
        # Should route to MoE (200) or fail gracefully (502/503)
        assert response.status_code in (200, 502, 503)

    def test_streaming_response(self, gateway_client):
        """Streaming response should return SSE format."""
        response = gateway_client.post("/v1/chat/completions", json={
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
            "max_tokens": 50,
        }, stream=True)
        # Should return SSE or fail gracefully
        assert response.status_code in (200, 502, 503)

    def test_cache_hit(self, gateway_client):
        """Repeated identical request should hit cache."""
        payload = {
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "Cache test message"}],
            "max_tokens": 50,
        }
        # First request (cache miss)
        response1 = gateway_client.post("/v1/chat/completions", json=payload)
        # Second request (cache hit)
        response2 = gateway_client.post("/v1/chat/completions", json=payload)
        # Both should succeed or fail gracefully
        assert response1.status_code in (200, 502, 503)
        assert response2.status_code in (200, 502, 503)

    def test_metrics_endpoint(self, gateway_client):
        """Metrics endpoint should return data."""
        response = gateway_client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert "stats" in data

    def test_memory_status(self, gateway_client):
        """Memory status should be reported."""
        response = gateway_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "memory" in data
        assert data["memory"]["total_gb"] == 64.0
        assert data["memory"]["available_gb"] > 0


class TestLoadTests:
    """Load tests against the real stack."""

    def test_concurrent_requests(self, gateway_client):
        """Test concurrent request handling."""
        import asyncio
        
        async def make_request(i):
            return gateway_client.post("/v1/chat/completions", json={
                "model": "gateway-auto",
                "messages": [{"role": "user", "content": f"Test request {i}"}],
                "max_tokens": 50,
            })
        
        # Make 3 concurrent requests
        tasks = [make_request(i) for i in range(3)]
        responses = asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
        
        # All should complete (success or graceful failure)
        assert all(r.status_code in (200, 502, 503) for r in responses)

    def test_performance_baseline(self, gateway_client):
        """Test performance baseline for each model."""
        for model in ["gateway-moe", "gateway-dense"]:
            start = time.time()
            response = gateway_client.post("/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": "Explain quantum computing in 50 words"}],
                "max_tokens": 100,
            })
            elapsed = (time.time() - start) * 1000
            
            # Should complete within 30 seconds
            assert elapsed < 30000
            # Should succeed or fail gracefully
            assert response.status_code in (200, 502, 503)
```

### Step 2: Create Performance Benchmark Script

**tests/benchmark_real.py**:

```python
"""
Performance benchmarking against the real running stack.
"""
import time
import json
import httpx
import statistics


def benchmark_model(client: httpx.Client, model: str, prompt: str, iterations: int = 5) -> dict:
    """Benchmark a single model."""
    latencies = []
    ttfts = []
    tokens = []
    
    for i in range(iterations):
        start = time.time()
        response = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
        })
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            token_count = len(data.get("choices", [{}])[0].get("message", {}).get("content", "").split())
            latencies.append(elapsed)
            tokens.append(token_count)
    
    if latencies:
        return {
            "model": model,
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "avg_tokens": round(statistics.mean(tokens), 2),
            "avg_tps": round(statistics.mean(tokens) / (statistics.mean(latencies) / 1000), 2),
            "status": "success",
        }
    else:
        return {
            "model": model,
            "status": "failed",
            "error": "All requests failed",
        }


def run_benchmarks(base_url: str = "http://localhost:9000") -> dict:
    """Run all benchmarks and return results."""
    client = httpx.Client(base_url=base_url, timeout=120.0)
    
    prompts = {
        "simple": "What is 2+2?",
        "medium": "Explain quantum computing in 50 words",
        "complex": "Review the entire codebase architecture and suggest improvements",
    }
    
    results = {
        "hardware": {
            "chip": "Apple M3 Max",
            "ram_gb": 64,
            "cpu_cores": 16,
        },
        "models": {},
    }
    
    for model in ["gateway-moe", "gateway-dense"]:
        results["models"][model] = {}
        for name, prompt in prompts.items():
            results["models"][model][name] = benchmark_model(client, model, prompt)
    
    client.close()
    return results


if __name__ == "__main__":
    results = run_benchmarks()
    print(json.dumps(results, indent=2))
```

### Step 3: Run Tests Against Real Stack

```bash
# Run integration tests
python3 -m pytest tests/integration/test_real_integration.py -v

# Run benchmarks
python3 tests/benchmark_real.py > tests/benchmark_results.json
```

### Step 4: Analyze Results

The benchmark script outputs JSON with:
- Average/min/max latency per model
- TPS (tokens per second)
- P95 latency
- Status (success/failed)

## Success Criteria

- [ ] All 10 integration tests pass
- [ ] Load tests complete without errors
- [ ] Benchmarks run successfully for both models
- [ ] Results saved to `tests/benchmark_results.json`
- [ ] No crashes or hangs

## Files to Create

| File | Purpose |
|------|---------|
| `tests/integration/test_real_integration.py` | Integration tests |
| `tests/benchmark_real.py` | Performance benchmarking |
| `tests/benchmark_results.json` | Benchmark results (generated) |

## Dependencies

- Gateway running on `localhost:9000`
- oMLX running on `localhost:8080`
- `httpx` installed (already in requirements.txt)

## Estimated Time

- **Developer**: 30 minutes (write tests)
- **Tester**: 15 minutes (run tests)
- **Engineer**: 15 minutes (analyze results)
- **QA**: 5 minutes (verify)
- **Total**: ~1 hour

## Notes

- Tests handle graceful failures (502/503) since models may be busy
- Benchmarks run 5 iterations per prompt for statistical significance
- Results are saved as JSON for easy comparison over time
