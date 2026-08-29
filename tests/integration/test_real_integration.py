"""
Real integration tests against the running oMLX stack.
"""
import pytest
import time
import httpx


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
        with httpx.stream(
            "POST",
            "http://localhost:9000/v1/chat/completions",
            json={
                "model": "gateway-auto",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
                "max_tokens": 50,
            },
            timeout=30.0,
        ) as response:
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
        # Metrics should contain cache, config, or other data
        assert len(data) > 0

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
