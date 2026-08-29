"""
Integration tests for the gateway.
"""
import pytest
from fastapi.testclient import TestClient

from proxy import app


@pytest.fixture
def gateway_client():
    """Test client for the gateway."""
    return TestClient(app)


class TestEndpoints:
    """Tests for gateway endpoints."""

    def test_health_endpoint(self, gateway_client):
        """Health endpoint should return 200."""
        response = gateway_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "models" in data

    def test_models_endpoint(self, gateway_client):
        """Models endpoint should return model list."""
        response = gateway_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        # Should have at least gateway models
        model_ids = [m["id"] for m in data["data"]]
        assert "gateway-auto" in model_ids
        assert "gateway-moe" in model_ids
        assert "gateway-dense" in model_ids

    def test_dashboard(self, gateway_client):
        """Dashboard should return HTML."""
        response = gateway_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Inference Gateway" in response.text

    def test_chat_completion_non_streaming(self, gateway_client):
        """Non-streaming chat completion should not crash."""
        response = gateway_client.post("/v1/chat/completions", json={
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        # May fail if models not loaded, but should not crash
        assert response.status_code in (200, 502, 503)

    def test_chat_completion_streaming(self, gateway_client):
        """Streaming chat completion should not crash."""
        response = gateway_client.post("/v1/chat/completions", json={
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        })
        # May fail if models not loaded, but should not crash
        assert response.status_code in (200, 502, 503)
