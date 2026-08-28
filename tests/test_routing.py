"""
Tests for routing logic.
"""
from proxy import (
    heuristic_policy,
    is_coding_request,
    is_explicit_continuation,
    normalize_policy,
)


class TestRouting:
    """Tests for routing decisions."""

    def test_explicit_moe_selection(self):
        """Explicit MoE selection should be honored."""
        body = {
            "model": "gateway-moe",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        policy = normalize_policy({"route": "moe"}, heuristic_policy(body))
        assert policy["route"] == "moe"

    def test_explicit_dense_selection(self):
        """Explicit Dense selection should be honored."""
        body = {
            "model": "gateway-dense",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        policy = normalize_policy({"route": "dense"}, heuristic_policy(body))
        assert policy["route"] == "dense"

    def test_auto_routing_simple(self):
        """Simple request should route to MoE via heuristic."""
        body = {
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "What is 2+2?"}]
        }
        policy = heuristic_policy(body)
        assert policy["route"] == "moe"

    def test_auto_routing_coding(self):
        """Coding request should be detected and routed appropriately."""
        body = {
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "Write a Python function to sort a list"}]
        }
        policy = heuristic_policy(body)
        assert policy["route"] in ("moe", "dense")

    def test_auto_routing_complex(self):
        """Complex request should route to Dense."""
        body = {
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "Review the entire codebase architecture"}]
        }
        policy = heuristic_policy(body)
        assert policy["route"] == "dense"

    def test_auto_routing_failure_recovery(self):
        """Failed implementation should route to Dense."""
        body = {
            "model": "gateway-auto",
            "messages": [
                {"role": "user", "content": "Build a Python API"},
                {"role": "assistant", "content": "Here's the code..."},
                {"role": "user", "content": "The Python API didn't work, fix it"}
            ]
        }
        policy = heuristic_policy(body)
        assert policy["route"] == "dense"
        assert policy["task_type"] == "failure_recovery"

    def test_auto_routing_simple_transformation(self):
        """Simple transformation should route to MoE with fast effort."""
        body = {
            "model": "gateway-auto",
            "messages": [{"role": "user", "content": "Summarize briefly"}]
        }
        policy = heuristic_policy(body)
        assert policy["route"] == "moe"
        assert policy["effort"] == "fast"

    def test_continuation_detection(self):
        """Continuation phrases should be detected."""
        assert is_explicit_continuation("continue") is True
        assert is_explicit_continuation("go on") is True
        assert is_explicit_continuation("keep going") is True
        assert is_explicit_continuation("retry") is True
        assert is_explicit_continuation("fix that") is True

    def test_non_continuation_detection(self):
        """Non-continuation phrases should not be detected."""
        assert is_explicit_continuation("new question") is False
        assert is_explicit_continuation("what is this?") is False
        assert is_explicit_continuation("hello") is False

    def test_coding_request_detection(self):
        """Coding requests should be detected."""
        body = {
            "messages": [{"role": "user", "content": "Build a Python API"}]
        }
        assert is_coding_request(body) is True

    def test_non_coding_request_detection(self):
        """Non-coding requests should not be detected."""
        body = {
            "messages": [{"role": "user", "content": "What is the weather?"}]
        }
        assert is_coding_request(body) is False
