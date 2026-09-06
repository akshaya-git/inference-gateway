"""
Unit tests for core proxy functions.
"""
import asyncio

import proxy
from proxy import (
    MODEL_CAPABILITIES,
    cache_key,
    capability_adjustment,
    compute_score,
    extract_finish_reason,
    extract_usage,
    normalize_policy,
    stream_delta_text,
)


class TestCache:
    """Tests for cache functions."""

    def test_cache_key_deterministic(self):
        """Same input should produce same cache key."""
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        key1 = cache_key(body)
        key2 = cache_key(body)
        assert key1 == key2

    def test_cache_key_different_for_different_inputs(self):
        """Different inputs should produce different cache keys."""
        body1 = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        body2 = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
        assert cache_key(body1) != cache_key(body2)

    def test_cache_key_order_independent(self):
        """Dict key order should not affect cache key."""
        body1 = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        body2 = {"messages": [{"role": "user", "content": "hi"}], "model": "test"}
        assert cache_key(body1) == cache_key(body2)

    def test_cache_key_is_sha256(self):
        """Cache key should be a valid SHA256 hex string."""
        body = {"model": "test", "messages": []}
        key = cache_key(body)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestExtractUsage:
    """Tests for usage extraction."""

    def test_extract_usage_complete(self):
        """Should extract all usage fields."""
        obj = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 10},
            }
        }
        usage = extract_usage(obj)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["cached_tokens"] == 10

    def test_extract_usage_empty(self):
        """Should handle missing usage."""
        obj = {}
        usage = extract_usage(obj)
        assert usage["prompt_tokens"] is None
        assert usage["completion_tokens"] is None
        assert usage["total_tokens"] is None
        assert usage["cached_tokens"] is None

    def test_extract_usage_partial(self):
        """Should handle partial usage data."""
        obj = {"usage": {"prompt_tokens": 100}}
        usage = extract_usage(obj)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] is None


class TestExtractFinishReason:
    """Tests for finish reason extraction."""

    def test_extract_finish_reason_stop(self):
        """Should extract 'stop' finish reason."""
        obj = {"choices": [{"finish_reason": "stop"}]}
        assert extract_finish_reason(obj) == "stop"

    def test_extract_finish_reason_length(self):
        """Should extract 'length' finish reason."""
        obj = {"choices": [{"finish_reason": "length"}]}
        assert extract_finish_reason(obj) == "length"

    def test_extract_finish_reason_empty(self):
        """Should return empty string for missing finish reason."""
        obj = {"choices": [{}]}
        assert extract_finish_reason(obj) == ""

    def test_extract_finish_reason_no_choices(self):
        """Should handle missing choices."""
        obj = {}
        assert extract_finish_reason(obj) == ""


class TestStreamDeltaText:
    """Tests for SSE delta text extraction."""

    def test_stream_delta_text_content(self):
        """Should extract content from delta."""
        obj = {"choices": [{"delta": {"content": "Hello"}}]}
        answer, reasoning = stream_delta_text(obj)
        assert answer == "Hello"
        assert reasoning == ""

    def test_stream_delta_text_reasoning(self):
        """Should extract reasoning content."""
        obj = {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
        answer, reasoning = stream_delta_text(obj)
        assert answer == ""
        assert reasoning == "thinking..."

    def test_stream_delta_text_empty(self):
        """Should handle empty delta."""
        obj = {"choices": [{"delta": {}}]}
        answer, reasoning = stream_delta_text(obj)
        assert answer == ""
        assert reasoning == ""

    def test_stream_delta_text_role_only(self):
        """Should handle role-only delta (no content)."""
        obj = {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
        answer, reasoning = stream_delta_text(obj)
        assert answer == ""
        assert reasoning == ""

    def test_stream_delta_text_list_content(self):
        """Should handle list content format."""
        obj = {"choices": [{"delta": {"content": [{"text": "Hello"}, {"text": " World"}]}}]}
        answer, reasoning = stream_delta_text(obj)
        assert answer == "Hello World"


class TestNormalizePolicy:
    """Tests for policy normalization."""

    def test_normalize_policy_valid(self):
        """Should normalize a valid policy."""
        policy = normalize_policy({
            "route": "moe",
            "confidence": 0.8,
            "reason": "test",
            "task_type": "coding",
            "effort": "balanced",
        })
        assert policy["route"] == "moe"
        assert policy["confidence"] == 0.8
        assert policy["task_type"] == "coding"
        assert policy["effort"] == "balanced"

    def test_normalize_policy_invalid_route(self):
        """Should fallback to default for invalid route."""
        policy = normalize_policy({"route": "invalid"})
        assert policy["route"] in ("moe", "dense")

    def test_normalize_policy_confidence_clamped(self):
        """Should clamp confidence to [0, 1]."""
        policy = normalize_policy({"confidence": 1.5})
        assert policy["confidence"] == 1.0

        policy = normalize_policy({"confidence": -0.5})
        assert policy["confidence"] == 0.0

    def test_normalize_policy_effort_valid(self):
        """Should accept valid effort levels."""
        for effort in ("fast", "balanced", "high"):
            policy = normalize_policy({"effort": effort})
            assert policy["effort"] == effort

    def test_normalize_policy_effort_invalid(self):
        """Should fallback to default for invalid effort."""
        policy = normalize_policy({"effort": "invalid"})
        assert policy["effort"] in ("fast", "balanced", "high")

    def test_normalize_policy_max_tokens_clamped(self):
        """Should clamp max_tokens to valid range."""
        policy = normalize_policy({"max_tokens": 100000})
        assert policy["max_tokens"] <= 32144

        policy = normalize_policy({"max_tokens": 10})
        assert policy["max_tokens"] >= 512


class TestCapabilityAdjustment:
    """Tests for capability-aware route adjustment (vision, context window)."""

    def test_vision_switch(self, monkeypatch):
        monkeypatch.setitem(MODEL_CAPABILITIES, "moe", {"context_window": 128000, "max_output": 32000, "vision": False})
        monkeypatch.setitem(MODEL_CAPABILITIES, "dense", {"context_window": 128000, "max_output": 32000, "vision": True})
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]}]}
        route, note = capability_adjustment("moe", body)
        assert route == "dense"
        assert "vision" in note

    def test_vision_no_capable_model(self, monkeypatch):
        monkeypatch.setitem(MODEL_CAPABILITIES, "moe", {"context_window": 128000, "max_output": 32000, "vision": False})
        monkeypatch.setitem(MODEL_CAPABILITIES, "dense", {"context_window": 128000, "max_output": 32000, "vision": False})
        body = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]}]}
        route, note = capability_adjustment("moe", body)
        assert route == "moe"
        assert "no vision-capable model" in note

    def test_context_window_switch(self, monkeypatch):
        monkeypatch.setitem(MODEL_CAPABILITIES, "moe", {"context_window": 1000, "max_output": 32000, "vision": False})
        monkeypatch.setitem(MODEL_CAPABILITIES, "dense", {"context_window": 128000, "max_output": 32000, "vision": False})
        body = {"messages": [{"role": "user", "content": "x" * 8000}]}  # ~2000 tokens
        route, note = capability_adjustment("moe", body)
        assert route == "dense"
        assert "window" in note

    def test_no_adjustment_when_fits(self, monkeypatch):
        monkeypatch.setitem(MODEL_CAPABILITIES, "moe", {"context_window": 128000, "max_output": 32000, "vision": False})
        monkeypatch.setitem(MODEL_CAPABILITIES, "dense", {"context_window": 128000, "max_output": 32000, "vision": False})
        body = {"messages": [{"role": "user", "content": "hello"}]}
        route, note = capability_adjustment("moe", body)
        assert route == "moe"
        assert note == ""


class TestChoosePolicy:
    """Tests for policy selection including the ROUTING_ENABLED pin."""

    def test_routing_disabled_pins_fallback(self, monkeypatch):
        monkeypatch.setattr(proxy, "ROUTING_ENABLED", False)
        body = {"messages": [{"role": "user", "content": "Fix the race condition in my worker pool"}]}
        policy, _ = asyncio.run(proxy.choose_policy(body))
        assert policy["route"] == proxy.FALLBACK_ROUTE
        assert policy["complexity"] is None
        assert "routing disabled" in policy["reason"]

    def test_routing_enabled_uses_engine(self, monkeypatch):
        monkeypatch.setattr(proxy, "ROUTING_ENABLED", True)
        body = {"messages": [{"role": "user", "content": "Fix the race condition in my worker pool"}]}
        policy, _ = asyncio.run(proxy.choose_policy(body))
        assert policy["route"] == "dense"
        assert policy["complexity"] == 6


class TestComputeScore:
    """Tests for compute score calculation."""

    def test_compute_score_zero(self):
        """Should return 0 for zero inputs."""
        score = compute_score(0, 0, 0, 0)
        assert score == 0.0

    def test_compute_score_latency(self):
        """Should account for latency."""
        score = compute_score(1000, 0, 0, 0)  # 1 second
        assert score > 0

    def test_compute_score_tokens(self):
        """Should account for tokens."""
        score = compute_score(0, 1000, 500, 0)
        assert score > 0

    def test_compute_score_swap(self):
        """Should account for swap time."""
        score = compute_score(0, 0, 0, 1000)  # 1 second swap
        assert score > 0
