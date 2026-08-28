"""
Unit tests for core proxy functions.
"""
from proxy import (
    cache_key,
    compute_score,
    conversation_key,
    extract_finish_reason,
    extract_usage,
    heuristic_policy,
    is_coding_request,
    is_explicit_continuation,
    normalize_policy,
    routing_context,
    stream_delta_text,
    task_state,
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


class TestHeuristicPolicy:
    """Tests for heuristic routing policy."""

    def test_heuristic_policy_simple(self):
        """Simple request should route to MoE."""
        body = {
            "messages": [{"role": "user", "content": "What is the capital of France?"}]
        }
        policy = heuristic_policy(body)
        assert policy["route"] == "moe"
        assert policy["confidence"] > 0.5

    def test_heuristic_policy_coding(self):
        """Coding request should be detected."""
        body = {
            "messages": [{"role": "user", "content": "Build a Python API for user management"}]
        }
        policy = heuristic_policy(body)
        assert policy["route"] in ("moe", "dense")

    def test_heuristic_policy_complex(self):
        """Complex request should route to dense."""
        body = {
            "messages": [{"role": "user", "content": "Review the entire codebase for architectural issues"}]
        }
        policy = heuristic_policy(body)
        assert policy["route"] == "dense"
        assert policy["confidence"] > 0.7

    def test_heuristic_policy_failure_recovery(self):
        """Failed implementation should route to dense."""
        body = {
            "messages": [
                {"role": "user", "content": "Build a Python API"},
                {"role": "assistant", "content": "Here's the code..."},
                {"role": "user", "content": "The Python API didn't work, fix it"}
            ]
        }
        policy = heuristic_policy(body)
        # The failure signal "didn't work" + coding context should trigger dense routing
        assert policy["route"] == "dense"
        assert policy["task_type"] == "failure_recovery"


class TestIsCodingRequest:
    """Tests for coding request detection."""

    def test_is_coding_request_true(self):
        """Should detect coding requests."""
        body = {
            "messages": [{"role": "user", "content": "Build a Python function to parse JSON"}]
        }
        assert is_coding_request(body) is True

    def test_is_coding_request_false(self):
        """Should not detect non-coding requests."""
        body = {
            "messages": [{"role": "user", "content": "What is the weather like today?"}]
        }
        assert is_coding_request(body) is False

    def test_is_coding_request_html(self):
        """Should detect HTML requests."""
        body = {
            "messages": [{"role": "user", "content": "Create an HTML page with a form"}]
        }
        assert is_coding_request(body) is True


class TestIsExplicitContinuation:
    """Tests for explicit continuation detection."""

    def test_is_explicit_continuation_true(self):
        """Should detect continuation phrases."""
        assert is_explicit_continuation("continue") is True
        assert is_explicit_continuation("go on") is True
        assert is_explicit_continuation("keep going") is True
        assert is_explicit_continuation("proceed") is True
        assert is_explicit_continuation("resume") is True

    def test_is_explicit_continuation_false(self):
        """Should not detect non-continuation phrases."""
        assert is_explicit_continuation("new question") is False
        assert is_explicit_continuation("hello") is False
        assert is_explicit_continuation("what is this?") is False

    def test_is_explicit_continuation_case_insensitive(self):
        """Should be case insensitive."""
        assert is_explicit_continuation("CONTINUE") is True
        assert is_explicit_continuation("Continue") is True


class TestTaskState:
    """Tests for task state extraction."""

    def test_task_state_single_user(self):
        """Should extract state for single user message."""
        body = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        state = task_state(body)
        assert state["user_turn_count"] == 1
        assert state["message_count"] == 1
        assert state["latest_user"] == "Hello"

    def test_task_state_multiple_users(self):
        """Should extract state for multiple user messages."""
        body = {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Second question"},
            ]
        }
        state = task_state(body)
        assert state["user_turn_count"] == 2
        assert state["message_count"] == 3
        assert state["latest_user"] == "Second question"

    def test_task_state_empty(self):
        """Should handle empty messages."""
        body = {"messages": []}
        state = task_state(body)
        assert state["user_turn_count"] == 0
        assert state["message_count"] == 0
        assert state["latest_user"] == ""


class TestConversationKey:
    """Tests for conversation key generation."""

    def test_conversation_key_deterministic(self):
        """Same conversation should produce same key."""
        body1 = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        }
        body2 = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        }
        assert conversation_key(body1) == conversation_key(body2)

    def test_conversation_key_different(self):
        """Different conversations should produce different keys."""
        body1 = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        body2 = {
            "messages": [{"role": "user", "content": "Goodbye"}]
        }
        assert conversation_key(body1) != conversation_key(body2)

    def test_conversation_key_no_user(self):
        """Should return None if no user message."""
        body = {
            "messages": [{"role": "system", "content": "You are helpful"}]
        }
        assert conversation_key(body) is None


class TestRoutingContext:
    """Tests for routing context extraction."""

    def test_routing_context_single_message(self):
        """Should extract context from single message."""
        body = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        context = routing_context(body)
        assert "USER" in context
        assert "Hello" in context

    def test_routing_context_multiple_messages(self):
        """Should extract context from multiple messages."""
        body = {
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second"},
            ]
        }
        context = routing_context(body)
        assert "First" in context
        assert "Response" in context
        assert "Second" in context

    def test_routing_context_truncation(self):
        """Should truncate context if too long."""
        body = {
            "messages": [{"role": "user", "content": "x" * 100000}]
        }
        context = routing_context(body)
        assert len(context) <= 25000  # Should be truncated


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
