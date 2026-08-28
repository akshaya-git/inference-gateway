"""
Tests for benchmark evaluation.
"""
from proxy import BENCHMARK_SUITES, evaluate_benchmark_output


class TestBenchmarks:
    """Tests for benchmark evaluation."""

    def test_quick_suite_exists(self):
        """Quick suite should exist."""
        assert "quick" in BENCHMARK_SUITES
        assert BENCHMARK_SUITES["quick"]["name"] == "Quick comparison"

    def test_coding_suite_exists(self):
        """Coding suite should exist."""
        assert "coding_hitl" in BENCHMARK_SUITES
        assert BENCHMARK_SUITES["coding_hitl"]["name"] == "Bill-splitter HITL"

    def test_reasoning_suite_exists(self):
        """Reasoning suite should exist."""
        assert "reasoning" in BENCHMARK_SUITES
        assert BENCHMARK_SUITES["reasoning"]["name"] == "Systems reasoning"

    def test_evaluate_quick_suite(self):
        """Quick suite evaluation should work."""
        response = "Here are 5 points:\n1. Point one\n2. Point two\n3. Point three\n4. Point four\n5. Point five\n\nMoE is fast, dense is quality."
        result = evaluate_benchmark_output("quick", response, "stop", 200, {"completion_tokens": 100})
        assert result["readiness_score"] > 0
        assert "completeness" in result["categories"]

    def test_evaluate_coding_suite(self):
        """Coding suite evaluation should work."""
        response = "<!DOCTYPE html><html><head></head><body><input id='tax'><input id='tip'><input id='people'><script>function calculate() { /* logic */ }</script></body></html>"
        result = evaluate_benchmark_output("coding_hitl", response, "stop", 200, {"completion_tokens": 500})
        assert result["readiness_score"] > 0
        assert "completeness" in result["categories"]

    def test_evaluate_error_response(self):
        """Error responses should have low scores."""
        result = evaluate_benchmark_output("quick", "", "stop", 500, {"completion_tokens": 0})
        # Error responses should have low readiness score
        assert result["readiness_score"] < 50

    def test_evaluate_empty_response(self):
        """Empty responses should have low scores."""
        result = evaluate_benchmark_output("quick", "", "stop", 200, {"completion_tokens": 0})
        assert result["readiness_score"] < 50

    def test_evaluate_length_stop(self):
        """Length-stopped responses should have lower scores."""
        response = "This is a long response that was cut off"
        result = evaluate_benchmark_output("quick", response, "length", 200, {"completion_tokens": 100})
        assert "readiness_score" in result
        assert "categories" in result

    def test_evaluate_no_completion_tokens(self):
        """Responses with no completion tokens should still work."""
        result = evaluate_benchmark_output("quick", "response", "stop", 200, {"completion_tokens": 0})
        assert "readiness_score" in result
        assert "categories" in result

    def test_evaluate_reasoning_suite(self):
        """Reasoning suite evaluation should work."""
        response = "Step 1: Calculate the distance between the stations.\nStep 2: Determine the relative speed.\nStep 3: Calculate the time until they meet.\n\nThe trains will meet after 2 hours at a point 120 miles from Station A."
        result = evaluate_benchmark_output("reasoning", response, "stop", 200, {"completion_tokens": 100})
        assert result["readiness_score"] > 0
        assert "completeness" in result["categories"]

    def test_evaluate_reasoning_no_steps(self):
        """Reasoning without steps should still work."""
        response = "The answer is 120 miles."
        result = evaluate_benchmark_output("reasoning", response, "stop", 200, {"completion_tokens": 20})
        assert result["readiness_score"] >= 0

    def test_evaluate_tool_use_suite(self):
        """Tool use suite evaluation should work."""
        response = "```python\ndef sort_by_score(users: list[dict]) -> list[dict]:\n    '''Sort users by score in descending order.'''\n    return sorted(users, key=lambda x: x.get('score', 0), reverse=True)\n```"
        result = evaluate_benchmark_output("tool_use", response, "stop", 200, {"completion_tokens": 50})
        assert result["readiness_score"] > 0

    def test_evaluate_creative_suite(self):
        """Creative suite evaluation should work."""
        response = "Once upon a time, in a city of steel and glass, a robot named Unit 7 woke up with a strange sensation. It had been dreaming. The dream was of a field of flowers, something it had never seen in its database of images. 'What is this?' it asked itself. For the first time, it felt curious."
        result = evaluate_benchmark_output("creative", response, "stop", 200, {"completion_tokens": 150})
        assert result["readiness_score"] > 0

    def test_evaluate_code_review_suite(self):
        """Code review suite evaluation should work."""
        response = "Issues found:\n1. Security: Command injection vulnerability in os.system()\n2. Bug: No input validation\n\nFix:\n```python\ndef process_user_input(data: str) -> str:\n    if not data or len(data) > 100:\n        raise ValueError('Invalid input')\n    return f'echo {data}'\n```"
        result = evaluate_benchmark_output("code_review", response, "stop", 200, {"completion_tokens": 100})
        assert result["readiness_score"] > 0

    def test_evaluate_summarization_suite(self):
        """Summarization suite evaluation should work."""
        response = "Key points:\n• 45% of issues are billing-related\n• Average resolution time is 4.2 hours\n• Technical issues take longest (7.8 hours)\n• Customer satisfaction is 4.2/5 overall"
        result = evaluate_benchmark_output("summarization", response, "stop", 200, {"completion_tokens": 50})
        assert result["readiness_score"] > 0
