"""
Performance benchmarking against the real running stack.
"""
import json
import statistics
import time

import httpx


def benchmark_model(client: httpx.Client, model: str, prompt: str, iterations: int = 5) -> dict:
    """Benchmark a single model."""
    latencies = []
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
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            token_count = len(content.split()) if content else 0
            latencies.append(elapsed)
            tokens.append(token_count)

    if latencies:
        avg_latency = statistics.mean(latencies)
        avg_tokens = statistics.mean(tokens)
        return {
            "model": model,
            "avg_latency_ms": round(avg_latency, 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "avg_tokens": round(avg_tokens, 2),
            "avg_tps": round(avg_tokens / (avg_latency / 1000), 2),
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
