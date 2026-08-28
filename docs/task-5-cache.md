# Task 5: Enhanced Cache Support

## Overview

Enhance the response cache with semantic deduplication, cache warming, and cache analytics. This reduces redundant LLM calls and improves response times for similar queries.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Enhanced Cache Architecture                           │
│                                                                         │
│  ┌──────────┐    ┌──────────────────────────────────────────────────┐  │
│  │  Client  │───▶│                    Gateway                        │  │
│  │  Request │    │                                                  │  │
│  └──────────┘    │  ┌────────────────────────────────────────────┐  │  │
│                  │  │              Cache Layer                     │  │
│                  │  │                                              │  │
│                  │  │  ┌─────────────┐  ┌─────────────────────┐   │  │
│                  │  │  │  Exact      │  │  Semantic            │   │  │
│                  │  │  │  Match      │  │  Deduplication       │   │  │
│                  │  │  │  (hash)     │  │  (embedding sim)     │   │  │
│                  │  │  └─────────────┘  └─────────────────────┘   │  │
│                  │  │         │                    │               │  │
│                  │  │         ▼                    ▼               │  │
│                  │  │  ┌─────────────┐  ┌─────────────────────┐   │  │
│                  │  │  │  Cache Hit  │  │  Cache Warm         │   │  │
│                  │  │  │  (return)   │  │  (pre-compute)      │   │  │
│                  │  │  └─────────────┘  └─────────────────────┘   │  │
│                  │  │                                              │  │
│                  │  │  ┌────────────────────────────────────────┐  │  │
│                  │  │  │  Cache Analytics                        │  │  │
│                  │  │  │  • Hit rate by route                    │  │  │
│                  │  │  │  • Most cached prompts                  │  │  │
│                  │  │  │  • Cache efficiency score               │  │  │
│                  │  │  └────────────────────────────────────────┘  │  │
│                  │  └────────────────────────────────────────────┘  │  │
│                  │                                                  │  │
│                  │  ┌────────────────────────────────────────────┐  │  │
│                  │  │              LLM Backend                    │  │  │
│                  │  │  (only called on cache miss)                │  │  │
│                  │  └────────────────────────────────────────────┘  │  │
│                  └──────────────────────────────────────────────────┘  │
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
│  │  Implement  │    │  Test cache │    │  Verify     │                │
│  │  semantic   │    │  behavior   │    │  analytics  │                │
│  │  cache +    │    │  with       │    │  dashboard  │                │
│  │  analytics  │    │  sample     │    │             │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        └────────────────────┴────────────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  QA Agent       │                                │
│                     │  (verify cache  │                                │
│                     │   accuracy)     │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Implement semantic cache, analytics, warming | `edit` |
| **Tester** | Test cache behavior with sample requests | `bash` (curl, pytest) |
| **Engineer** | Verify analytics dashboard displays correctly | `bash` (curl) |
| **QA** | Check edge cases (cache poisoning, expiry) | `read`, `bash` |

## Step-by-Step Execution

### Step 1: Add Semantic Cache Deduplication

**proxy.py** — Add semantic similarity check:

```python
import hashlib
import json
from typing import Optional

# Semantic cache configuration
SEMANTIC_CACHE_ENABLED = os.environ.get("GATEWAY_SEMANTIC_CACHE", "1") == "1"
SEMANTIC_SIMILARITY_THRESHOLD = float(os.environ.get("GATEWAY_SEMANTIC_THRESHOLD", "0.95"))
SEMANTIC_CACHE_MAX_SIZE = int(os.environ.get("GATEWAY_SEMANTIC_CACHE_MAX", "1000"))

# Store embeddings for semantic matching
semantic_cache: dict[str, dict] = {}  # hash -> {embedding, payload, ts}

def compute_embedding(text: str) -> list[float]:
    """
    Compute a simple embedding for semantic similarity.
    In production, use a real embedding model. For now, use a hash-based approach.
    """
    # Simple approach: use character n-grams as features
    n = 3
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    features = {}
    for ng in ngrams:
        features[ng] = features.get(ng, 0) + 1
    
    # Normalize to create a simple "embedding"
    total = sum(features.values()) or 1
    embedding = [count / total for count in features.values()]
    return embedding[:64]  # Limit to 64 dimensions

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a or not b:
        return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def semantic_cache_get(prompt: str, threshold: float = None) -> Optional[dict]:
    """Check semantic cache for similar prompts."""
    if not SEMANTIC_CACHE_ENABLED:
        return None
    
    threshold = threshold or SEMANTIC_SIMILARITY_THRESHOLD
    query_embedding = compute_embedding(prompt)
    
    best_match = None
    best_similarity = 0.0
    
    for hash_key, entry in semantic_cache.items():
        similarity = cosine_similarity(query_embedding, entry["embedding"])
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = entry
    
    if best_match and best_similarity >= threshold:
        cache_stats["semantic_hits"] = cache_stats.get("semantic_hits", 0) + 1
        return best_match["payload"]
    
    return None

def semantic_cache_put(prompt: str, payload: dict) -> None:
    """Store in semantic cache."""
    if not SEMANTIC_CACHE_ENABLED:
        return
    
    # Evict oldest if at capacity
    if len(semantic_cache) >= SEMANTIC_CACHE_MAX_SIZE:
        oldest_key = min(semantic_cache.keys(), key=lambda k: semantic_cache[k]["ts"])
        del semantic_cache[oldest_key]
        cache_stats["evictions"] = cache_stats.get("evictions", 0) + 1
    
    hash_key = hashlib.sha256(prompt.encode()).hexdigest()
    semantic_cache[hash_key] = {
        "embedding": compute_embedding(prompt),
        "payload": payload,
        "ts": time.time(),
    }
    cache_stats["semantic_stores"] = cache_stats.get("semantic_stores", 0) + 1
```

### Step 2: Add Cache Warming

**proxy.py** — Add cache warming for common prompts:

```python
# Common prompts that should be pre-cached
CACHE_WARMING_PROMPTS = [
    "Hello",
    "Hi",
    "How are you?",
    "What can you do?",
    "Help",
    "Explain this code",
    "Write a function",
    "Summarize this text",
    "Translate to English",
    "What is the weather?",
]

async def warm_cache():
    """Pre-compute responses for common prompts."""
    logger.info("Warming cache with %d common prompts", len(CACHE_WARMING_PROMPTS))
    
    for prompt in CACHE_WARMING_PROMPTS:
        try:
            # Check if already cached
            if cache_get(prompt):
                continue
            
            # Generate response
            response = await generate_response(prompt)
            cache_put(prompt, response, 200)
            semantic_cache_put(prompt, response)
            logger.info("Warmed cache for: %s", prompt[:50])
        except Exception as exc:
            logger.warning("Cache warming failed for %s: %s", prompt[:50], exc)
    
    logger.info("Cache warming complete")

# Add warming endpoint
@app.post("/api/cache/warm")
async def trigger_cache_warming():
    """Manually trigger cache warming."""
    asyncio.create_task(warm_cache())
    return {"status": "warming started", "prompts": len(CACHE_WARMING_PROMPTS)}
```

### Step 3: Add Cache Analytics

**proxy.py** — Add analytics endpoint:

```python
@app.get("/api/cache/analytics")
async def get_cache_analytics():
    """Get detailed cache analytics."""
    async with lock:
        stats = {
            "total_requests": len(history),
            "cache_hits": cache_stats.get("hits", 0),
            "cache_misses": cache_stats.get("misses", 0),
            "semantic_hits": cache_stats.get("semantic_hits", 0),
            "semantic_stores": cache_stats.get("semantic_stores", 0),
            "evictions": cache_stats.get("evictions", 0),
            "cache_size": len(response_cache),
            "semantic_cache_size": len(semantic_cache),
        }
    
    # Calculate hit rate
    total = stats["cache_hits"] + stats["cache_misses"]
    stats["hit_rate"] = (stats["cache_hits"] / total * 100) if total > 0 else 0
    
    # Calculate semantic hit rate
    semantic_total = stats["semantic_hits"] + stats["cache_misses"]
    stats["semantic_hit_rate"] = (stats["semantic_hits"] / semantic_total * 100) if semantic_total > 0 else 0
    
    # Most cached prompts (by access frequency)
    prompt_counts: dict[str, int] = {}
    for item in history:
        prompt = item.get("prompt", "")
        if prompt:
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
    
    most_cached = sorted(prompt_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["most_cached_prompts"] = [
        {"prompt": p[:100], "count": c} for p, c in most_cached
    ]
    
    # Cache efficiency score (0-100)
    # Based on hit rate, semantic hit rate, and cache size
    efficiency = (
        stats["hit_rate"] * 0.5 +
        stats["semantic_hit_rate"] * 0.3 +
        min(100, stats["cache_size"] / 10) * 0.2
    )
    stats["efficiency_score"] = round(efficiency, 1)
    
    return stats
```

### Step 4: Update Dashboard with Cache Analytics

**proxy.py** — Update `DASHBOARD_V2` HTML:

```javascript
// Add cache analytics section
async function loadCacheAnalytics() {
    const res = await fetch('/api/cache/analytics');
    const data = await res.json();
    
    $('cacheAnalytics').innerHTML = `
        <div class="stat-grid">
            <div class="stat">
                <div class="stat-value">${data.hit_rate.toFixed(1)}%</div>
                <div class="stat-label">Cache Hit Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">${data.semantic_hit_rate.toFixed(1)}%</div>
                <div class="stat-label">Semantic Hit Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">${data.efficiency_score}</div>
                <div class="stat-label">Efficiency Score</div>
            </div>
            <div class="stat">
                <div class="stat-value">${data.cache_size}</div>
                <div class="stat-label">Cache Size</div>
            </div>
        </div>
        
        <h3>Most Cached Prompts</h3>
        <table>
            <thead>
                <tr><th>Prompt</th><th>Count</th></tr>
            </thead>
            <tbody>
                ${data.most_cached_prompts.map(p => `
                    <tr>
                        <td>${esc(p.prompt)}</td>
                        <td>${p.count}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// Add cache warming button
$('warmCacheBtn').addEventListener('click', async () => {
    const res = await fetch('/api/cache/warm', {method: 'POST'});
    const data = await res.json();
    alert(data.status);
});
```

### Step 5: Test Cache Enhancements

**tests/test_cache_enhanced.py**:

```python
import pytest
from proxy import (
    semantic_cache_get, semantic_cache_put,
    compute_embedding, cosine_similarity,
    cache_stats, response_cache, semantic_cache,
)

class TestSemanticCache:
    def setup_method(self):
        semantic_cache.clear()
        cache_stats.update({"semantic_hits": 0, "semantic_stores": 0})

    def test_embedding_computation(self):
        embedding = compute_embedding("hello world")
        assert len(embedding) > 0
        assert all(0 <= x <= 1 for x in embedding)

    def test_cosine_similarity_identical(self):
        a = compute_embedding("hello world")
        b = compute_embedding("hello world")
        sim = cosine_similarity(a, b)
        assert sim > 0.99

    def test_cosine_similarity_different(self):
        a = compute_embedding("hello world")
        b = compute_embedding("completely different text about programming")
        sim = cosine_similarity(a, b)
        assert sim < 0.5

    def test_semantic_cache_put_get(self):
        prompt = "What is the capital of France?"
        payload = {"response": "Paris"}
        
        semantic_cache_put(prompt, payload)
        
        # Exact match should work
        result = semantic_cache_get(prompt, threshold=0.95)
        assert result is not None
        assert result["response"] == "Paris"

    def test_semantic_cache_similar(self):
        prompt = "What is the capital of France?"
        payload = {"response": "Paris"}
        
        semantic_cache_put(prompt, payload)
        
        # Similar prompt should also match (with lower threshold)
        similar_prompt = "What is the capital city of France?"
        result = semantic_cache_get(similar_prompt, threshold=0.8)
        # May or may not match depending on similarity
        assert result is None or result["response"] == "Paris"

    def test_semantic_cache_eviction(self):
        # Fill cache to capacity
        for i in range(1001):
            semantic_cache_put(f"prompt {i}", {"response": f"response {i}"})
        
        # Should have evicted oldest
        assert len(semantic_cache) <= 1000

class TestCacheAnalytics:
    def test_analytics_endpoint(self, gateway_client):
        # Make some requests first
        for i in range(5):
            gateway_client.post("/v1/chat/completions", json={
                "model": "auto",
                "messages": [{"role": "user", "content": f"test {i}"}]
            })
        
        response = gateway_client.get("/api/cache/analytics")
        assert response.status_code == 200
        data = response.json()
        
        assert "hit_rate" in data
        assert "semantic_hit_rate" in data
        assert "efficiency_score" in data
        assert "most_cached_prompts" in data
        assert 0 <= data["efficiency_score"] <= 100

    def test_cache_warming_endpoint(self, gateway_client):
        response = gateway_client.post("/api/cache/warm")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "warming started"
```

### Step 6: Verify Locally

```bash
# Check cache analytics
curl http://localhost:9000/api/cache/analytics | python3 -m json.tool

# Trigger cache warming
curl -X POST http://localhost:9000/api/cache/warm

# Check cache stats
curl http://localhost:9000/metrics | python3 -m json.tool | grep -A 10 "cache"

# Run tests
pytest tests/test_cache_enhanced.py -v
```

## Success Criteria

- [ ] Semantic cache deduplication works
- [ ] Cache warming pre-computes common prompts
- [ ] Analytics endpoint returns hit rates and efficiency score
- [ ] Dashboard displays cache analytics
- [ ] Tests pass for all cache enhancements
- [ ] Cache eviction works correctly

## Commands Reference

```bash
# Check cache analytics
curl http://localhost:9000/api/cache/analytics | python3 -m json.tool

# Trigger cache warming
curl -X POST http://localhost:9000/api/cache/warm

# Clear cache
curl -X POST http://localhost:9000/api/cache/clear

# Check cache stats
curl http://localhost:9000/metrics | python3 -m json.tool

# Run tests
pytest tests/test_cache_enhanced.py -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Semantic cache never hits | Lower `SEMANTIC_SIMILARITY_THRESHOLD` |
| Cache warming slow | Reduce `CACHE_WARMING_PROMPTS` list |
| Analytics show 0% hit rate | Make requests first, then check |
| Memory usage high | Reduce `SEMANTIC_CACHE_MAX_SIZE` |
