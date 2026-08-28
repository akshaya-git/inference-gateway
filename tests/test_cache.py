"""
Tests for cache functionality.
"""
import time

from proxy import (
    CACHE_ENABLED,
    CACHE_MAX_ENTRIES,
    CACHE_TTL_SEC,
    cache_get,
    cache_key,
    cache_put,
    cache_stats,
    response_cache,
)


class TestCache:
    """Tests for cache behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        response_cache.clear()
        cache_stats.update({"hits": 0, "misses": 0, "stores": 0, "evictions": 0})

    def test_cache_miss(self):
        """Cache miss should return None."""
        result = cache_get("nonexistent")
        assert result is None
        assert cache_stats["misses"] == 1

    def test_cache_hit(self):
        """Cache hit should return stored payload."""
        key = cache_key({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
        cache_put(key, {"data": "value"}, 200)
        result = cache_get(key)
        assert result is not None
        assert result["payload"] == {"data": "value"}
        assert cache_stats["hits"] == 1

    def test_cache_put_with_invalid_status(self):
        """Cache should not store responses with invalid status codes."""
        key = "test_key"
        cache_put(key, {"data": "value"}, 500)
        result = cache_get(key)
        assert result is None

    def test_cache_put_with_success_status(self):
        """Cache should store responses with 2xx status codes."""
        key = "test_key"
        cache_put(key, {"data": "value"}, 200)
        result = cache_get(key)
        assert result is not None

    def test_cache_eviction(self):
        """Cache should evict oldest entry when full."""
        # Fill cache to capacity
        for i in range(CACHE_MAX_ENTRIES):
            key = f"key_{i}"
            cache_put(key, {"data": f"value_{i}"}, 200)

        # Cache should be full
        assert len(response_cache) == CACHE_MAX_ENTRIES

        # Add one more to trigger eviction
        cache_put("new_key", {"data": "new_value"}, 200)

        # Oldest entry should be evicted
        assert "key_0" not in response_cache
        assert "new_key" in response_cache

    def test_cache_ttl_expiry(self):
        """Cache entries should expire after TTL."""
        key = "test_key"
        cache_put(key, {"data": "value"}, 200)
        assert cache_get(key) is not None

        # Manually expire the entry
        response_cache[key]["stored_at"] = time.time() - CACHE_TTL_SEC - 1
        result = cache_get(key)
        assert result is None
        assert cache_stats["evictions"] == 1

    def test_cache_disabled(self):
        """Cache should return None when disabled."""
        global CACHE_ENABLED
        original = CACHE_ENABLED
        CACHE_ENABLED = False
        try:
            result = cache_get("any_key")
            assert result is None
        finally:
            CACHE_ENABLED = original

    def test_cache_stats_tracking(self):
        """Cache should track hits, misses, and stores."""
        key = cache_key({"model": "test", "messages": [{"role": "user", "content": "hi"}]})

        # Miss
        cache_get(key)
        assert cache_stats["misses"] == 1

        # Store
        cache_put(key, {"data": "value"}, 200)
        assert cache_stats["stores"] == 1

        # Hit
        cache_get(key)
        assert cache_stats["hits"] == 1
