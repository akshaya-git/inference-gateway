"""
Tests for memory guard functionality.
"""
from proxy import (
    MEMORY_GUARD_GB,
    MEMORY_HARD_GB,
    memory_state,
    update_memory_state,
)


class TestMemoryGuard:
    """Tests for memory guard functionality."""

    def test_memory_state_update(self):
        """Should update memory state with current values."""
        update_memory_state()
        assert memory_state["available_gb"] > 0
        assert memory_state["total_gb"] > 0
        assert memory_state["used_gb"] >= 0
        assert memory_state["percent"] >= 0

    def test_memory_state_totals(self):
        """Available + used should approximately equal total."""
        update_memory_state()
        total = memory_state["available_gb"] + memory_state["used_gb"]
        assert abs(total - memory_state["total_gb"]) < 1.0  # Allow 1GB tolerance

    def test_memory_pressure_threshold(self):
        """Should detect memory pressure when available < guard_gb."""
        update_memory_state()
        if memory_state["available_gb"] <= MEMORY_GUARD_GB:
            assert memory_state["pressure"] is True
        else:
            assert memory_state["pressure"] is False

    def test_memory_hard_pressure_threshold(self):
        """Should detect hard pressure when available < hard_gb."""
        update_memory_state()
        if memory_state["available_gb"] <= MEMORY_HARD_GB:
            assert memory_state["hard_pressure"] is True
        else:
            assert memory_state["hard_pressure"] is False

    def test_memory_state_timestamp(self):
        """Should update last_update timestamp."""
        update_memory_state()
        assert memory_state["last_update"] > 0
        assert abs(memory_state["last_update"] - __import__('time').time()) < 1.0

    def test_memory_state_percent(self):
        """Percent should be between 0 and 100."""
        update_memory_state()
        assert 0 <= memory_state["percent"] <= 100
