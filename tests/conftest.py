"""
Test fixtures for the inference gateway.
"""
import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_chat_request():
    """Sample chat completion request."""
    return {
        "model": "gateway-auto",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "max_tokens": 100,
    }


@pytest.fixture
def sample_coding_request():
    """Sample coding request."""
    return {
        "model": "gateway-auto",
        "messages": [{"role": "user", "content": "Build a Python API for user management"}],
        "stream": False,
        "max_tokens": 1000,
    }


@pytest.fixture
def sample_complex_request():
    """Sample complex request that should route to dense."""
    return {
        "model": "gateway-auto",
        "messages": [{"role": "user", "content": "Review the entire codebase for architectural issues and propose a redesign"}],
        "stream": False,
        "max_tokens": 2000,
    }


@pytest.fixture
def sample_stream_request():
    """Sample streaming request."""
    return {
        "model": "gateway-auto",
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "stream": True,
        "max_tokens": 500,
    }
