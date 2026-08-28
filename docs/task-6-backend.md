# Task 6: Generic Backend Abstraction

## Overview

Make the gateway work with any model-hosting framework (oMLX, MLX, LLaMA, vLLM, TGI) by abstracting the backend-specific logic. This makes the gateway portable and future-proof.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Generic Backend Abstraction                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Gateway Core (proxy.py)                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │   │
│  │  │  Router      │  │  Cache       │  │  Metrics             │  │   │
│  │  │  Logic       │  │  Layer       │  │  & Analytics         │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │   │
│  │         │                    │                    │              │   │
│  │         └────────────────────┴────────────────────┘              │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │              Backend Abstraction Layer                    │   │   │
│  │  │                                                           │   │   │
│  │  │  ┌─────────────────────────────────────────────────┐   │   │   │
│  │  │  │  BackendInterface (abstract)                     │   │   │   │
│  │  │  │  • list_models()                                 │   │   │   │
│  │  │  │  • load_model(model_id)                          │   │   │   │
│  │  │  │  • unload_model(model_id)                        │   │   │   │
│  │  │  │  • chat_completion(request)                      │   │   │   │
│  │  │  │  • health_check()                                │   │   │   │
│  │  │  └─────────────────────────────────────────────────┘   │   │   │
│  │  │         │                    │                    │      │   │   │
│  │  │         ▼                    ▼                    ▼      │   │   │
│  │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │   │   │
│  │  │  │  oMLX       │    │  MLX        │    │  LLaMA      │  │   │   │
│  │  │  │  Backend    │    │  Backend    │    │  Backend    │  │   │   │
│  │  │  └─────────────┘    └─────────────┘    └─────────────┘  │   │   │
│  │  │                                                           │   │   │
│  │  │  ┌─────────────┐    ┌─────────────┐                     │   │   │
│  │  │  │  vLLM       │    │  TGI        │                     │   │   │
│  │  │  │  Backend    │    │  Backend    │                     │   │   │
│  │  │  └─────────────┘    └─────────────┘                     │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Configuration (router.yaml)                   │   │
│  │  backend: omllm  # or: mlx, llama, vllm, tgi                    │   │
│  │  upstream: http://127.0.0.1:8080                                │   │
│  │  models:                                                         │   │
│  │    moe: Qwen3.6-35B-A3B-oQ5e-mtp                                │   │
│  │    dense: Qwen3.8-27B-oQ4e-mtp                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
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
│  │  Create     │    │  Test with  │    │  Verify     │                │
│  │  backend    │    │  multiple   │    │  config     │                │
│  │  interface  │    │  backends   │    │  switching  │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        └────────────────────┴────────────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  QA Agent       │                                │
│                     │  (verify        │                                │
│                     │   compatibility)│                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Create backend interface, implement adapters | `write`, `edit` |
| **Tester** | Test with multiple backends (mock + real) | `bash` (pytest, curl) |
| **Engineer** | Verify config switching works | `bash` (git, curl) |
| **QA** | Check edge cases, error handling | `read`, `bash` |

## Step-by-Step Execution

### Step 1: Define Backend Interface

**backend_interface.py** (new file):

```python
"""
Abstract backend interface for model hosting frameworks.
"""
import abc
from typing import Any, AsyncIterator, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    id: str
    loaded: bool
    is_loading: bool = False
    size_gb: Optional[float] = None
    quantization: Optional[str] = None


@dataclass
class ChatRequest:
    """Standardized chat completion request."""
    model: str
    messages: list[dict]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    extra: dict = None  # Backend-specific parameters


@dataclass
class ChatResponse:
    """Standardized chat completion response."""
    id: str
    model: str
    content: str
    finish_reason: str
    usage: dict
    raw: dict  # Original response for debugging


class BackendInterface(abc.ABC):
    """Abstract interface for model hosting backends."""
    
    @abc.abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List all available models."""
        ...
    
    @abc.abstractmethod
    async def load_model(self, model_id: str) -> bool:
        """Load a model into memory."""
        ...
    
    @abc.abstractmethod
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        ...
    
    @abc.abstractmethod
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Execute a chat completion request."""
        ...
    
    @abc.abstractmethod
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[dict]:
        """Execute a streaming chat completion request."""
        ...
    
    @abc.abstractmethod
    async def health_check(self) -> dict:
        """Check backend health status."""
        ...
    
    @abc.abstractmethod
    def name(self) -> str:
        """Return backend name."""
        ...
```

### Step 2: Implement oMLX Backend

**backends/omlx.py** (new file):

```python
"""
oMLX backend implementation.
"""
import httpx
from typing import AsyncIterator
from backend_interface import BackendInterface, ModelInfo, ChatRequest, ChatResponse


class OMLXBackend(BackendInterface):
    """Backend for oMLX (Open Model LLaMA eXtended)."""
    
    def __init__(self, upstream: str, timeout: float = 300.0):
        self.upstream = upstream.rstrip("/")
        self.timeout = timeout
    
    async def list_models(self) -> list[ModelInfo]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.upstream}/admin/api/models")
            response.raise_for_status()
            data = response.json()
            return [
                ModelInfo(
                    id=m["id"],
                    loaded=m.get("loaded", False),
                    is_loading=m.get("is_loading", False),
                )
                for m in data.get("models", [])
            ]
    
    async def load_model(self, model_id: str) -> bool:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.upstream}/admin/api/models/{model_id}/load"
            )
            return response.status_code == 200
    
    async def unload_model(self, model_id: str) -> bool:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.upstream}/admin/api/models/{model_id}/unload"
            )
            return response.status_code == 200
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        body = {
            "model": request.model,
            "messages": request.messages,
            "stream": False,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            **(request.extra or {}),
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.upstream}/v1/chat/completions", json=body
            )
            response.raise_for_status()
            data = response.json()
        
        choice = data["choices"][0]
        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=choice["message"]["content"],
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            raw=data,
        )
    
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[dict]:
        body = {
            "model": request.model,
            "messages": request.messages,
            "stream": True,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            **(request.extra or {}),
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.upstream}/v1/chat/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield dict(data)
    
    async def health_check(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.upstream}/health")
                return {
                    "ok": response.status_code == 200,
                    "backend": "omlx",
                    "upstream": self.upstream,
                }
        except Exception as exc:
            return {"ok": False, "backend": "omlx", "error": str(exc)}
    
    def name(self) -> str:
        return "omlx"
```

### Step 3: Implement MLX Backend

**backends/mlx.py** (new file):

```python
"""
MLX backend implementation (direct MLX server).
"""
import httpx
from typing import AsyncIterator
from backend_interface import BackendInterface, ModelInfo, ChatRequest, ChatResponse


class MLXBackend(BackendInterface):
    """Backend for direct MLX server."""
    
    def __init__(self, upstream: str, timeout: float = 300.0):
        self.upstream = upstream.rstrip("/")
        self.timeout = timeout
    
    async def list_models(self) -> list[ModelInfo]:
        # MLX server typically has a single model
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.upstream}/v1/models")
            response.raise_for_status()
            data = response.json()
            return [
                ModelInfo(
                    id=m["id"],
                    loaded=True,
                )
                for m in data.get("data", [])
            ]
    
    async def load_model(self, model_id: str) -> bool:
        # MLX server doesn't support dynamic loading
        return True
    
    async def unload_model(self, model_id: str) -> bool:
        # MLX server doesn't support dynamic unloading
        return True
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        body = {
            "model": request.model,
            "messages": request.messages,
            "stream": False,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.upstream}/v1/chat/completions", json=body
            )
            response.raise_for_status()
            data = response.json()
        
        choice = data["choices"][0]
        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=choice["message"]["content"],
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            raw=data,
        )
    
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[dict]:
        body = {
            "model": request.model,
            "messages": request.messages,
            "stream": True,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.upstream}/v1/chat/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield dict(data)
    
    async def health_check(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.upstream}/v1/models")
                return {
                    "ok": response.status_code == 200,
                    "backend": "mlx",
                    "upstream": self.upstream,
                }
        except Exception as exc:
            return {"ok": False, "backend": "mlx", "error": str(exc)}
    
    def name(self) -> str:
        return "mlx"
```

### Step 4: Implement LLaMA Backend

**backends/llama.py** (new file):

```python
"""
LLaMA backend implementation (llama.cpp server).
"""
import httpx
from typing import AsyncIterator
from backend_interface import BackendInterface, ModelInfo, ChatRequest, ChatResponse


class LLaMABackend(BackendInterface):
    """Backend for llama.cpp server."""
    
    def __init__(self, upstream: str, timeout: float = 300.0):
        self.upstream = upstream.rstrip("/")
        self.timeout = timeout
    
    async def list_models(self) -> list[ModelInfo]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.upstream}/v1/models")
            response.raise_for_status()
            data = response.json()
            return [
                ModelInfo(
                    id=m["id"],
                    loaded=True,
                )
                for m in data.get("data", [])
            ]
    
    async def load_model(self, model_id: str) -> bool:
        return True
    
    async def unload_model(self, model_id: str) -> bool:
        return True
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        body = {
            "model": request.model,
            "messages": request.messages,
            "stream": False,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.upstream}/v1/chat/completions", json=body
            )
            response.raise_for_status()
            data = response.json()
        
        choice = data["choices"][0]
        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=choice["message"]["content"],
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            raw=data,
        )
    
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[dict]:
        body = {
            "model": request.model,
            "messages": request.messages,
            "stream": True,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.upstream}/v1/chat/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield dict(data)
    
    async def health_check(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.upstream}/v1/models")
                return {
                    "ok": response.status_code == 200,
                    "backend": "llama",
                    "upstream": self.upstream,
                }
        except Exception as exc:
            return {"ok": False, "backend": "llama", "error": str(exc)}
    
    def name(self) -> str:
        return "llama"
```

### Step 5: Create Backend Factory

**backends/__init__.py** (new file):

```python
"""
Backend factory for creating backend instances.
"""
from backend_interface import BackendInterface
from backends.omlx import OMLXBackend
from backends.mlx import MLXBackend
from backends.llama import LLaMABackend


def create_backend(backend_type: str, upstream: str, timeout: float = 300.0) -> BackendInterface:
    """Create a backend instance based on type."""
    backends = {
        "omlx": OMLXBackend,
        "mlx": MLXBackend,
        "llama": LLaMABackend,
        # Add more backends here:
        # "vllm": VLLMBackend,
        # "tgi": TGIBackend,
    }
    
    if backend_type not in backends:
        raise ValueError(f"Unknown backend type: {backend_type}. Available: {list(backends.keys())}")
    
    return backends[backend_type](upstream, timeout)
```

### Step 6: Update proxy.py to Use Backend Interface

**proxy.py** — Replace direct oMLX calls with backend interface:

```python
# At the top of proxy.py
from backends import create_backend

# Initialize backend based on config
BACKEND_TYPE = os.environ.get("GATEWAY_BACKEND", "omlx")
backend = create_backend(BACKEND_TYPE, OMLX_UPSTREAM, ROUTER_TIMEOUT_SEC)

# Replace direct oMLX calls with backend calls
# Before:
# async with httpx.AsyncClient(timeout=timeout) as client:
#     response = await client.post(f"{OMLX_UPSTREAM}/v1/chat/completions", json=request_body)

# After:
request = ChatRequest(
    model=ROUTER_MODEL,
    messages=[{"role": "user", "content": prompt}],
    stream=False,
    temperature=ROUTER_TEMPERATURE,
    max_tokens=ROUTER_MAX_TOKENS,
)
response = await backend.chat_completion(request)
```

### Step 7: Update Configuration

**router.yaml** — Add backend configuration:

```yaml
# Backend configuration
backend:
  type: omllm  # or: mlx, llama, vllm, tgi
  upstream: http://127.0.0.1:8080
  timeout: 300.0

# Model configuration
models:
  moe: Qwen3.6-35B-A3B-oQ5e-mtp
  dense: Qwen3.8-27B-oQ4e-mtp
  router: Arch-Router-1.5B-mlx-8Bit
```

### Step 8: Test Backend Abstraction

**tests/test_backends.py**:

```python
import pytest
from backend_interface import BackendInterface, ModelInfo, ChatRequest, ChatResponse
from backends import create_backend
from backends.omlx import OMLXBackend
from backends.mlx import MLXBackend
from backends.llama import LLaMABackend

class TestBackendFactory:
    def test_create_omlx_backend(self):
        backend = create_backend("omlx", "http://localhost:8080")
        assert isinstance(backend, OMLXBackend)
        assert backend.name() == "omlx"

    def test_create_mlx_backend(self):
        backend = create_backend("mlx", "http://localhost:8080")
        assert isinstance(backend, MLXBackend)
        assert backend.name() == "mlx"

    def test_create_llama_backend(self):
        backend = create_backend("llama", "http://localhost:8080")
        assert isinstance(backend, LLaMABackend)
        assert backend.name() == "llama"

    def test_create_unknown_backend(self):
        with pytest.raises(ValueError):
            create_backend("unknown", "http://localhost:8080")

class TestBackendInterface:
    def test_omlx_implements_interface(self):
        backend = OMLXBackend("http://localhost:8080")
        assert isinstance(backend, BackendInterface)
        # Verify all abstract methods are implemented
        assert hasattr(backend, "list_models")
        assert hasattr(backend, "load_model")
        assert hasattr(backend, "unload_model")
        assert hasattr(backend, "chat_completion")
        assert hasattr(backend, "chat_completion_stream")
        assert hasattr(backend, "health_check")
        assert hasattr(backend, "name")

    def test_mlx_implements_interface(self):
        backend = MLXBackend("http://localhost:8080")
        assert isinstance(backend, BackendInterface)

    def test_llama_implements_interface(self):
        backend = LLaMABackend("http://localhost:8080")
        assert isinstance(backend, BackendInterface)
```

### Step 9: Verify Locally

```bash
# Test with oMLX backend (default)
GATEWAY_BACKEND=omlx python proxy.py

# Test with MLX backend
GATEWAY_BACKEND=mlx OMLX_UPSTREAM=http://localhost:8080 python proxy.py

# Test with LLaMA backend
GATEWAY_BACKEND=llama OMLX_UPSTREAM=http://localhost:8080 python proxy.py

# Run tests
pytest tests/test_backends.py -v
```

## Success Criteria

- [ ] Backend interface defined with all required methods
- [ ] oMLX, MLX, LLaMA backends implemented
- [ ] Backend factory creates correct backend based on config
- [ ] proxy.py uses backend interface (no direct oMLX calls)
- [ ] Configuration supports backend switching
- [ ] Tests pass for all backends
- [ ] Gateway works with any backend via config

## Commands Reference

```bash
# List available backends
python -c "from backends import create_backend; print('Available: omlx, mlx, llama')"

# Test backend creation
python -c "from backends import create_backend; b = create_backend('omlx', 'http://localhost:8080'); print(b.name())"

# Run with specific backend
GATEWAY_BACKEND=mlx python proxy.py

# Run tests
pytest tests/test_backends.py -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend not found | Check `GATEWAY_BACKEND` env var |
| Connection refused | Verify upstream server is running |
| Model not loading | Check backend-specific load API |
| Streaming fails | Verify backend supports SSE |
