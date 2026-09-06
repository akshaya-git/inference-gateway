"""Backend abstraction layer for model-hosting frameworks.

The gateway talks to model servers through ``BackendInterface``. Two
implementations ship today:

- ``OMLXBackend``: oMLX — OpenAI-compatible chat plus an ``/admin``
  residency API (query / load / unload models at runtime).
- ``OpenAIBackend``: any generic OpenAI-compatible server (MLX
  ``mlx_lm.server``, llama.cpp ``llama-server``, Ollama, LM Studio).
  No admin residency: models are fixed at server start, so
  ``load_model`` / ``unload_model`` raise ``NotImplementedError``.

Adding a new framework means adding a new backend class (or reusing
``OpenAIBackend``) and a ``backends:`` entry in ``router.yaml`` — no
changes to the routing, caching, or metrics layers.
"""
from __future__ import annotations

import abc
from typing import Any

import httpx


class BackendError(RuntimeError):
    """Backend admin or health failure."""


class BackendInterface(abc.ABC):
    """Abstract model-hosting backend."""

    name: str = "abstract"

    def __init__(self, upstream: str, api_key: str = ""):
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key

    # -- inference ---------------------------------------------------------

    @abc.abstractmethod
    def chat_url(self) -> str:
        """Full URL of the OpenAI-compatible /v1/chat/completions endpoint."""

    def headers(self) -> dict[str, str]:
        """Headers to send to the upstream (auth, etc.)."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    # -- residency ---------------------------------------------------------

    @property
    def supports_residency(self) -> bool:
        """True if the backend can load/unload models at runtime."""
        return False

    @abc.abstractmethod
    async def list_models(self) -> dict[str, dict[str, Any]]:
        """Return ``{model_id: {"loaded": bool, "is_loading": bool}}``."""

    async def load_model(self, model_id: str) -> None:
        """Load a model. Raise NotImplementedError if unsupported."""
        raise NotImplementedError(f"{self.name} backend has no admin load API")

    async def unload_model(self, model_id: str) -> None:
        """Unload a model. Raise NotImplementedError if unsupported."""
        raise NotImplementedError(f"{self.name} backend has no admin unload API")

    # -- health ------------------------------------------------------------

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """True if the backend server is reachable and serving models."""


class OMLXBackend(BackendInterface):
    """oMLX: OpenAI-compatible chat + ``/admin/api/models`` residency."""

    name = "omlx"

    def __init__(self, upstream: str, api_key: str = "", admin_path: str = "/admin"):
        super().__init__(upstream, api_key)
        self.admin = f"{self.upstream}{admin_path}"

    def chat_url(self) -> str:
        return f"{self.upstream}/v1/chat/completions"

    @property
    def supports_residency(self) -> bool:
        return True

    async def list_models(self) -> dict[str, dict[str, Any]]:
        async with httpx.AsyncClient(timeout=5.0, headers=self.headers()) as client:
            response = await client.get(f"{self.admin}/api/models")
            response.raise_for_status()
            data = response.json()
        return {
            item["id"]: {
                "loaded": bool(item.get("loaded")),
                "is_loading": bool(item.get("is_loading")),
            }
            for item in data.get("models", [])
        }

    async def load_model(self, model_id: str) -> None:
        async with httpx.AsyncClient(timeout=30.0, headers=self.headers()) as client:
            response = await client.post(f"{self.admin}/api/models/{model_id}/load")
            response.raise_for_status()

    async def unload_model(self, model_id: str) -> None:
        async with httpx.AsyncClient(timeout=30.0, headers=self.headers()) as client:
            response = await client.post(f"{self.admin}/api/models/{model_id}/unload")
            response.raise_for_status()

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0, headers=self.headers()) as client:
                response = await client.get(f"{self.upstream}/v1/models")
                return response.status_code == 200
        except httpx.HTTPError:
            return False


class OpenAIBackend(BackendInterface):
    """Generic OpenAI-compatible server (MLX, llama.cpp, Ollama, LM Studio).

    Models are fixed at server start; there is no runtime residency API.
    A model listed by ``/v1/models`` is treated as loaded.
    """

    name = "openai"

    def chat_url(self) -> str:
        return f"{self.upstream}/v1/chat/completions"

    async def list_models(self) -> dict[str, dict[str, Any]]:
        async with httpx.AsyncClient(timeout=5.0, headers=self.headers()) as client:
            response = await client.get(f"{self.upstream}/v1/models")
            response.raise_for_status()
            data = response.json()
        return {
            item["id"]: {"loaded": True, "is_loading": False}
            for item in data.get("data", [])
        }

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0, headers=self.headers()) as client:
                response = await client.get(f"{self.upstream}/v1/models")
                return response.status_code == 200
        except httpx.HTTPError:
            return False


BACKEND_TYPES: dict[str, type[BackendInterface]] = {
    "omlx": OMLXBackend,
    "openai": OpenAIBackend,
}


def build_backend(spec: dict[str, Any]) -> BackendInterface:
    """Build a backend from a ``router.yaml`` ``backends:`` entry.

    ``spec``: ``{"type": "omlx"|"openai", "upstream": "http://...",
    "api_key": "..."}``.
    """
    backend_type = str(spec.get("type", "omlx")).lower()
    if backend_type not in BACKEND_TYPES:
        raise ValueError(
            f"Unknown backend type: {backend_type!r} "
            f"(expected one of {sorted(BACKEND_TYPES)})"
        )
    upstream = str(spec.get("upstream", "") or "").strip()
    if not upstream:
        raise ValueError(f"backend {backend_type!r} requires a non-empty 'upstream'")
    api_key = str(spec.get("api_key", "") or "")
    return BACKEND_TYPES[backend_type](upstream=upstream, api_key=api_key)
