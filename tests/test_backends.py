"""Backend abstraction tests: adapters, factory, and model swappability.

Model swappability is the point of the abstraction: changing the model in
router.yaml (config, not code) must route to the new model. The swap test
runs a different router.yaml through ``proxy.derive_route_state`` — the
same pure function the gateway uses at startup — with no module reload.
"""
import httpx
import pytest

import backends
from backends import OMLXBackend, OpenAIBackend, build_backend

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_build_backend_omlx_defaults():
    backend = build_backend({"type": "omlx", "upstream": "http://127.0.0.1:8000"})
    assert isinstance(backend, OMLXBackend)
    assert backend.chat_url() == "http://127.0.0.1:8000/v1/chat/completions"
    assert backend.admin == "http://127.0.0.1:8000/admin"
    assert backend.supports_residency is True
    assert backend.headers() == {}


def test_build_backend_openai():
    backend = build_backend({"type": "openai", "upstream": "http://127.0.0.1:8100/"})
    assert isinstance(backend, OpenAIBackend)
    assert backend.chat_url() == "http://127.0.0.1:8100/v1/chat/completions"
    assert backend.supports_residency is False


def test_build_backend_api_key_header():
    backend = build_backend({"type": "openai", "upstream": "http://x", "api_key": "sekret"})
    assert backend.headers() == {"Authorization": "Bearer sekret"}


def test_build_backend_unknown_type():
    with pytest.raises(ValueError, match="Unknown backend type"):
        build_backend({"type": "vllm", "upstream": "http://x"})


def test_build_backend_missing_upstream():
    with pytest.raises(ValueError, match="requires a non-empty 'upstream'"):
        build_backend({"type": "omlx"})


# ---------------------------------------------------------------------------
# OMLXBackend against a fake oMLX server
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_omlx(monkeypatch):
    """Fake oMLX: /admin/api/models + /v1/models + /v1/chat/completions."""
    state = {"models": {"model-a": {"loaded": True, "is_loading": False}},
             "actions": []}

    def handler(request):
        path = request.url.path
        if path == "/admin/api/models" and request.method == "GET":
            return httpx.Response(200, json={"models": [
                {"id": mid, **meta} for mid, meta in state["models"].items()
            ]})
        if path.startswith("/admin/api/models/") and request.method == "POST":
            mid, action = path.split("/")[-2:]
            state["actions"].append((mid, action))
            state["models"][mid]["loaded"] = action == "load"
            return httpx.Response(200, json={"ok": True})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [
                {"id": mid} for mid in state["models"]
            ]})
        if path == "/v1/chat/completions":
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        return httpx.Response(404)

    client_class = httpx.AsyncClient
    monkeypatch.setattr(backends.httpx, "AsyncClient",
                        lambda **kw: client_class(transport=httpx.MockTransport(handler), **kw))
    return state


@pytest.mark.asyncio
async def test_omlx_list_models(fake_omlx):
    backend = OMLXBackend("http://fake")
    models = await backend.list_models()
    assert models == {"model-a": {"loaded": True, "is_loading": False}}


@pytest.mark.asyncio
async def test_omlx_load_unload(fake_omlx):
    backend = OMLXBackend("http://fake")
    await backend.unload_model("model-a")
    assert fake_omlx["actions"] == [("model-a", "unload")]
    assert (await backend.list_models())["model-a"]["loaded"] is False
    await backend.load_model("model-a")
    assert fake_omlx["actions"] == [("model-a", "unload"), ("model-a", "load")]
    assert (await backend.list_models())["model-a"]["loaded"] is True


@pytest.mark.asyncio
async def test_omlx_health_check(fake_omlx):
    assert await OMLXBackend("http://fake").health_check() is True


@pytest.mark.asyncio
async def test_omlx_health_check_down(monkeypatch):
    def handler(request):
        return httpx.Response(500)
    client_class = httpx.AsyncClient
    monkeypatch.setattr(backends.httpx, "AsyncClient",
                        lambda **kw: client_class(transport=httpx.MockTransport(handler), **kw))
    assert await OMLXBackend("http://fake").health_check() is False


# ---------------------------------------------------------------------------
# OpenAIBackend against a fake OpenAI-compatible server
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_list_models(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})
    client_class = httpx.AsyncClient
    monkeypatch.setattr(backends.httpx, "AsyncClient",
                        lambda **kw: client_class(transport=httpx.MockTransport(handler), **kw))
    backend = OpenAIBackend("http://fake")
    models = await backend.list_models()
    # Generic servers have no load state: listed means loaded.
    assert models == {"m1": {"loaded": True, "is_loading": False},
                      "m2": {"loaded": True, "is_loading": False}}


@pytest.mark.asyncio
async def test_openai_no_residency_api():
    backend = OpenAIBackend("http://fake")
    with pytest.raises(NotImplementedError):
        await backend.load_model("m1")
    with pytest.raises(NotImplementedError):
        await backend.unload_model("m1")


# ---------------------------------------------------------------------------
# Model swappability: config change, no code change
# ---------------------------------------------------------------------------

SWAPPED_CONFIG = """\
version: 3
backends:
  moe:
    type: omlx
    upstream: http://127.0.0.1:8000
  dense:
    type: omlx
    upstream: http://127.0.0.1:8000
routes:
  moe:
    model: swapped-in--NewModel-8B
    capabilities:
      context_window: 32000
      max_output: 8000
      vision: false
  dense:
    model: scottlowry--Qwen3.8-27B-oQ6e-mtp
    capabilities:
      context_window: 128000
      max_output: 32000
      vision: false
routing:
  fallback_route: moe
"""


def test_model_swap_via_config(tmp_path, monkeypatch):
    """Changing the model in router.yaml routes to the new model.

    Runs a config with a different moe model through the same derivation
    the gateway uses at startup. No proxy.py code changes are involved.
    """
    import proxy

    config_path = tmp_path / "router.yaml"
    config_path.write_text(SWAPPED_CONFIG)
    monkeypatch.setattr(proxy, "ROUTER_CONFIG_PATH", str(config_path))
    config = proxy.load_router_config()
    state = proxy.derive_route_state(config)

    assert state["model_ids"]["moe"] == "swapped-in--NewModel-8B"
    assert state["model_ids"]["dense"] == "scottlowry--Qwen3.8-27B-oQ6e-mtp"
    # The swapped model is what the route targets (alias -> route).
    assert state["model_routes"]["swapped-in--NewModel-8B"] == "moe"
    # Capability metadata follows the config too.
    assert state["model_capabilities"]["moe"]["context_window"] == 32000
    assert state["model_capabilities"]["moe"]["max_output"] == 8000
    # Backends are built from the config as well.
    assert all(isinstance(b, OMLXBackend) for b in state["backends"].values())


def test_backend_config_per_route(tmp_path, monkeypatch):
    """Routes can point at different backend types/upstreams from config."""
    import proxy

    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        "version: 3\n"
        "backends:\n"
        "  moe:\n"
        "    type: omlx\n"
        "    upstream: http://127.0.0.1:8000\n"
        "  dense:\n"
        "    type: openai\n"
        "    upstream: http://127.0.0.1:8100\n"
        "routes:\n"
        "  moe:\n"
        "    model: m1\n"
        "  dense:\n"
        "    model: m2\n"
        "routing:\n"
        "  fallback_route: moe\n"
    )
    monkeypatch.setattr(proxy, "ROUTER_CONFIG_PATH", str(config_path))
    config = proxy.load_router_config()
    state = proxy.derive_route_state(config)

    assert isinstance(state["backends"]["moe"], OMLXBackend)
    assert isinstance(state["backends"]["dense"], OpenAIBackend)
    # Different upstreams -> distinct backend instances.
    assert state["backends"]["moe"] is not state["backends"]["dense"]
    assert state["endpoints"]["dense"] == "http://127.0.0.1:8100"


def test_env_override_model_id(monkeypatch):
    """MOE_MODEL / DENSE_MODEL env vars still override the config."""
    import proxy

    monkeypatch.setenv("MOE_MODEL", "env--Override-Model")
    state = proxy.derive_route_state(proxy.ROUTER_CONFIG)
    assert state["model_ids"]["moe"] == "env--Override-Model"
    assert state["model_ids"]["dense"] == proxy.DENSE_MODEL
