"""Model switching against a fake oMLX server; never touches live models."""
import httpx
import pytest

import proxy


@pytest.fixture(autouse=True)
def single_backend_mode(monkeypatch):
    monkeypatch.setattr(proxy, "KEEP_MODELS_LOADED", False)


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['moe', 'dense'])
@pytest.mark.parametrize('both_loaded', [False, True])
async def test_single_backend_transition(monkeypatch, route, both_loaded):
    target = proxy.MODEL_IDS[route]
    other = proxy.MODEL_IDS['dense' if route == 'moe' else 'moe']
    states = {target: both_loaded, other: True}
    actions = []

    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'models': [
                {'id': mid, 'loaded': loaded} for mid, loaded in states.items()
            ]})
        mid, action = request.url.path.split('/')[-2:]
        actions.append((mid, action))
        states[mid] = action == 'load'
        return httpx.Response(200, json={'ok': True})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, 'AsyncClient', lambda **kw: client_class(
        transport=httpx.MockTransport(handler), **kw))
    await proxy.ensure_route(route)
    assert states == {target: True, other: False}
    assert actions == [(other, 'unload')] + ([] if both_loaded else [(target, 'load')])


@pytest.mark.asyncio
async def test_admin_failure_does_not_switch(monkeypatch):
    requests = []
    def handler(request):
        requests.append(request.method)
        return httpx.Response(401)
    client_class = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, 'AsyncClient', lambda **kw: client_class(
        transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(httpx.HTTPStatusError):
        await proxy.ensure_route('moe')
    assert requests == ['GET']


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['moe', 'dense'])
async def test_queued_unload_finishes_before_target_load(monkeypatch, route):
    target = proxy.MODEL_IDS[route]
    other = proxy.MODEL_IDS['dense' if route == 'moe' else 'moe']
    states = {target: False, other: True}
    remaining = None
    actions = []

    def handler(request):
        nonlocal remaining
        if request.method == 'GET':
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    states[other] = False
            return httpx.Response(200, json={'models': [
                {'id': mid, 'loaded': loaded} for mid, loaded in states.items()
            ]})
        mid, action = request.url.path.split('/')[-2:]
        actions.append((mid, action))
        if action == 'unload':
            remaining = 3
            return httpx.Response(202, json={'status': 'unloading'})
        assert not states[other], 'Never load target before queued unload completes'
        states[mid] = True
        return httpx.Response(200, json={'status': 'ok'})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, 'AsyncClient', lambda **kw: client_class(
        transport=httpx.MockTransport(handler), **kw))
    await proxy.ensure_route(route)
    assert actions == [(other, 'unload'), (target, 'load')]
    assert states[target] and not states[other]


@pytest.mark.asyncio
async def test_unload_timeout_never_loads_target(monkeypatch):
    actions = []
    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'models': [
                {'id': proxy.MOE_MODEL, 'loaded': True},
                {'id': proxy.DENSE_MODEL, 'loaded': False},
            ]})
        actions.append(request.url.path)
        return httpx.Response(202, json={'status': 'unloading'})
    client_class = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, 'AsyncClient', lambda **kw: client_class(
        transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(proxy, 'SWAP_TIMEOUT_SEC', 0)
    with pytest.raises(TimeoutError, match='loaded=False'):
        await proxy.ensure_route('dense')
    assert len(actions) == 1 and actions[0].endswith('/unload')


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['moe', 'dense'])
async def test_resident_mode_never_unloads(monkeypatch, route):
    monkeypatch.setattr(proxy, 'KEEP_MODELS_LOADED', True)
    target = proxy.MODEL_IDS[route]
    states = {mid: mid != target for mid in proxy.MODEL_IDS.values()}
    actions = []
    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'models': [
                {'id': mid, 'loaded': loaded} for mid, loaded in states.items()]})
        actions.append(request.url.path)
        assert request.url.path.endswith('/load')
        states[target] = True
        return httpx.Response(200, json={'status': 'ok'})
    client_class = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, 'AsyncClient', lambda **kw: client_class(
        transport=httpx.MockTransport(handler), **kw))
    await proxy.ensure_route(route)
    await proxy.ensure_route(route)
    assert all(states.values())
    assert len(actions) == 1
