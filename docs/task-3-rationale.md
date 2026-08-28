# Task 3: Routing Rationale Capture

## Overview

Capture and store the judge model's reasoning for each routing decision. This enables analysis of *why* the judge chose a specific model, which is essential for improving routing accuracy over time.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Routing Rationale Capture                             │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  Client  │───▶│  Gateway     │───▶│  Judge Model │───▶│  Store   │  │
│  │  Request │    │  (proxy.py)  │    │  (1.5B MoE)  │    │  Rationale│  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│                    │                    │                    │           │
│                    │                    ▼                    │           │
│                    │           ┌─────────────────┐           │           │
│                    │           │  Judge Response │           │           │
│                    │           │  {              │           │           │
│                    │           │    route: "moe" │           │           │
│                    │           │    reason: "..."│           │           │
│                    │           │    confidence:  │           │           │
│                    │           │    raw: "..."   │           │           │
│                    │           │  }              │           │           │
│                    │           └─────────────────┘           │           │
│                    │                    │                    │           │
│                    ▼                    ▼                    ▼           │
│           ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│           │  Request        │  │  Metrics        │  │  Dashboard      ││
│           │  Metadata       │  │  (router_ms,    │  │  (Routing       ││
│           │  (task_type,    │  │   confidence)   │  │   Decisions     ││
│           │   effort)       │  │                 │  │   tab)          ││
│           └─────────────────┘  └─────────────────┘  └─────────────────┘│
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
│  │  Modify     │    │  Test with  │    │  Verify     │                │
│  │  judge call │    │  sample     │    │  dashboard  │                │
│  │  to capture │    │  requests   │    │  shows      │                │
│  │  rationale  │    │             │    │  rationale  │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        └────────────────────┴────────────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  QA Agent       │                                │
│                     │  (verify data   │                                │
│                     │   quality)      │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Modify judge call to capture raw response, add to Metric | `edit` |
| **Tester** | Test with sample requests, verify rationale captured | `bash` (curl, pytest) |
| **Engineer** | Verify dashboard displays rationale correctly | `bash` (curl) |
| **QA** | Check edge cases (empty response, timeout, etc.) | `read`, `bash` |

## Step-by-Step Execution

### Step 1: Modify Judge Call to Capture Raw Response

**proxy.py** — Update `arch_router_policy()`:

```python
async def arch_router_policy(
    body: dict, req_id: Optional[str] = None
) -> tuple[dict, float]:
    """Use the judge for every non-explicit inference request, without affinity."""
    started = now_ms()
    fallback = normalize_policy({
        "route": FALLBACK_ROUTE,
        "confidence": 0.0,
        "reason": f"configured fallback route: {FALLBACK_ROUTE}",
        "task_type": "router_fallback",
    }, heuristic_policy(body))
    if not ROUTING_ENABLED:
        return fallback, now_ms() - started

    try:
        async with router_transition_lock:
            await wait_for_backend_drain_before_routing(req_id)
            await ensure_router_loaded()
            conversation = router_request_context(body)
            states = model_state()
            runtime_state = {
                route: {
                    "resident": bool(states[route]["running"]),
                    "switch_required": not bool(states[route]["running"]),
                }
                for route in ("moe", "dense")
            }
            prompt = (
                "Select the best model route for the NEXT assistant completion represented by "
                "the complete OpenAI-style request below. Evaluate the unresolved objective, "
                "message roles, tool calls, tool results, and requested output—not the identity "
                "of the calling application. The resident model is a preference, not a mandate. "
                "Keep it when candidates are similarly capable, but switch when another route "
                "offers a meaningful quality or capability advantage.\n"
                f"<routes>\n{json.dumps(ARCH_ROUTES)}\n</routes>\n"
                f"<runtime_state>\n{json.dumps(runtime_state)}\n</runtime_state>\n"
                f"<conversation>\n{json.dumps(conversation)}\n</conversation>\n"
                "Select the route that best matches the latest user request. Return only the "
                "exact route name from the supplied routes."
            )
            request_body = {
                "model": ROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": ROUTER_MAX_TOKENS,
                "temperature": ROUTER_TEMPERATURE,
                "chat_template_kwargs": {"enable_thinking": ROUTER_THINKING},
                "stream": False,
            }
            timeout = httpx.Timeout(ROUTER_TIMEOUT_SEC)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{OMLX_UPSTREAM}/v1/chat/completions", json=request_body
                )
                response.raise_for_status()
                payload = response.json()
        
        # Extract judge response details
        message = (payload.get("choices") or [{}])[0].get("message") or {}
        raw = str(message.get("content") or "").strip()
        usage = payload.get("usage", {})
        
        # Parse route from response
        route_pattern = "|".join(
            re.escape(name) for name in sorted(JUDGE_ROUTE_NAMES, key=len, reverse=True)
        )
        match = re.search(rf"\b({route_pattern})\b", raw)
        if not match:
            raise ValueError(f"Arch-Router returned no valid route: {raw[:120]}")
        
        arch_route = match.group(1)
        route = JUDGE_ROUTE_NAMES[arch_route]
        
        # Capture full rationale
        decision = {
            "route": route,
            "confidence": 0.85,
            "reason": f"Arch-Router selected {arch_route}",
            "judge_raw": raw,  # NEW: Capture raw judge response
            "judge_tokens": usage.get("total_tokens", 0),  # NEW: Token usage
            "judge_latency_ms": now_ms() - started,  # NEW: Latency
        }
        policy = normalize_policy(decision, fallback)
        return policy, now_ms() - started
    except Exception as exc:
        fallback["reason"] = f"Arch-Router fallback ({type(exc).__name__}); {fallback['reason']}"
        return fallback, now_ms() - started
```

### Step 2: Add Rationale Fields to Metric Dataclass

**proxy.py** — Update `Metric` dataclass:

```python
@dataclass
class Metric:
    # ... existing fields ...
    route_reason: str = ""
    router_confidence: Optional[float] = None
    router_ms: Optional[float] = None
    # NEW fields for rationale capture
    judge_raw: str = ""  # Raw judge model response
    judge_tokens: Optional[int] = None  # Tokens used by judge
    judge_latency_ms: Optional[float] = None  # Judge inference latency
```

### Step 3: Update Record Calls to Include Rationale

**proxy.py** — Update `proxy_nonstream()` and `stream_request()`:

```python
# In proxy_nonstream():
metric = Metric(
    # ... existing fields ...
    route_reason=route_reason,
    router_confidence=confidence,
    router_ms=router_ms,
    # NEW: Add rationale fields
    judge_raw=policy.get("judge_raw", ""),
    judge_tokens=policy.get("judge_tokens"),
    judge_latency_ms=policy.get("judge_latency_ms"),
)

# In stream_request():
await record(Metric(
    # ... existing fields ...
    route_reason=route_reason,
    router_confidence=confidence,
    router_ms=router_ms,
    # NEW: Add rationale fields
    judge_raw=policy.get("judge_raw", ""),
    judge_tokens=policy.get("judge_tokens"),
    judge_latency_ms=policy.get("judge_latency_ms"),
))
```

### Step 4: Update Dashboard to Display Rationale

**proxy.py** — Update `DASHBOARD_V2` HTML:

```javascript
// In the routing decisions table, add expandable rationale
$('routeRows').innerHTML = h.slice().reverse().slice(0, 100).map(x => `
    <tr>
        <td>${new Date(x.ts*1000).toLocaleTimeString()}</td>
        <td>${esc(x.route)}</td>
        <td>${esc(x.task_type || '—')} · ${esc(x.effort || '—')}</td>
        <td>${x.router_confidence==null?'—':(x.router_confidence*100).toFixed(0)+'%'}</td>
        <td class=reason>${esc(x.route_reason)}</td>
        <td>${sec(x.router_ms)}</td>
        <td>${x.judge_tokens || '—'}</td>
        <td>${x.judge_latency_ms ? (x.judge_latency_ms/1000).toFixed(2)+'s' : '—'}</td>
        <td>
            ${x.judge_raw ? `<details><summary>Rationale</summary><pre>${esc(x.judge_raw)}</pre></details>` : '—'}
        </td>
    </tr>
`).join('') || '<tr><td colspan=9>No requests</td></tr>';
```

### Step 5: Add Rationale Export Endpoint

**proxy.py** — Add new endpoint:

```python
@app.get("/api/routing-rationale")
async def get_routing_rationale(limit: int = 100):
    """Export routing rationale for analysis."""
    async with lock:
        items = [
            {
                "ts": item["ts"],
                "route": item["route"],
                "route_reason": item["route_reason"],
                "confidence": item["router_confidence"],
                "judge_raw": item.get("judge_raw", ""),
                "judge_tokens": item.get("judge_tokens"),
                "judge_latency_ms": item.get("judge_latency_ms"),
                "task_type": item.get("task_type", ""),
                "effort": item.get("effort", ""),
            }
            for item in history
            if item.get("judge_raw")
        ][-limit:]
    return {"rationale": items, "count": len(items)}
```

### Step 6: Test Rationale Capture

**tests/test_routing_rationale.py**:

```python
import pytest
from proxy import arch_router_policy, Metric

class TestRoutingRationale:
    def test_judge_raw_captured(self):
        """Verify judge raw response is captured."""
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "simple question"}]
        }
        policy, router_ms = arch_router_policy(body)
        assert "judge_raw" in policy
        assert len(policy["judge_raw"]) > 0

    def test_judge_tokens_captured(self):
        """Verify judge token usage is captured."""
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "simple question"}]
        }
        policy, _ = arch_router_policy(body)
        assert "judge_tokens" in policy
        assert policy["judge_tokens"] > 0

    def test_judge_latency_captured(self):
        """Verify judge latency is captured."""
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "simple question"}]
        }
        policy, _ = arch_router_policy(body)
        assert "judge_latency_ms" in policy
        assert policy["judge_latency_ms"] > 0

    def test_rationale_export_endpoint(self, gateway_client):
        """Verify rationale export endpoint works."""
        # Make a request first
        gateway_client.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "test"}]
        })
        # Export rationale
        response = gateway_client.get("/api/routing-rationale?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "rationale" in data
        assert data["count"] >= 1
```

### Step 7: Verify Dashboard

```bash
# Make a test request
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}'

# Check metrics include rationale
curl http://localhost:9000/metrics | python3 -m json.tool | grep -A 5 "judge_raw"

# Export rationale
curl "http://localhost:9000/api/routing-rationale?limit=10" | python3 -m json.tool
```

## Success Criteria

- [ ] Judge raw response captured for every automatic request
- [ ] Judge token usage recorded
- [ ] Judge latency recorded
- [ ] Dashboard displays expandable rationale
- [ ] Export endpoint returns rationale data
- [ ] Tests verify rationale capture

## Commands Reference

```bash
# Make a test request
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "test"}]}'

# Check metrics
curl http://localhost:9000/metrics | python3 -m json.tool

# Export rationale
curl "http://localhost:9000/api/routing-rationale?limit=50" | python3 -m json.tool

# Run rationale tests
pytest tests/test_routing_rationale.py -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `judge_raw` empty | Check judge model is loaded and responding |
| Rationale not in dashboard | Clear browser cache, hard refresh |
| Export endpoint returns empty | Make requests first, then export |
| Judge timeout | Increase `ROUTER_TIMEOUT_SEC` |
