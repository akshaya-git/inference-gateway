# Task 3: Routing Rationale Capture

## Status: ✅ COMPLETE (CP-2, 2026-09-06)

Routing is deterministic — there is no judge model to capture. The "rationale"
is the full decision record of the rule engine: level, matched rules,
modifiers, invalid controls, rules version, and source. Everything below is
implemented and tested.

## What is captured

`RoutingEngine.route()` returns a decision:

```json
{
  "route": "dense",
  "complexity": 6,
  "code_related": true,
  "matched_rules": ["concurrency"],
  "modifiers": [],
  "invalid_controls": [],
  "rules_version": 2,
  "source": "concurrency",
  "reason": "Level 6; concurrency; lookup v2"
}
```

- `complexity` — the 1-6 level (or `null` when routing is disabled).
- `matched_rules` / `modifiers` — the rule IDs that fired, in rule order.
- `invalid_controls` — stripped `[model:...]` / `[complexity:...]` controls
  that were out of range (never forwarded, never clamped).
- `source` — `explicit model` | `explicit complexity` | rule IDs | `default`
  | `routing disabled` (prefixed with `invalid control stripped: ...` when
  applicable).
- `reason` — one-line human summary, stored on the metric and shown in the
  dashboard.

## Where it is stored / exposed

| Surface | Field / endpoint |
|---------|------------------|
| `Metric` (per request) | `routing_level`, `routing_rules` (comma-joined rule IDs), `route_reason` |
| `GET /metrics` | history entries carry the fields above; `config.routing_rules_version`, `config.routing_rules_error`, `config.model_capabilities` |
| Dashboard | **Level** column (L1–L6) + reason text naming the matched rules (replaces the old fake "Confidence %" column) |
| `GET /api/routing-rationale?limit=N` | export of recent decisions: ts, route, level, matched_rules, reason, requested_model, model + current `rules_version` / `dense_above` |
| `GET /routing/rules` | the full versioned lookup currently in effect |

## Refinement loop (Task 7 input)

Human benchmark verdicts append to `.inference-stack/routing-feedback.jsonl`.
Review verdicts + `/api/routing-rationale` exports to propose term/level
changes, then install a reviewed lookup via `PUT /routing/rules` (version must
increment; validated; atomic) or `RoutingEngine.update_rules()`. Invalid edits
retain the last valid lookup; the error is exposed in metrics. No model output
or isolated failure rewrites policy automatically.

## Tests

- `tests/test_routing_logic.py` — 12-prompt regression suite (the CP-2 gap
  analysis prompts), invalid-control handling, multi-part stripping,
  code-gate/rule decoupling, configurable `dense_above`, `level_routes` map,
  rules validation, stickiness across rule reloads.
- `tests/test_proxy.py` — capability adjustment (vision switch, context-window
  switch, no-capable-model warning) and `ROUTING_ENABLED` pinning.
