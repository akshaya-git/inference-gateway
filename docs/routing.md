# Deterministic routing

`routing_logic.py` owns decisions; `routing_rules.json` is the versioned lookup
(currently version 2). The gateway calls `RoutingEngine.route()` and does not
call any judge model.

## Scoring model

1. **Code gate.** `code_terms` (languages, frameworks, artifacts, technical
   verbs, infrastructure terms) set the *default* level: 3 for code-ish tasks,
   1 otherwise. The gate no longer vetoes rules: a non-code prompt that matches
   a base rule still gets that rule's level.
2. **Base rules.** Each rule lists phrase variants. Matching is stemmed
   token-set containment (word order, singular/plural, and common inflections
   do not kill a match) with an exact word-boundary fallback. The highest
   matching base level wins; if no rule matches, the code-gate default applies.
3. **Modifiers.** Each matched modifier adds one level, at most two; the final
   level is capped at 6.
4. **Route.** The `level_routes` map (optional, per-level override) or the
   `dense_above` threshold (default 4) selects the route: levels above the
   threshold use dense, the rest use the MoE workhorse.

Matching is a transparent term heuristic, not semantic understanding.
Benchmark vocabulary (tetris, kanban, dashboard, visualizer) is deliberately
absent from production rules; benchmark runs use explicit model aliases.

## Controls

Leading `[model:dense]`, `[model:moe]`, and `[complexity:1-6]` controls are
consumed and stripped before the request is forwarded. **Invalid controls**
(unknown model, complexity outside 1-6) are also stripped and reported in the
decision (`invalid_controls`, `source`) — they are never clamped and never
forwarded to the model. Explicit model aliases (`gateway-moe`, `gateway-dense`,
`benchmark-*`) override everything.

## Capability metadata

Each route in `router.yaml` declares `capabilities` (context window, max
output, vision). After the routing decision, the gateway adjusts the route
when the request needs a capability the selected model lacks: image content
switches to a vision-capable model if one exists, and prompts estimated to
exceed the context window switch to a larger window if one exists. Adjustments
and gaps are noted in the route reason.

## Rationale capture

Every decision records `complexity` (level), `matched_rules`, `modifiers`,
`invalid_controls`, `rules_version`, `source`, and `reason`. Metrics carry
`routing_level` and `routing_rules`; the dashboard shows the level and the
reason (which names the matched rules) instead of a confidence percentage.
`GET /api/routing-rationale` exports recent decisions for review.

## Stickiness and updates

Decisions are retained for the same user-message history through tool turns.
A new user task is scored again; short continuations ("continue", "go on",
"keep going") retain the prior decision. The bounded 512-task memory resets on
restart or history compaction. Identical user-message histories share a
decision; session IDs are not yet integrated.

Rule updates are loaded for new tasks. Invalid edits retain the last valid
lookup, with the error exposed in metrics. Existing task decisions stay fixed.
The gateway can read `/routing/rules` and PUT a complete validated lookup to
that endpoint; increment `version` for updates. This is a local administrative
API, not an endpoint to expose publicly.

Human benchmark verdicts append evidence to `.inference-stack/routing-feedback.jsonl`.
Use repeated outcomes to propose revised terms or levels, then install a reviewed
lookup through `RoutingEngine.update_rules()` or the API. Updates are atomic.
No model output, tool output, or isolated failure automatically rewrites policy.
This is feedback collection plus an update hook, not autonomous learning.

Restart the proxy to activate code changes (routing rules themselves hot-reload
per task). Existing benchmark aliases still select exact models and bypass
complexity routing. Backend failures are surfaced rather than silently
substituting the other model.
