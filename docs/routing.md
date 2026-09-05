# Deterministic routing

`routing_logic.py` owns decisions; `routing_rules.json` is the versioned lookup.
The gateway calls `RoutingEngine.route()` and does not call the judge model.
Code requests above level 4 use dense; other automatic requests use the MoE
workhorse (Qwen3.6-35B-A3B 6-bit).
Explicit model aliases override everything. Leading `[model:dense]`,
`[model:moe]`, and `[complexity:5]` controls are supported and stripped upstream.

The rules file defines the six levels, code terms, base task patterns, and
modifier categories. Highest matching base wins; default code level is 3.
Each modifier counts once, at most two; final level is capped at 6.
Matching is a transparent phrase heuristic, not semantic understanding.

Decisions are retained for the same user-message history through tool turns.
A new user task is scored again; short continuations retain the prior decision.
The bounded 512-task memory resets on restart or history compaction. Identical
user-message histories share a decision; session IDs are not yet integrated.

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

The old judge can be unloaded in oMLX; it is no longer used or reloaded by routing.
Restart the proxy to activate this code. Existing benchmark aliases still select
exact models and bypass complexity routing. Backend failures are surfaced rather
than silently substituting the other model.
