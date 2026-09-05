# Local MLX inference gateway

OpenAI-compatible gateway for Pi on `http://127.0.0.1:9000/v1`, backed by the
oMLX server on `http://127.0.0.1:8000`.

## Models

- `gateway-moe` / `benchmark-moe`: `mlx-community--Qwen3.6-35B-A3B-6bit`
- `gateway-dense` / `benchmark-dense`: `scottlowry--Qwen3.8-27B-oQ6e-mtp`
- `gateway-auto`: deterministic routing from `routing_logic.py` and
  `routing_rules.json`

There is no judge model. Code requests above complexity level 4 use the dense
model. Explicit `[model:moe]`, `[model:dense]`, and `[complexity:1-6]` controls
are supported and removed before forwarding the request to oMLX.

## Start and operate

```bash
~/Documents/inference-stack.sh start
~/Documents/inference-stack.sh status
~/Documents/inference-stack.sh logs all
~/Documents/inference-stack.sh restart
~/Documents/inference-stack.sh stop
~/Documents/inference-stack.sh moe-on
~/Documents/inference-stack.sh dense-on
```

Restart and stop affect the gateway only. oMLX and loaded models remain under
the oMLX application's control.

## Pi

Copy `docs/pi-models.json` to `~/.pi/agent/models.json`. The normal default is:

```json
{
  "defaultProvider": "mlx-proxy",
  "defaultModel": "gateway-auto"
}
```

Use `/model` inside Pi to select `gateway-moe` or `gateway-dense` explicitly.

## Endpoints

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`
- `GET /metrics`
- `GET /dashboard`
- `POST /benchmarks/run`
- `POST /control/model/moe`
- `POST /control/model/dense`

## Configuration

Model IDs and routing defaults live in `router.yaml`. Rules live in
`routing_rules.json`. Common environment overrides are `MOE_MODEL`,
`DENSE_MODEL`, `OMLX_UPSTREAM`, `OMLX_API_KEY`, `KEEP_MODELS_LOADED`,
`MEMORY_GUARD_GB`, `MEMORY_HARD_GB`, and `PI_BENCHMARK_TIMEOUT_SEC`.

The gateway serializes normal inference by default (`MAX_ACTIVE_REQUESTS=1`) so
Lightning MTP stays on its efficient single-request path. It does not impose an
idle timeout on oMLX streaming requests.

## Tests

```bash
.venv/bin/python -m pytest
```
