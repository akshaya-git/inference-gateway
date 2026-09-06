# Model startup commands (benchmark lab)

Exact commands the lab uses to load each model in each framework. The lab runs
these automatically via `FrameworkManager`; this file documents them so you can
reproduce or debug a load manually.

All commands run from the project root (`/Users/darthvader/code/inference-gateway`).
Server logs land in `.inference-stack/bench/<framework>.log`.

## Resident layer (always on)

### oMLX (:8000) — the routing models
oMLX is the resident server. Models are loaded/unloaded via its admin API, not a
subprocess. The two routing models stay resident (serves the gateway + the agent).

```bash
# list models + load state
curl -s http://127.0.0.1:8000/admin/api/models | python3 -m json.tool

# load / unload a model
curl -X POST http://127.0.0.1:8000/admin/api/models/<model_id>/load
curl -X POST http://127.0.0.1:8000/admin/api/models/<model_id>/unload
```

Dense model id: `scottlowry--Qwen3.8-27B-oQ6e-mtp`
MoE model id: `mlx-community--Qwen3.6-35B-A3B-6bit`

## Transient layer (loaded per cell, unloaded after)

### MLX — `mlx_lm.server` (:8200)
```bash
~/.omlx/bench-venv/bin/mlx_lm.server \
  --model ~/.cache/huggingface/hub/models--scottlowry--Qwen3.8-27B-oQ6e-mtp/snapshots/<hash> \
  --host 127.0.0.1 --port 8200 --max-tokens 32768
```
- Dense model: `scottlowry/Qwen3.8-27B-oQ6e-mtp` (OptiQ 6-bit **MTP**, MLX-format,
  native 256K context). Same build oMLX serves; already in the HF cache.
- The request `model` id is the HF repo name (`scottlowry/Qwen3.8-27B-oQ6e-mtp`).
- `mlx_lm.server` loads the MTP model but does not run MTP spec decoding.

### MTPLX — `mtplx serve` (:8400)
```bash
mtplx serve \
  --model Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality \
  --host 127.0.0.1 --port 8400 --no-auth --download --yes \
  --context-window 131072 --max-tokens 32768
```
- Native MTP speculative decoding (the model's own MTP heads; no external
  drafter). OpenAI-compatible on :8400.
- Pre-pull the model once: `mtplx pull Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality`
- Stop cleanly: `mtplx stop`

### llama.cpp — `llama-server` (:8300)
```bash
llama-server \
  -m models/gguf/Qwen3.8-27B-MTP-Q6_K.gguf \
  --host 127.0.0.1 --port 8300 \
  --alias qwen3_8_27b --ctx-size 131072 --spec-type draft-mtp
```
- Dense GGUF: `models/gguf/Qwen3.8-27B-MTP-Q6_K.gguf` (Q6_K **with MTP tensors**,
  from `Jackrong/Qwen3.8-27B-MTP-GGUF`).
- `--spec-type draft-mtp` enables MTP speculative decoding (draft acceptance
  ~55-65%, mean ~2.7 tokens/step).
- The request `model` id is the alias (`qwen3_8_27b`).

### Ollama (:11434)
```bash
# server (if not already running)
ollama serve

# create the model from the local MTP GGUF (one-time; ~minutes)
printf 'FROM models/gguf/Qwen3.8-27B-MTP-Q6_K.gguf\nPARAMETER num_ctx 131072\nPARAMETER num_predict 32768\n' > /tmp/Modelfile-qwen38
ollama create bench/qwen38-27b-q6 -f /tmp/Modelfile-qwen38
```
- The request `model` id is the Ollama name (`bench/qwen38-27b-q6`).
- Ollama auto-loads the model on first request (no MTP spec decoding).

### Bionic (LM Studio GUI) — LM Studio server (:1234)
Bionic is a GUI app (`/Applications/Bionic.app`); it has no headless CLI. It
drives the LM Studio server on :1234. To benchmark it:
1. Open **Bionic.app** and load the dense model in its GUI (LM Studio downloads
   it into its own library on first use).
2. The lab's `bionic` harness then measures the LM Studio server on :1234
   (OpenAI-compatible). Its full agent behaviour (tool use, multi-step) is only
   exercised through the GUI.

```bash
# the LM Studio server endpoints (started by Bionic.app)
curl -s http://127.0.0.1:1234/v1/models
```

## Port map
| framework | port | resident |
|-----------|------|----------|
| oMLX | 8000 | yes |
| MLX (mlx_lm.server) | 8200 | no |
| MTPLX | 8400 | no |
| llama.cpp | 8300 | no |
| Ollama | 11434 | no (server) / model auto-loads |
| Bionic (LM Studio) | 1234 | no (GUI-managed) |

## RAM note (M5 Max, 128 GB)
Resident oMLX (both routing models) ≈ 39 GB. One transient benchmark copy adds
≈ 17–21 GB. Benchmarks run strictly sequentially so at most one transient copy
is loaded at a time.
