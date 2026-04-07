# llm-service

GPU-accelerated LLM inference service built on [llama.cpp](https://github.com/ggerganov/llama.cpp), deployed via Docker Compose with NVIDIA CUDA. Designed to be deployed by **gateway-control-plane** as a `container-service` using `build.strategy: repo-compose`.

---

## Quick start

```bash
cp .env.example .env
# Edit .env: set MODEL_PATH to your GGUF file (see Model Selection below)
docker compose up -d
curl http://localhost:5301/health
```

---

## Configuration

All tunables live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/data/models/llm/model.gguf` | Absolute path **inside the container** to the GGUF file |
| `N_GPU_LAYERS` | `-1` | Layers offloaded to GPU. `-1` = all (full GPU offload) |
| `CTX_SIZE` | `4096` | Context window in tokens |
| `HOST_PORT` | `5301` | Host port mapped to the container's internal port 8080 |

---

## Volume mounts

| Host path | Container path | Purpose |
|---|---|---|
| `/data/models` | `/data/models` (read-only) | GGUF model files |
| `/data/llm` | `/data/llm` | Runtime state, logs, and cache |

---

## Model selection

All three candidates fit comfortably within an 8 GB VRAM budget at 4096-token context:

| Model | Quant | VRAM (weights) | VRAM at 4096 ctx | Notes |
|---|---|---|---|---|
| **Mistral 7B v0.3 Q4_K_M** | Q4_K_M | ~4.5 GB | ~5.2 GB | ✅ **Recommended default** — widest community support, strong general-purpose perf |
| Llama 3.1 8B Q4_K_M | Q4_K_M | ~5.0 GB | ~5.7 GB | Good alternative; slightly higher VRAM |
| Qwen 2.5 7B Q4_K_M | Q4_K_M | ~4.5 GB | ~5.2 GB | Excellent multilingual; similar footprint to Mistral |

**Recommendation: Mistral 7B v0.3 Q4_K_M** — ~4.5 GB weights + ~0.7 GB KV cache at 4096 ctx ≈ **5.2 GB total**, leaving ~2.8 GB headroom on an 8 GB card for the CUDA runtime and OS.

Download:

```bash
# Using huggingface-cli
huggingface-cli download \
  TheBloke/Mistral-7B-v0.3-GGUF \
  mistral-7b-v0.3.Q4_K_M.gguf \
  --local-dir /data/models/llm/

# Then set in .env:
MODEL_PATH=/data/models/llm/mistral-7b-v0.3.Q4_K_M.gguf
```

---

## VRAM budget

| Component | Approximate VRAM |
|---|---|
| Mistral 7B Q4_K_M weights | 4.5 GB |
| KV cache @ 4096 ctx | 0.7 GB |
| CUDA runtime overhead | 0.3 GB |
| **Total** | **~5.5 GB** |
| **Available headroom (8 GB card)** | **~2.5 GB** |

The remaining ~2.5 GB provides room for a concurrent STT (speech-to-text) process.  
**Scheduling note:** simultaneous STT inference + LLM inference on the same GPU is feasible but may cause latency spikes. Recommendation:
- Run STT in a separate CUDA stream / container with `--gpus` scoped to a fraction of the device.
- If both services must run on the same physical GPU, limit LLM `CTX_SIZE` to `2048` to reclaim ~0.4 GB of KV-cache VRAM and reduce peak pressure.
- Monitor with `nvidia-smi dmon` and set `CUDA_MPS_PIPE_DIRECTORY` if using Multi-Process Service for fine-grained sharing.

---

## Health endpoint

llama-server exposes `GET /health` on its HTTP port. The Docker Compose healthcheck polls it every 30 s and gateway-control-plane can use the same endpoint on the published port:

```
GET http://<host>:5301/health
```

Expected response while the model is loaded and ready:

```json
{"status":"ok"}
```

---

## Validating inference and VRAM usage

### 1. Check the server is healthy

```bash
curl -s http://localhost:5301/health | jq .
```

### 2. Run a test prompt

```bash
curl -s http://localhost:5301/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }' | jq '{content: .choices[0].message.content, usage: .usage}'
```

Key metrics to capture:
- **tokens/sec** — target ≥ 20 t/s for interactive use on a mid-range GPU
- **time-to-first-token** — target < 500 ms
- **VRAM peak** — observe with `nvidia-smi`

### 3. Measure VRAM concurrently

```bash
# In one terminal — watch VRAM every second while STT is idle:
nvidia-smi dmon -s mu -d 1

# In another terminal — send a prompt:
curl -s http://localhost:5301/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain CUDA in 3 sentences.", "n_predict": 128}' | jq .
```

Compare the `mem_used` column between STT-idle and STT-active runs to confirm the combined footprint stays below the card's total VRAM.

---

## Deployment via gateway-control-plane

This repository is deployable as a `container-service` using `build.strategy: repo-compose`:

```yaml
services:
  llm-service:
    type: container-service
    build:
      strategy: repo-compose
      repo: goblinsan/llm-service
    config:
      HOST_PORT: "5301"
      MODEL_PATH: /data/models/llm/mistral-7b-v0.3.Q4_K_M.gguf
      N_GPU_LAYERS: "-1"
      CTX_SIZE: "4096"
```