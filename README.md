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
| `MODEL_PATH` | `/data/models/llm/model.gguf` | Absolute path **inside the container** to the initial GGUF file |
| `N_GPU_LAYERS` | `-1` | Layers offloaded to GPU. `-1` = all (full GPU offload) |
| `CTX_SIZE` | `4096` | Context window in tokens |
| `HOST_PORT` | `5301` | Host port mapped to the container's internal port 8080 |
| `ADMIN_TOKEN` | *(empty)* | Bearer token required for model management write endpoints. Leave empty to disable auth (dev only). |

---

## Volume mounts

| Host path | Container path | Purpose |
|---|---|---|
| `/data/models` | `/data/models` (read-write) | GGUF model files — read-write so that the download endpoint can place new models |
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

## Model management

The service ships a **Python/FastAPI wrapper** (`wrapper/main.py`) that manages
`llama-server` as a child process and adds a model management API.  All model
management endpoints are available on the same published port (default `5301`).

> **Architecture note:** The wrapper is the only container process.  It starts
> `llama-server` internally, proxies all `/v1/*` and other llama-server requests
> transparently, and handles `/api/models/*` itself.

### Admin authentication

Set `ADMIN_TOKEN` in `.env` to a strong random string.  All write endpoints
(`POST /api/models/download` and `POST /api/models/load`) require the header:

```
Authorization: Bearer <ADMIN_TOKEN>
```

When `ADMIN_TOKEN` is empty the admin endpoints are **unauthenticated** — do
not leave it empty in production.

---

### List models — `GET /api/models`

Returns metadata for every `.gguf` file found in `/data/models/llm/`.

```bash
curl -s http://localhost:5301/api/models | jq .
```

Response:

```json
{
  "models": [
    {
      "filename": "mistral-7b-v0.3.Q4_K_M.gguf",
      "path": "/data/models/llm/mistral-7b-v0.3.Q4_K_M.gguf",
      "size_bytes": 4368438976,
      "quantization": "Q4_K_M",
      "loaded": true
    },
    {
      "filename": "llama-3.1-8b-Q5_K_S.gguf",
      "path": "/data/models/llm/llama-3.1-8b-Q5_K_S.gguf",
      "size_bytes": 5368709120,
      "quantization": "Q5_K_S",
      "loaded": false
    }
  ],
  "loaded_model": "/data/models/llm/mistral-7b-v0.3.Q4_K_M.gguf",
  "status": "ready"
}
```

---

### Download a model — `POST /api/models/download`

Downloads a GGUF file from a URL into `/data/models/llm/` without rebuilding
the container.

```bash
curl -s -X POST http://localhost:5301/api/models/download \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://huggingface.co/TheBloke/Mistral-7B-v0.3-GGUF/resolve/main/mistral-7b-v0.3.Q4_K_M.gguf",
    "filename": "mistral-7b-v0.3.Q4_K_M.gguf"
  }' | jq .
```

Response (HTTP 202 Accepted):

```json
{
  "task_id": "a3b2c1d0-...",
  "status": "pending",
  "filename": "mistral-7b-v0.3.Q4_K_M.gguf",
  "bytes_downloaded": 0,
  "total_bytes": null,
  "error": null
}
```

The download runs in the background.  Poll for progress:

```bash
curl -s http://localhost:5301/api/models/download/<task_id> | jq .
```

**Validation rules:**
* Destination filename must end with `.gguf`.
* Path separators (`/`, `\`) in the filename are rejected.
* At least **1 GiB** of free disk space must be available before the download starts.
* After the download, the first four bytes are verified against the GGUF magic
  (`GGUF`). Files that fail validation are deleted automatically.

---

### Switch active model — `POST /api/models/load`

Restarts `llama-server` with a different GGUF file already present in
`/data/models/llm/`.

```bash
curl -s -X POST http://localhost:5301/api/models/load \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"filename": "llama-3.1-8b-Q5_K_S.gguf"}' | jq .
```

Response (once the new model is ready):

```json
{
  "loaded_model": "/data/models/llm/llama-3.1-8b-Q5_K_S.gguf",
  "status": "ready",
  "error": null
}
```

**Readiness semantics — no zero-downtime guarantee:**

| Phase | `GET /health` response | Inference (`/v1/*`) |
|---|---|---|
| Model switch in progress | `{"status":"loading"}` | HTTP 503 |
| New model ready | `{"status":"ok"}` | Normal |
| New model failed to load | `{"status":"error"}` | HTTP 503 |

Poll `GET /health` after calling this endpoint and only resume sending
inference requests once `{"status":"ok"}` is returned.

---

### Health endpoint — `GET /health`

Reflects the wrapper state, not just the llama-server process.

| Response | Meaning |
|---|---|
| `{"status":"ok"}` | llama-server is running and healthy |
| `{"status":"loading"}` | Container just started, or a model switch is in progress |
| `{"status":"error","detail":"..."}` | llama-server failed to start or become healthy |

---

### Admin UI model picker (control-plane contract)

A future gateway-control-plane admin UI can integrate using these endpoints:

| Action | Endpoint |
|---|---|
| Populate model picker | `GET /api/models` |
| Download a new model | `POST /api/models/download` + poll `GET /api/models/download/{task_id}` |
| Switch active model | `POST /api/models/load` + poll `GET /health` |

The UI should treat all write calls as asynchronous:
1. Call the endpoint.
2. Show a loading/spinner state.
3. Poll until `status` reaches a terminal value (`ready`, `done`, or `error`).

The admin token must be kept in the control-plane secret store and injected as
the `Authorization` header — it must **not** be exposed to end users.

---

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

## OpenAI API compatibility

llama-server natively exposes `/v1/chat/completions` and `/v1/completions` in OpenAI-compatible format. The table below summarises compatibility with the chat-platform OpenAI provider adapter.

| Feature | Status | Notes |
|---|---|---|
| `messages` array format | ✅ Compatible | `role`/`content` fields match the OpenAI schema exactly |
| `temperature`, `top_p` | ✅ Compatible | Same semantics and value ranges |
| `max_tokens` | ✅ Compatible | llama-server accepts `max_tokens` as an alias for the native `n_predict` field |
| `stop` sequences | ✅ Compatible | Accepts the same string array as OpenAI |
| Streaming SSE (`"stream": true`) | ✅ Compatible | Returns `data: {...}` lines in OpenAI delta format, terminated with `data: [DONE]` |
| `model` field in responses | ⚠️ Cosmetic difference | llama-server echoes back the model filename (e.g. `mistral-7b-v0.3.Q4_K_M.gguf`) rather than a short alias. The chat-platform adapter ignores this field for routing, so no shim is required. |
| `tool_calls` / function calling | ⚠️ Not supported | llama-server does not implement the OpenAI `tools` / `tool_calls` response schema. If the chat-platform relies on structured function-calling, a thin proxy shim must be inserted (see note below). |
| `/v1/models` | ✅ Available | Returns a single-entry list for the loaded model; the chat-platform adapter uses this for model discovery (see [Chat-platform provider](#chat-platform-provider) section). |

### Streaming example

```bash
curl -sN http://<llm-service-host>:5301/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Count to five."}],
    "max_tokens": 64,
    "stream": true
  }'
```

Expected: a series of `data: {"choices":[{"delta":{"content":"..."}},...]}` lines ending with `data: [DONE]`.

### tool_calls shim note

If the downstream chat-platform requires `tool_calls` in responses (i.e. structured function-calling), a lightweight reverse-proxy layer (e.g. a small FastAPI or Node.js service) can be introduced between the platform and llama-server to:

1. Parse the model's text output against a function-call grammar.
2. Re-format the result as an OpenAI-compatible `tool_calls` array before returning it to the caller.

This shim is only needed when function-calling is required; for plain chat and text completion the native llama-server API is fully compatible.

---

## Chat-platform provider

Register this service as an **OpenAI-compatible** provider in the chat-platform settings:

| Setting | Value |
|---|---|
| Provider type | `openai-compatible` |
| Base URL | `http://<llm-service-host>:5301/v1` |
| API key | *(not required — leave blank or set any non-empty string)* |
| Default model | `local` *(or the value returned by `GET /v1/models`)* |

> **Note:** Replace `<llm-service-host>` with the hostname or IP of the machine running `docker compose up`. Do **not** commit a private LAN address in repository files; use the placeholder above in docs and config templates.

### /v1/models behaviour

llama-server exposes a `GET /v1/models` endpoint that returns the currently loaded model as a single list entry:

```json
{
  "object": "list",
  "data": [
    {
      "id": "mistral-7b-v0.3.Q4_K_M.gguf",
      "object": "model",
      "created": 1712467200,
      "owned_by": "llamacpp"
    }
  ]
}
```

The chat-platform provider adapter calls this endpoint to populate its model picker. Because only one model is loaded at a time, the list will always contain exactly one entry.

---

## Deployment via gateway-control-plane

This repository is deployable as a `container-service` using `build.strategy: repo-compose`. The wrapper service is built from `./wrapper` so the control-plane must have access to the full repo. The full workload manifest for gateway-control-plane is:

```yaml
services:
  llm-service:
    type: container-service
    build:
      strategy: repo-compose
      repo: goblinsan/llm-service
    network:
      mode: bridge
    runtime:
      class: nvidia
    config:
      HOST_PORT: "5301"
      MODEL_PATH: /data/models/llm/mistral-7b-v0.3.Q4_K_M.gguf
      N_GPU_LAYERS: "-1"
      CTX_SIZE: "4096"
      ADMIN_TOKEN: "<your-admin-token>"
    mounts:
      - source: /data/models
        target: /data/models
        readOnly: false
      - source: /data/llm
        target: /data/llm
        readOnly: false
    healthCheck:
      http:
        path: /health
        port: 5301
      interval: 30s
      timeout: 10s
      retries: 5
      startPeriod: 120s
```

Key manifest fields:

| Field | Value | Reason |
|---|---|---|
| `build.strategy` | `repo-compose` | Instructs the control plane to deploy the `docker-compose.yml` from this repo |
| `network.mode` | `bridge` | Standard Docker bridge networking; exposes `HOST_PORT` on the host |
| `runtime.class` | `nvidia` | Enables NVIDIA GPU access inside the container |
| `mounts[0]` | `/data/models` → `/data/models` (read-write) | GGUF model files — read-write so that `POST /api/models/download` can write new models |
| `mounts[1]` | `/data/llm` → `/data/llm` (read-write) | Runtime state, logs, and KV-cache scratch space |
| `healthCheck.http.path` | `/health` | llama-server health probe; returns `{"status":"ok"}` when ready |
| `healthCheck.http.port` | `5301` | Published host port (matches `HOST_PORT`) |