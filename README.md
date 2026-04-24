# llm-service

GPU-accelerated LLM inference service built on [llama.cpp](https://github.com/ggml-org/llama.cpp), deployed via Docker Compose with NVIDIA CUDA. Designed to be deployed by **gateway-control-plane** as a `container-service` using `build.strategy: repo-compose`.

---

## Quick start

```bash
cp .env.example .env
# Edit .env: set MODEL_PATH to your GGUF file (see Model Selection below)
docker compose up -d
curl http://localhost:5301/health
# Optional: open the local operator UI
open http://localhost:5302
```

---

## Configuration

All tunables live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MODELS_HOST_DIR` | `/data/models` | Host path mounted to `/data/models` inside the container |
| `LLM_STATE_DIR` | `/data/llm` | Host path mounted to `/data/llm` for wrapper state and downloads |
| `MODELS_DIR` | `/data/models/llm` | Directory inside the container scanned for GGUF models |
| `MODEL_PATH` | `/data/models/llm/model.gguf` | Absolute path **inside the container** to the initial GGUF file. If it does not exist yet, the wrapper starts in `no-model` mode so you can download or load a model later. |
| `N_GPU_LAYERS` | `-1` | Layers offloaded to GPU. `-1` = all (full GPU offload) |
| `CTX_SIZE` | `4096` | Context window in tokens |
| `HOST_PORT` | `5301` | Host port mapped to the container's internal port 8080 |
| `UI_PORT` | `5302` | Host port for the local operator UI (nginx container, port 80 internally). The UI proxies `/health`, `/api/*`, and `/v1/*` to the `llm-service` container. |
| `ADMIN_TOKEN` | *(empty)* | Bearer token required for model management write endpoints. Leave empty to disable auth (dev only). |
| `LLAMA_BIN` | *(auto-detect)* | Optional absolute path to `llama-server` inside the container. Only set this if the base image places the binary somewhere unusual. |
| `MAX_TOKENS` | `2048` | Hard cap on `max_tokens` per inference request. Requests that exceed this value (or omit it) are silently clamped. Lower values reduce peak KV-cache VRAM pressure. |
| `REQUEST_TIMEOUT` | `120` | Seconds before an in-flight inference request is abandoned and a `504` is returned to the caller. |
| `MAX_CONCURRENT_REQUESTS` | `1` | Maximum simultaneous inference requests. Additional requests receive `HTTP 429`. Default of `1` serialises inference to avoid starving other GPU workloads. |

---

## Volume mounts

| Host path | Container path | Purpose |
|---|---|---|
| `MODELS_HOST_DIR` | `/data/models` (read-write) | GGUF model files — read-write so that the download endpoint can place new models |
| `LLM_STATE_DIR` | `/data/llm` | Runtime state, logs, and cache |

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

Set `ADMIN_TOKEN` in `.env` to a strong random string.  The following write endpoints require the header:

```
Authorization: Bearer <ADMIN_TOKEN>
```

| Write endpoint | Requires admin auth |
|---|---|
| `POST /api/models/download` | ✅ Yes |
| `POST /api/models/load` | ✅ Yes |
| `POST /api/models/unload` | ✅ Yes |

When `ADMIN_TOKEN` is empty the admin endpoints are **unauthenticated** — the wrapper logs a warning at startup and all admin writes are open to any caller.  **Never leave `ADMIN_TOKEN` empty in production or in any environment shared with other services.**

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
  "loaded_model_filename": "mistral-7b-v0.3.Q4_K_M.gguf",
  "ctx_size": 4096,
  "n_gpu_layers": -1,
  "status": "ready",
  "llama": {
    "offloaded_layers": 32,
    "total_layers": 32,
    "flash_attn": true,
    "cuda_device": "CUDA0",
    "cuda_device_name": "NVIDIA GeForce RTX 3070",
    "cuda_free_mib_at_load": 4096,
    "recent_log_tail": ["..."]
  }
}
```

`loaded_model_filename` is the basename of the active model (empty string when no model is loaded).
Orchestrators should prefer this field over `loaded_model` when they only need to identify the model
by name rather than by its container-local path.

---

### Node capability metadata — `GET /api/node`

Returns stable, read-only metadata that describes what this node can serve.
No authentication is required.  Intended for upstream routing agents that need to inspect
context limits, concurrency caps, and GPU configuration before dispatching requests.

```bash
curl -s http://localhost:5301/api/node | jq .
```

Response:

```json
{
  "status": "ok",
  "loaded_model": "mistral-7b-v0.3.Q4_K_M.gguf",
  "ctx_size": 4096,
  "max_tokens": 2048,
  "max_concurrent_requests": 1,
  "n_gpu_layers": -1,
  "llama": {
    "offloaded_layers": 32,
    "total_layers": 32,
    "flash_attn": true,
    "cuda_device": "CUDA0",
    "cuda_device_name": "NVIDIA GeForce RTX 3070",
    "cuda_free_mib_at_load": 4096,
    "recent_log_tail": ["..."]
  }
}
```

| Field | Description |
|---|---|
| `status` | Current readiness: `ok`, `loading`, `no-model`, or `error` |
| `loaded_model` | Basename of the active GGUF file; empty string when no model is loaded |
| `ctx_size` | Context window configured for llama-server (tokens) |
| `max_tokens` | Per-request token cap enforced by the wrapper (`MAX_TOKENS` env var) |
| `max_concurrent_requests` | Maximum simultaneous inference requests before HTTP 429 (`MAX_CONCURRENT_REQUESTS` env var) |
| `n_gpu_layers` | Layers offloaded to GPU; `-1` = all layers |
| `llama` | GPU/load diagnostics parsed from llama-server startup logs |

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
| No model loaded yet | `{"status":"no-model"}` | HTTP 503 |
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
| Inspect node capabilities | `GET /api/node` |
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

## Local operator UI

> **Scope:** This UI is for **local operator testing only**. It is intentionally not hardened for public access, has no per-user authentication or session isolation, and is not intended to be exposed beyond the operator's LAN. Public authenticated access and per-user sessions belong in **gateway-tools-platform**.

The compose stack ships a second service, `ui`, which is a React/Vite single-page app built into an nginx container. When both services are running, navigate to:

```
http://<host>:5302
```

(where `5302` is the default `UI_PORT`; change it in `.env` if needed).

### How the UI connects to the wrapper

The UI container (nginx) proxies all API traffic to the `llm-service` container on the internal Docker network — no CORS configuration is required. Proxied paths:

| UI request path | Forwarded to |
|---|---|
| `GET /health` | `llm-service:8080/health` |
| `GET /api/models` | `llm-service:8080/api/models` |
| `POST /api/models/download` | `llm-service:8080/api/models/download` |
| `GET /api/models/download/{task_id}` | `llm-service:8080/api/models/download/{task_id}` |
| `POST /api/models/load` | `llm-service:8080/api/models/load` |
| `POST /v1/chat/completions` | `llm-service:8080/v1/chat/completions` |
| All other paths | Served from nginx static files (`/usr/share/nginx/html`) |

### UI features

| Feature | Description |
|---|---|
| Status panel | Live health and loaded-model display; polls `/health` every 5 s |
| Model inventory | Lists all `.gguf` files from `/api/models`; shows which is loaded |
| Model admin | Operator form for `POST /api/models/download`, download progress polling, and `POST /api/models/load` |
| Chat panel | Multi-turn chat with streaming SSE support |
| Session history | Browser `localStorage`-backed session list (survives page refresh) |
| Export | Download chat transcript as JSON |
| System prompt | Per-session system prompt field |

### Boundary with gateway-tools-platform

| Concern | This UI (`llm-service`) | gateway-tools-platform |
|---|---|---|
| Target user | Local operator (single person) | End users, agents |
| Authentication | None (admin token kept in `.env`) | Per-user auth, OAuth/OIDC |
| Session isolation | None (single shared localStorage) | Per-user sessions |
| Public exposure | No — LAN / VPN only | Yes |
| Streaming | Yes (operator convenience) | Yes |
| Model switching | Yes (operator admin) | No (models fixed per deployment) |

---

## VRAM coexistence

This service is designed to coexist on an **8 GB VRAM** GPU with a concurrent STT (Whisper) workload.

### Combined VRAM budget

| Workload | Component | Approximate VRAM |
|---|---|---|
| **LLM** (Mistral 7B Q4_K_M) | Model weights | 4.5 GB |
| **LLM** | KV cache @ 4096 ctx | 0.7 GB |
| **LLM** | CUDA runtime overhead | 0.3 GB |
| **STT** (Whisper large-v3) | Model weights + activations | ~3.1 GB |
| **Combined total** | | **~8.6 GB** |

> **Note:** 8.6 GB slightly exceeds an 8 GB card when both services run simultaneously.  
> Apply the mitigations below to stay within budget.

### Scheduling strategy

The service enforces **sequential GPU access by default** (`MAX_CONCURRENT_REQUESTS=1`).  
Only one inference request runs at a time; additional requests receive `HTTP 429` immediately and callers are expected to retry after a short delay.  This avoids VRAM spikes from KV-cache allocations for multiple concurrent generations.

To coexist with the STT service:

| Mitigation | How to apply | VRAM saved |
|---|---|---|
| Reduce LLM context size | Set `CTX_SIZE=2048` in `.env` | ~0.4 GB (KV cache) |
| Stagger workloads | Route STT and LLM requests to different time windows at the control-plane layer | Peak overlap eliminated |
| CUDA MPS | Set `CUDA_MPS_PIPE_DIRECTORY` on the host to share the GPU at the kernel level | Reduces driver overhead |
| Monitor live | `nvidia-smi dmon -s mu -d 1` | — |

**Recommended baseline** for a shared 8 GB card:

```bash
CTX_SIZE=2048          # KV cache ~0.4 GB, total LLM ~5.1 GB
MAX_CONCURRENT_REQUESTS=1  # serialise inference
```

Expected combined VRAM with these settings: ~5.1 GB (LLM) + ~3.1 GB (STT) = **~8.2 GB**.  
This stays within an 8 GB card provided the two services are never active simultaneously — a property enforced by `MAX_CONCURRENT_REQUESTS=1` together with an upstream scheduler that drains STT before routing to the LLM (or vice versa).

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
| `tool_calls` / function calling | ⚠️ Not supported | llama-server still does not implement the OpenAI `tools` / `tool_calls` response schema. `llm-service` now includes an optional wrapper-side `gateway_tools` shim for local operator use, but it is not a drop-in replacement for OpenAI tool calling. |
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

### Built-in local tools (`gateway_tools`)

> ⚠️ **Transitional feature — do not expand.**
> `gateway_tools` is a convenience shim for direct operator use only.
> General-purpose tool orchestration belongs in **agent-service**, not here.
> This path will not be extended with new tools or capabilities; new integrations
> should rely on agent-service owning the tool-execution loop instead
> (see [When agent-service owns tool execution](#when-agent-service-owns-tool-execution)).

For local operator use, `llm-service` can optionally run a lightweight wrapper-side tool loop for:

1. `time_now`
2. `web_search`

Enable it by including a custom `gateway_tools` object in the chat-completions request body:

```json
{
  "model": "local",
  "messages": [{"role": "user", "content": "What day is it, and what are today's top Apple headlines?"}],
  "stream": true,
  "gateway_tools": {
    "enabled": true,
    "time": true,
    "web_search": true
  }
}
```

Notes:

* This is a wrapper-specific extension, not standard OpenAI `tool_calls`.
* The wrapper prompts the model to emit a deterministic `<tool_call>{...}</tool_call>` block, executes the tool, then asks the model for a final answer.
* Streaming remains available, but tool-enabled requests are resolved server-side before the final answer is streamed back.
* **Do not use `gateway_tools` in agent-service or automated orchestration pipelines.** Use plain `/v1/chat/completions` requests instead and let agent-service own the tool-execution loop.

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

This repository is deployable as a `container-service` using `build.strategy: repo-compose`. Both services (`llm-service` and `ui`) are built from this repo — the control-plane must have access to the full repo. The full workload manifest for gateway-control-plane is:

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
      UI_PORT: "5302"
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
| `build.strategy` | `repo-compose` | Instructs the control plane to deploy the `docker-compose.yml` from this repo — both `llm-service` and `ui` services are started |
| `network.mode` | `bridge` | Standard Docker bridge networking; exposes `HOST_PORT` and `UI_PORT` on the host |
| `runtime.class` | `nvidia` | Enables NVIDIA GPU access inside the `llm-service` container |
| `mounts[0]` | `/data/models` → `/data/models` (read-write) | GGUF model files — read-write so that `POST /api/models/download` can write new models |
| `mounts[1]` | `/data/llm` → `/data/llm` (read-write) | Runtime state, logs, and KV-cache scratch space |
| `healthCheck.http.path` | `/health` | llama-server health probe; returns `{"status":"ok"}` when ready |
| `healthCheck.http.port` | `5301` | Published host port (matches `HOST_PORT`) |

### Operator follow-up after initial deploy

After merging and triggering a control-plane deploy, the operator must complete these steps manually:

1. **Register the workload in the control plane.** Add the `llm-service` manifest above to the control-plane's service registry if it is not already present. The control plane will not discover new workloads automatically.

2. **Deploy to the GPU node.** Trigger a deploy from the control-plane UI or CLI targeting the GPU node that has the required VRAM and NVIDIA driver. Confirm the node selector matches the node where `/data/models` and `/data/llm` are provisioned.

3. **Verify the UI container rebuilt successfully.** After the deploy, confirm both containers are running:
   ```bash
   # On the GPU node (from the repo directory)
   docker compose ps
   ```
   The `ui` container is built from `./ui` (multi-stage Vite + nginx, requires Node.js 22). If the build fails (e.g. a Node.js version mismatch on the build host), the `llm-service` inference container continues to run — check `docker compose logs ui` for build errors and re-trigger the deploy after fixing them.

4. **Validate the UI is reachable.** Open `http://<node-host>:5302` in a browser and confirm the status panel shows `ok` once a model is loaded.

---

## Agent assignment

This section documents the provider capabilities and model-naming conventions required for a follow-up chat-platform / control-plane change that routes selected agents to this local LLM.

### Provider capabilities

| Capability | Supported | Notes |
|---|---|---|
| Plain chat (`/v1/chat/completions`) | ✅ Yes | OpenAI-compatible; use `"model": "local"` |
| Text completion (`/v1/completions`) | ✅ Yes | OpenAI-compatible |
| Embeddings (`/v1/embeddings`) | ✅ Yes | Passed through to llama-server |
| Streaming SSE (`"stream": true`) | ✅ Yes | Standard OpenAI delta format |
| `tool_calls` / function calling | ⚠️ Not supported | No native OpenAI tool-calling schema. Wrapper-specific `gateway_tools` is available for local date/time and web search, but agents that require standard `tool_calls` should still use a proper shim or cloud provider. |
| Vision / multimodal | ❌ No | Text-only models (GGUF); no image input support |
| Fine-grained rate limits per agent | ❌ No | Global `MAX_CONCURRENT_REQUESTS` limit applies to all callers equally |

### Model naming conventions

The wrapper is intentionally model-agnostic.  The chat-platform should use the canonical alias `local` as the `model` field value in all requests — llama-server accepts any string and the wrapper does not validate it:

```json
{ "model": "local", "messages": [...] }
```

`GET /v1/models` always returns exactly one entry whose `id` is the loaded GGUF filename (e.g. `mistral-7b-v0.3.Q4_K_M.gguf`).  The chat-platform provider adapter should:

1. Call `GET /v1/models` at startup (or periodically) to discover the canonical model ID.
2. Cache the result and surface it to agents as `local/<filename>` or simply `local`.
3. Never hard-code the filename in agent profiles — the loaded model may change via `POST /api/models/load`.

### Recommended agents for the local LLM

The following agent archetypes are well-suited to a quantised 7B model on a shared GPU node:

| Agent type | Rationale |
|---|---|
| Internal summarisation / digest | Low latency tolerance; benefits from on-premises processing |
| Code explanation / review | Strong performance from Mistral 7B class models |
| Draft / first-pass text generation | Good quality at low token budget (`MAX_TOKENS=2048`) |
| Classification / routing | Short-context tasks; fast time-to-first-token |

Agents that require **function calling**, **vision input**, or **very long context windows** (> 4096 tokens) should be routed to a cloud provider instead.

### Guardrails applied to every request

The wrapper enforces these limits automatically — no per-agent configuration is required:

| Guardrail | Value (default) | Override via |
|---|---|---|
| `max_tokens` cap | 2048 | `MAX_TOKENS` env var |
| Request timeout | 120 s | `REQUEST_TIMEOUT` env var |
| Concurrent inference slots | 1 | `MAX_CONCURRENT_REQUESTS` env var |
| Response when slot is busy | `HTTP 429` with `Retry-After: 5` | — |

---

## Agent-service integration contract

This section defines the stable interface that **agent-service** (or any upstream orchestrator building a multi-node backend pool) should use to consume this service.  `llm-service` exposes the facts; routing policy belongs in the orchestrator.

### Service URLs

| Purpose | URL |
|---|---|
| Inference | `http://<host>:<HOST_PORT>/v1/chat/completions` |
| Node capability snapshot | `http://<host>:<HOST_PORT>/api/node` |
| Model inventory | `http://<host>:<HOST_PORT>/api/models` |
| Readiness probe | `http://<host>:<HOST_PORT>/health` |
| Serving metrics | `http://<host>:<HOST_PORT>/api/metrics` |

Replace `<host>` and `<HOST_PORT>` (default `5301`) with the values from the deployment manifest.
Do **not** hardcode private LAN addresses in agent-service configuration — resolve them at runtime from
the control-plane service registry.

### Authentication expectations

| Endpoint | Auth required |
|---|---|
| `GET /api/node` | ❌ None — read-only capability snapshot |
| `GET /api/models` | ❌ None — read-only inventory |
| `GET /api/metrics` | ❌ None — read-only serving metrics |
| `GET /health` | ❌ None — readiness probe |
| `GET /api/models/download/{task_id}` | ❌ None — read-only progress poll |
| `POST /api/models/download` | ✅ `Authorization: Bearer <ADMIN_TOKEN>` |
| `POST /api/models/load` | ✅ `Authorization: Bearer <ADMIN_TOKEN>` |
| `POST /api/models/unload` | ✅ `Authorization: Bearer <ADMIN_TOKEN>` |
| `POST /v1/chat/completions` (and other `/v1/*`) | ❌ None by default |

The `ADMIN_TOKEN` must be stored in the agent-service's secret store and never exposed to end users.

### Polling pattern for agent-service registration

When agent-service registers a new `llm-service` node, it should:

1. **Poll `GET /api/node`** to confirm the node is `"status": "ok"` and has a model loaded.
2. **Read `GET /api/node`** to cache `ctx_size`, `max_tokens`, `max_concurrent_requests`,
   and `loaded_model` for routing decisions.
3. **Refresh periodically** (e.g. every 30 s or on inference error) — the loaded model and
   capability values can change if the operator hot-swaps a model via `POST /api/models/load`.

### Health and readiness semantics

`GET /health` and `GET /api/node` both return a `status` field:

| `status` | HTTP code (`/health`) | Meaning | Inference safe? |
|---|---|---|---|
| `ok` | 200 | llama-server is ready | ✅ Yes |
| `loading` | 503 | Starting up or switching model | ❌ No — retry after `Retry-After` header |
| `no-model` | 503 | No GGUF loaded yet | ❌ No — operator must load a model first |
| `error` | 503 | llama-server failed | ❌ No — check `detail` for the reason |

`GET /api/node` always returns HTTP 200 regardless of status so orchestrators can always read
capability metadata even when the node is not serving inference.

### Model inventory semantics

`GET /api/models` lists every `.gguf` file present in `MODELS_DIR` alongside a per-file `loaded` flag.
The top-level `loaded_model_filename` field is the stable, portable identifier to use for routing:

```json
{
  "loaded_model_filename": "mistral-7b-v0.3.Q4_K_M.gguf",
  "models": [
    { "filename": "mistral-7b-v0.3.Q4_K_M.gguf", "loaded": true, ... },
    { "filename": "llama-3.1-8b-Q5_K_S.gguf",    "loaded": false, ... }
  ]
}
```

* `loaded_model_filename` is the basename only (no path) — it is safe to compare across nodes even
  if the absolute `MODELS_DIR` differs between deployments.
* A model entry with `"loaded": true` means llama-server is actively serving that model.
* Only one model can be loaded at a time; `"loaded": false` means the file is present but idle.
* `loaded_model_filename` is an empty string when `status` is `no-model` or `error`.

### Concurrency and back-pressure

Each node enforces `MAX_CONCURRENT_REQUESTS` (default `1`).  When the slot is occupied,
further inference requests receive `HTTP 429` with a `Retry-After: 5` header.  Agent-service
should implement an exponential-backoff retry on 429 and treat a sustained 429 as a
capacity-full signal for load-shedding or re-routing to another node.

The `max_concurrent_requests` value from `GET /api/node` lets agent-service estimate how many
in-flight requests this node can absorb before it will start returning 429.

### Timeout behaviour

Inference requests that exceed `REQUEST_TIMEOUT` (default `120 s`) are abandoned with `HTTP 504`.
Non-inference proxy requests (e.g. `GET /v1/models`) have no timeout.

| Condition | HTTP status | Header |
|---|---|---|
| Model not loaded | `503 Service Unavailable` | — |
| Model loading / switching | `503 Service Unavailable` | `Retry-After: 5` |
| Concurrency slot full | `429 Too Many Requests` | `Retry-After: 5` |
| Inference request timed out | `504 Gateway Timeout` | — |
| Backend connect error | `503 Service Unavailable` | — |

Callers should:
1. Retry `503` with `Retry-After` header using the indicated delay before re-sending.
2. Retry `429` with exponential back-off starting at the `Retry-After` value.
3. Treat `504` as a terminal failure for the request and surface it to the caller.
4. Never retry a `400`/`403`/`404` response — these indicate a bad request or configuration error.

### Per-node serving metrics — `GET /api/metrics`

Exposes cumulative counters and the current node state for monitoring.  No authentication is required.

```bash
curl -s http://<host>:5301/api/metrics | jq .
```

```json
{
  "uptime_seconds": 3600.0,
  "active_requests": 0,
  "requests_total": 142,
  "requests_rejected_429_total": 3,
  "requests_timeout_total": 1,
  "requests_error_total": 0,
  "model_load_total": 2,
  "model_load_error_total": 0,
  "request_timeout": 120.0,
  "max_concurrent_requests": 1
}
```

| Field | Meaning |
|---|---|
| `uptime_seconds` | Seconds since the wrapper process started |
| `active_requests` | Inference requests currently in flight |
| `requests_total` | Total inference requests that entered service |
| `requests_rejected_429_total` | Requests rejected because the concurrency slot was full |
| `requests_timeout_total` | Requests abandoned after exceeding `REQUEST_TIMEOUT` |
| `requests_error_total` | Requests that failed with a backend connect/upstream error |
| `model_load_total` | Successful model-load (hot-swap) operations |
| `model_load_error_total` | Model-load attempts that ended in error |
| `request_timeout` | Configured `REQUEST_TIMEOUT` in seconds |
| `max_concurrent_requests` | Configured `MAX_CONCURRENT_REQUESTS` |

Counters are reset when the wrapper process restarts.  They are not persisted between restarts.

### Token budget

The wrapper silently clamps `max_tokens` to `MAX_TOKENS` (default `2048`).  Agent-service
**must not** send requests with a `max_tokens` value larger than the `max_tokens` field
returned by `GET /api/node` — doing so will result in truncated responses without an error.

### Minimal polling example

```bash
# 1. Check readiness
curl -s http://<host>:5301/api/node | jq '{status, loaded_model, ctx_size, max_tokens, max_concurrent_requests}'

# 2. List available models
curl -s http://<host>:5301/api/models | jq '{loaded_model_filename, models: [.models[] | {filename, loaded}]}'

# 3. Confirm liveness (for health-check loop)
curl -s http://<host>:5301/health | jq .status

# 4. Inspect serving metrics (for observability / alerting)
curl -s http://<host>:5301/api/metrics | jq '{active_requests, requests_total, requests_rejected_429_total, requests_timeout_total}'
```

---

### When agent-service owns tool execution

When an upstream orchestrator (e.g. **agent-service**) owns the tool-execution loop, `llm-service`
acts as a **raw inference node only**.  Send standard `/v1/chat/completions` requests without the
`gateway_tools` extension and let agent-service handle function calling, result injection, and
multi-turn tool loops.

#### What to send

```json
{
  "model": "local",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the weather in New York?"},
    {"role": "assistant", "content": "<tool_call>{\"name\":\"weather\",\"arguments\":{\"location\":\"New York\"}}</tool_call>"},
    {"role": "tool",   "content": "Sunny, 22 °C"},
    {"role": "user",   "content": "Thanks, summarise that."}
  ],
  "stream": true,
  "max_tokens": 512
}
```

Agent-service builds the full conversation (including injected tool results) and calls
`POST /v1/chat/completions` with the assembled message list.  `llm-service` forwards the request
directly to llama-server — it does **not** inspect or re-execute tool calls in the messages.

#### What llm-service provides in this mode

| Responsibility | Owner |
|---|---|
| Raw text generation | `llm-service` (llama-server) |
| Tool call detection | agent-service |
| Tool execution | agent-service |
| Result injection into conversation | agent-service |
| Multi-turn tool loop | agent-service |
| Model routing / load balancing | agent-service |

#### Stable inference contract for orchestrators

* **No `gateway_tools` field** — omit it entirely; the request is forwarded to llama-server as-is.
* **Model name echoing** — the `model` field in every response (streaming and non-streaming) echoes
  the value sent in the request, regardless of the loaded GGUF filename.  Use any stable alias
  (e.g. `"local"`, `"llm-node-0"`) for routing identity.
* **`GET /api/node` always returns HTTP 200** — orchestrators can always read capability metadata
  (`ctx_size`, `max_tokens`, `max_concurrent_requests`, `status`) even when the node is not serving
  inference (e.g. while loading or after a crash).
* **`GET /health` and `GET /api/node` share the same `status` field values** (`ok`, `loading`,
  `no-model`, `error`) — use `/api/node` for polling (always 200) and `/health` for
  liveness-probe semantics (returns 503 for any non-`ok` state).
* **503 with `Retry-After: 5`** is returned for inference requests during `loading` and `error`
  states; **503 without `Retry-After`** is returned when `status` is `no-model`.
* **429 with `Retry-After: 5`** when the concurrency slot is full — implement exponential backoff
  in the orchestrator starting at the indicated delay.
