#!/usr/bin/env python3
"""llm-service model-management wrapper.

Manages a llama-server subprocess and exposes additional REST endpoints:

  GET  /api/models                    - list available GGUF models
  POST /api/models/download           - download a new GGUF model  (admin)
  GET  /api/models/download/{task_id} - poll download progress
  POST /api/models/load               - switch active model         (admin)
  POST /api/models/unload             - unload active model         (admin)
  GET  /health                        - readiness probe
  *    /*                             - transparent proxy to llama-server

All admin endpoints require  Authorization: Bearer <ADMIN_TOKEN>  when the
ADMIN_TOKEN environment variable is set to a non-empty string.

Environment variables
---------------------
MODEL_PATH          Path inside the container to the initial GGUF file.
                    If the file does not exist on first boot, the wrapper
                    starts in "no-model" mode so models can be downloaded
                    and loaded later. Default: /data/models/llm/model.gguf
MODELS_DIR          Directory scanned for GGUF files and used as the
                    download destination.
                    Default: /data/models/llm
N_GPU_LAYERS        Layers offloaded to GPU (-1 = all).  Default: -1
CTX_SIZE            Context window in tokens.  Default: 4096
ADMIN_TOKEN         Bearer token required for write operations.
                    Leave empty to disable authentication (dev only).
LLAMA_PORT          Internal port llama-server listens on.  Default: 8081
LLAMA_STARTUP_TIMEOUT Seconds to wait for llama-server /health after launch.
                    Default: 120
DOWNLOAD_READ_TIMEOUT Read timeout in seconds for model downloads.
                    Default: 300
MAX_TOKENS          Hard cap on max_tokens for each inference request.
                    Requests that omit max_tokens or exceed this cap are
                    silently clamped to this value.  Default: 2048
REQUEST_TIMEOUT     Seconds before an in-flight inference request is
                    abandoned and a 504 is returned to the caller.
                    Default: 120
MAX_CONCURRENT_REQUESTS
                    Maximum number of inference requests that may run
                    simultaneously.  Additional requests receive HTTP 429.
                    Default: 1  (serialise, like the STT service)
LLAMA_BIN           Optional absolute path to the llama-server binary.
                    Leave unset to auto-detect it in the container image.
SKIP_LLAMA_STARTUP  Set to "1" to skip launching llama-server (test mode).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import urllib.parse
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  (module-level; override directly in tests)
# ---------------------------------------------------------------------------

MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "/data/models/llm"))
ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "")
LLAMA_PORT: int = int(os.getenv("LLAMA_PORT", "8081"))
INITIAL_N_GPU_LAYERS: int = int(os.getenv("N_GPU_LAYERS", "-1"))
INITIAL_CTX_SIZE: int = int(os.getenv("CTX_SIZE", "4096"))
INITIAL_MODEL: str = os.getenv("MODEL_PATH", str(MODELS_DIR / "model.gguf"))
LLAMA_STARTUP_TIMEOUT: int = int(os.getenv("LLAMA_STARTUP_TIMEOUT", "120"))
DOWNLOAD_READ_TIMEOUT: float = float(os.getenv("DOWNLOAD_READ_TIMEOUT", "300"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "120"))
MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))
_SKIP_LLAMA_STARTUP: bool = os.getenv("SKIP_LLAMA_STARTUP", "0") == "1"


def _find_llama_bin() -> Optional[str]:
    """Locate the llama-server binary (path varies by image version)."""
    explicit = os.getenv("LLAMA_BIN", "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) and os.access(explicit, os.X_OK) else None
    candidates = [
        shutil.which("llama-server"),
        "/llama-server",
        "/app/llama-server",
        "/usr/bin/llama-server",
        "/usr/local/bin/llama-server",
        "/opt/llama.cpp/bin/llama-server",
        "/opt/llama.cpp/build/bin/llama-server",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    for root in ("/app", "/opt", "/usr/local", "/usr"):
        if not os.path.isdir(root):
            continue
        try:
            for dirpath, _, filenames in os.walk(root):
                if "llama-server" not in filenames:
                    continue
                path = os.path.join(dirpath, "llama-server")
                if os.access(path, os.X_OK):
                    return path
        except OSError:
            continue
    return None


_LLAMA_BIN: Optional[str] = _find_llama_bin()

# ---------------------------------------------------------------------------
# Mutable runtime state
# ---------------------------------------------------------------------------

_state: dict = {
    "process": None,         # subprocess.Popen | None
    "model": INITIAL_MODEL,  # currently active model path
    "ctx_size": INITIAL_CTX_SIZE,
    "n_gpu_layers": INITIAL_N_GPU_LAYERS,
    "status": "loading",     # "loading" | "ready" | "error" | "no-model"
    "error": None,           # str | None
}

_downloads: dict[str, dict] = {}  # task_id → progress dict

# Counter of in-flight inference requests.  Because asyncio is single-threaded,
# incrementing/decrementing this integer is safe without a lock (there is no
# await between the check and the increment below).
_active_inference: int = 0

# Paths that count as inference requests (GPU-bound; subject to concurrency cap).
_INFERENCE_PATHS: frozenset[str] = frozenset(
    {"v1/chat/completions", "v1/completions", "v1/embeddings"}
)
# Inference paths where a max_tokens budget must be enforced.
_TOKEN_BUDGET_PATHS: frozenset[str] = frozenset(
    {"v1/chat/completions", "v1/completions"}
)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):
    """Manage llama-server process lifetime alongside the FastAPI app."""
    await _on_startup()
    yield
    _on_shutdown()


app = FastAPI(
    title="llm-service wrapper",
    description="Model management wrapper around llama-server.",
    version="1.0.0",
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_QUANT_RE = re.compile(r"\b(Q\d+[_A-Z0-9]*)\b", re.IGNORECASE)


def _parse_quant(filename: str) -> Optional[str]:
    """Extract a quantization label from a GGUF filename (e.g. 'Q4_K_M')."""
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else None


def _free_bytes(path: Path) -> int:
    """Return free bytes on the filesystem that contains *path*."""
    return shutil.disk_usage(str(path)).free


def _require_admin(authorization: Optional[str]) -> None:
    """Raise HTTP 403 when ADMIN_TOKEN is set and the bearer token doesn't match."""
    if not ADMIN_TOKEN:
        return
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=403, detail="Invalid or missing admin token")


def _start_llama(model_path: str, ctx_size: int, n_gpu_layers: int) -> subprocess.Popen:
    if _LLAMA_BIN is None:
        raise RuntimeError(
            "llama-server binary not found. "
            "Set LLAMA_BIN explicitly or use an image that ships llama-server."
        )
    # Verify the model file is a real, accessible file before exec-ing.
    # This check also prevents unexpected values from reaching subprocess.
    if not os.path.isfile(model_path):
        raise ValueError(f"Model file is not accessible: {Path(model_path).name}")
    cmd = [
        _LLAMA_BIN,
        "--model", model_path,
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size", str(ctx_size),
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT),
    ]
    log.info("Starting llama-server: %s", " ".join(cmd))
    return subprocess.Popen(cmd)


def _stop_llama() -> None:
    proc: Optional[subprocess.Popen] = _state.get("process")
    if proc is not None and proc.poll() is None:
        log.info("Stopping llama-server (pid=%d)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log.warning("llama-server did not exit cleanly — killing")
            proc.kill()
            proc.wait()
    _state["process"] = None


async def _wait_for_llama(timeout: int = LLAMA_STARTUP_TIMEOUT) -> bool:
    """Poll llama-server /health until it returns HTTP 200 or *timeout* seconds elapse."""
    url = f"http://127.0.0.1:{LLAMA_PORT}/health"
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    async with httpx.AsyncClient() as client:
        while loop.time() < deadline:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200:
                    log.info("llama-server is healthy")
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
    log.error("llama-server did not become healthy within %ds", timeout)
    return False

# ---------------------------------------------------------------------------
# Startup / shutdown helpers (called by _lifespan)
# ---------------------------------------------------------------------------


async def _on_startup() -> None:
    if _SKIP_LLAMA_STARTUP:
        _state["status"] = "ready"
        log.info("SKIP_LLAMA_STARTUP=1 — skipping llama-server launch (test/dev mode)")
        return

    if not os.path.isfile(_state["model"]):
        _state["process"] = None
        _state["status"] = "no-model"
        _state["error"] = f"Initial model file not found: {Path(_state['model']).name}"
        log.warning(
            "Initial model file is missing (%s); starting in no-model mode",
            _state["model"],
        )
        return

    _state["status"] = "loading"
    try:
        proc = _start_llama(
            _state["model"],
            int(_state["ctx_size"]),
            int(_state["n_gpu_layers"]),
        )
        _state["process"] = proc
    except Exception as exc:
        log.error("Failed to start llama-server: %s", exc)
        _state["status"] = "error"
        _state["error"] = "Failed to start llama-server"
        return

    ok = await _wait_for_llama()
    if ok:
        _state["status"] = "ready"
        _state["error"] = None
    else:
        _state["status"] = "error"
        _state["error"] = "llama-server did not become healthy during startup"


def _on_shutdown() -> None:
    _stop_llama()

# ---------------------------------------------------------------------------
# Model management API
# ---------------------------------------------------------------------------


@app.get("/api/models", summary="List available GGUF models")
def list_models() -> dict:
    """Return metadata for every .gguf file found in MODELS_DIR."""
    models = []
    if MODELS_DIR.is_dir():
        for f in sorted(MODELS_DIR.glob("*.gguf")):
            stat = f.stat()
            models.append(
                {
                    "filename": f.name,
                    "path": str(f),
                    "size_bytes": stat.st_size,
                    "quantization": _parse_quant(f.name),
                    "loaded": (
                        str(f) == _state["model"]
                        and _state["status"] == "ready"
                    ),
                }
            )
    return {
        "models": models,
        "loaded_model": _state["model"] if _state["status"] in {"ready", "loading"} else "",
        "ctx_size": _state["ctx_size"],
        "n_gpu_layers": _state["n_gpu_layers"],
        "status": _state["status"],
    }


# ---- Download ---------------------------------------------------------------


class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None  # inferred from URL tail if not supplied


@app.post(
    "/api/models/download",
    status_code=202,
    summary="Download a GGUF model (admin)",
)
async def download_model(
    req: DownloadRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    """
    Enqueue a model download from *url* into MODELS_DIR.

    The download runs in the background; poll
    `GET /api/models/download/{task_id}` for progress.

    * Validates that the destination filename ends with `.gguf`.
    * Rejects the request when free disk space is below 1 GiB.
    * After the download completes the file is verified against the GGUF
      magic bytes (`GGUF`); invalid files are removed automatically.
    """
    _require_admin(authorization)

    # Derive destination filename; accept only the bare basename.
    raw_name = req.filename or Path(urllib.parse.urlparse(req.url).path).name
    safe_filename = os.path.basename(raw_name)
    if not safe_filename or safe_filename in {".", ".."}:
        raise HTTPException(
            status_code=400,
            detail="Cannot infer filename from URL; provide 'filename' in the request body",
        )
    if safe_filename != raw_name:
        raise HTTPException(status_code=400, detail="filename must not contain path separators")
    if not safe_filename.lower().endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Destination filename must end with .gguf")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Check for conflicts via a directory scan so we never use a user-derived
    # path in a file-existence check (avoids path-injection taint).
    if MODELS_DIR.is_dir() and any(
        f.name == safe_filename for f in MODELS_DIR.iterdir() if f.is_file()
    ):
        raise HTTPException(status_code=409, detail=f"Model already exists: {safe_filename}")
    # Build destination from the trusted directory and the sanitized basename.
    dest = MODELS_DIR / safe_filename

    free = _free_bytes(MODELS_DIR)
    if free < 1_073_741_824:  # 1 GiB
        raise HTTPException(
            status_code=507,
            detail=(
                f"Insufficient disk space: {free // (1024 ** 3)} GiB free, "
                "need at least 1 GiB"
            ),
        )

    task_id = str(uuid.uuid4())
    _downloads[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "filename": safe_filename,
        "bytes_downloaded": 0,
        "total_bytes": None,
        "error": None,
    }
    background_tasks.add_task(_do_download, task_id, req.url, dest)
    return _downloads[task_id]


@app.get(
    "/api/models/download/{task_id}",
    summary="Poll download progress",
)
def download_status(task_id: str) -> dict:
    """Return the current progress of an in-flight or completed download task."""
    if task_id not in _downloads:
        raise HTTPException(status_code=404, detail="Download task not found")
    return _downloads[task_id]


async def _do_download(task_id: str, url: str, dest: Path) -> None:
    info = _downloads[task_id]
    info["status"] = "downloading"
    tmp = dest.with_suffix(".tmp")
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=30.0,
                read=DOWNLOAD_READ_TIMEOUT,
                write=DOWNLOAD_READ_TIMEOUT,
                pool=30.0,
            ),
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total = response.headers.get("content-length")
                info["total_bytes"] = int(total) if total else None
                with open(tmp, "wb") as fh:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        fh.write(chunk)
                        info["bytes_downloaded"] += len(chunk)

        # Validate GGUF magic bytes (0x47 0x47 0x55 0x46 == "GGUF")
        with open(tmp, "rb") as fh:
            magic = fh.read(4)
        if magic != b"GGUF":
            tmp.unlink(missing_ok=True)
            info["status"] = "error"
            info["error"] = "Downloaded file is not a valid GGUF file (bad magic bytes)"
            return

        tmp.rename(dest)
        info["status"] = "done"
        log.info("Model download complete: %s", dest.name)

    except httpx.HTTPStatusError as exc:
        tmp.unlink(missing_ok=True)
        info["status"] = "error"
        info["error"] = f"Download failed: HTTP {exc.response.status_code}"
        log.error("Model download failed (task=%s): %s", task_id, exc)
    except httpx.RequestError as exc:
        tmp.unlink(missing_ok=True)
        info["status"] = "error"
        info["error"] = f"Download failed: network error ({exc.__class__.__name__}: {exc})"
        log.error("Model download failed (task=%s): network error", task_id, exc_info=True)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        info["status"] = "error"
        info["error"] = f"Download failed: disk write error ({exc.__class__.__name__}: {exc})"
        log.error("Model download failed (task=%s): disk write error", task_id, exc_info=True)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        info["status"] = "error"
        info["error"] = f"Download failed: unexpected error ({exc.__class__.__name__}: {exc})"
        log.error("Model download failed (task=%s)", task_id, exc_info=True)


# ---- Load / hot-swap --------------------------------------------------------


class LoadRequest(BaseModel):
    filename: str  # filename within MODELS_DIR, e.g. "mistral-7b-v0.3.Q4_K_M.gguf"
    ctx_size: Optional[int] = Field(default=None, ge=256, le=262144)
    n_gpu_layers: Optional[int] = Field(default=None, ge=-1, le=200)


@app.post("/api/models/load", summary="Switch active model (admin)")
async def load_model(
    req: LoadRequest,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    """
    Switch llama-server to a different GGUF model.

    The current llama-server process is stopped and a new one is started with
    the requested model.

    **Readiness semantics:**
    * While the new model is loading, `GET /health` returns `{"status":"loading"}`
      and all proxy requests return HTTP 503.
    * Once ready, `GET /health` returns `{"status":"ok"}`.
    * On failure, `GET /health` returns `{"status":"error"}`.

    This endpoint does **not** guarantee zero-downtime; callers should poll
    `/health` and retry inference requests after the transition completes.
    """
    _require_admin(authorization)

    # Accept only the bare filename — no path separators.
    safe_filename = os.path.basename(req.filename)
    if not safe_filename or not safe_filename.lower().endswith(".gguf"):
        raise HTTPException(status_code=400, detail="filename must end with .gguf")
    if safe_filename != req.filename:
        raise HTTPException(status_code=400, detail="filename must not contain path separators")

    # Resolve model path via a directory scan — never construct a path from user
    # input directly so that the value passed to _start_llama is always trusted.
    model_path: Optional[Path] = next(
        (f for f in MODELS_DIR.iterdir() if f.is_file() and f.name == safe_filename),
        None,
    )
    if model_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found in {MODELS_DIR}: {safe_filename}",
        )

    log.info("Switching model to: %s", safe_filename)
    _state["status"] = "loading"
    _state["model"] = str(model_path)
    _state["ctx_size"] = req.ctx_size or int(_state["ctx_size"]) or INITIAL_CTX_SIZE
    _state["n_gpu_layers"] = (
        req.n_gpu_layers
        if req.n_gpu_layers is not None
        else int(_state["n_gpu_layers"])
    )
    _state["error"] = None

    _stop_llama()
    try:
        proc = _start_llama(
            str(model_path),
            int(_state["ctx_size"]),
            int(_state["n_gpu_layers"]),
        )
        _state["process"] = proc
    except Exception as exc:
        _state["status"] = "error"
        _state["error"] = "Failed to start llama-server"
        log.error("Failed to start llama-server with model %s: %s", req.filename, exc)
        return {
            "loaded_model": _state["model"],
            "ctx_size": _state["ctx_size"],
            "n_gpu_layers": _state["n_gpu_layers"],
            "status": _state["status"],
            "error": _state["error"],
        }

    ok = await _wait_for_llama()
    _state["status"] = "ready" if ok else "error"
    if not ok:
        _state["error"] = "llama-server did not become healthy after model switch"

    return {
        "loaded_model": _state["model"],
        "ctx_size": _state["ctx_size"],
        "n_gpu_layers": _state["n_gpu_layers"],
        "status": _state["status"],
        "error": _state.get("error"),
    }


@app.post("/api/models/unload", summary="Unload the active model (admin)")
async def unload_model(
    authorization: Optional[str] = Header(default=None),
) -> dict:
    """Stop llama-server and release the currently active model from service state."""
    _require_admin(authorization)

    had_model = _state["status"] in {"ready", "loading"} and bool(_state["model"])
    previous_model = _state["model"]
    _stop_llama()
    _state["model"] = ""
    _state["status"] = "no-model"
    _state["error"] = "Model unloaded by operator" if had_model else "No model loaded"

    log.info(
        "Model unloaded%s",
        f": {Path(previous_model).name}" if previous_model else "",
    )

    return {
        "loaded_model": "",
        "ctx_size": _state["ctx_size"],
        "n_gpu_layers": _state["n_gpu_layers"],
        "status": _state["status"],
        "error": _state["error"],
    }

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health", summary="Service readiness probe")
def health() -> Response:
    """
    Returns the current readiness state of the wrapper.

    * `{"status":"loading"}` — llama-server is starting or a model switch is in progress.
    * `{"status":"no-model"}` — the wrapper is healthy but no initial GGUF file exists yet.
    * `{"status":"ok"}`      — llama-server is ready to serve inference requests.
    * `{"status":"error"}`   — llama-server failed to start or become healthy.
    """
    if _state["status"] == "loading":
        return JSONResponse({
            "status": "loading",
            "ctx_size": _state["ctx_size"],
            "n_gpu_layers": _state["n_gpu_layers"],
        })
    if _state["status"] == "no-model":
        return JSONResponse(
            {
                "status": "no-model",
                "detail": _state.get("error", ""),
                "ctx_size": _state["ctx_size"],
                "n_gpu_layers": _state["n_gpu_layers"],
            },
        )
    if _state["status"] == "error":
        return JSONResponse(
            {
                "status": "error",
                "detail": _state.get("error", ""),
                "ctx_size": _state["ctx_size"],
                "n_gpu_layers": _state["n_gpu_layers"],
            },
            status_code=503,
        )
    return JSONResponse({
        "status": "ok",
        "ctx_size": _state["ctx_size"],
        "n_gpu_layers": _state["n_gpu_layers"],
    })

# ---------------------------------------------------------------------------
# Transparent reverse proxy for all llama-server endpoints
# ---------------------------------------------------------------------------

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    summary="Proxy to llama-server",
    include_in_schema=False,
)
async def proxy(request: Request, path: str) -> Response:
    """Transparently forward all other requests to llama-server."""
    if _state["status"] == "no-model":
        return JSONResponse(
            {"error": "no model loaded; download or load a GGUF model first"},
            status_code=503,
        )
    if _state["status"] != "ready":
        return JSONResponse(
            {"error": "model is loading, please retry later"},
            status_code=503,
        )

    is_inference = path in _INFERENCE_PATHS

    # ------------------------------------------------------------------ #
    # Concurrency guard — serialise inference requests (one slot by        #
    # default) so that a long generation cannot starve other GPU work.     #
    # ------------------------------------------------------------------ #
    if is_inference:
        global _active_inference
        # asyncio is single-threaded: the check and increment below are
        # atomic with respect to other coroutines (no await in between).
        if _active_inference >= MAX_CONCURRENT_REQUESTS:
            return JSONResponse(
                {"error": "inference slot busy, please retry later"},
                status_code=429,
                headers={"Retry-After": "5"},
            )
        _active_inference += 1

    target_url = f"http://127.0.0.1:{LLAMA_PORT}/{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
    }
    body = await request.body()

    # ------------------------------------------------------------------ #
    # Token-budget enforcement — silently clamp max_tokens to MAX_TOKENS  #
    # so a single request cannot monopolise VRAM via a huge KV cache.     #
    # ------------------------------------------------------------------ #
    if is_inference and path in _TOKEN_BUDGET_PATHS and body:
        try:
            payload = json.loads(body)
            effective_max = payload.get("max_tokens")
            if effective_max is None or effective_max > MAX_TOKENS:
                payload["max_tokens"] = MAX_TOKENS
                body = json.dumps(payload).encode()
                fwd_headers["content-length"] = str(len(body))
                fwd_headers.setdefault("content-type", "application/json")
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass  # malformed body — let llama-server handle it

    # Use a finite read timeout only for GPU-bound inference requests.
    timeout = (
        httpx.Timeout(connect=30.0, read=REQUEST_TIMEOUT, write=30.0, pool=30.0)
        if is_inference
        else httpx.Timeout(None)
    )

    client = httpx.AsyncClient(timeout=timeout)
    try:
        upstream_req = client.build_request(
            method=request.method,
            url=target_url,
            headers=fwd_headers,
            content=body,
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        await client.aclose()
        if is_inference:
            _active_inference -= 1
        log.warning("Proxy connect error: %s", exc)
        return JSONResponse({"error": "backend unavailable"}, status_code=503)
    except httpx.ReadTimeout:
        await client.aclose()
        if is_inference:
            _active_inference -= 1
        log.warning("Inference request timed out after %.0fs", REQUEST_TIMEOUT)
        return JSONResponse({"error": "inference timeout"}, status_code=504)
    except Exception:
        await client.aclose()
        if is_inference:
            _active_inference -= 1
        raise

    resp_headers = {
        k: v
        for k, v in upstream_resp.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    async def body_gen():
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        except httpx.ReadTimeout:
            log.warning("Inference response timed out during streaming")
        finally:
            await upstream_resp.aclose()
            await client.aclose()
            if is_inference:
                global _active_inference
                _active_inference -= 1

    return StreamingResponse(
        body_gen(),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
