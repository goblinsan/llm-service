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
import datetime as dt
import html
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import urllib.parse
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

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
LLAMA_STARTUP_TIMEOUT: int = int(os.getenv("LLAMA_STARTUP_TIMEOUT", "300"))
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
_llama_log_tail: deque[str] = deque(maxlen=200)

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
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\/(\d+)\s+layers\s+to\s+GPU", re.IGNORECASE)
_FLASH_ATTN_RE = re.compile(r"flash_attn\s*=\s*(\d+)", re.IGNORECASE)
_CUDA_DEVICE_RE = re.compile(r"using device\s+([A-Z0-9_]+)\s+\((.+?)\)\s+-\s+(\d+)\s+MiB free", re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_ALT_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\>\s*call:(?P<name>[a-zA-Z0-9_]+)\s*(?P<arguments>\{.*\})",
    re.IGNORECASE | re.DOTALL,
)
_TIME_INTENT_RE = re.compile(r"\b(time|date|today|day|timezone|clock)\b", re.IGNORECASE)
_TIME_LOCATION_RE = re.compile(r"\b(?:i am|i'm|im)\s+in\s+([a-z0-9 ,.\-]+)$", re.IGNORECASE)
_DIRECT_TIME_REQUEST_RE = re.compile(
    r"^\s*(?:"
    r"what(?:'s| is)?\s+the\s+time(?:\s+is\s+it)?"
    r"|what\s+time\s+is\s+it"
    r"|what(?:'s| is)?\s+the\s+date"
    r"|what\s+is\s+today'?s\s+date"
    r"|today'?s\s+date"
    r"|what\s+day\s+is\s+it"
    r"|what\s+is\s+the\s+time\s+in\s+my\s+timezone"
    r"|what\s+time\s+is\s+it\s+in\s+my\s+timezone"
    r")"
    r"(?:\s*,?\s*(?:i am|i'm|im)\s+in\s+[a-z0-9 ,.\-]+)?"
    r"\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(?P<snippet_div>.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)
_US_STATE_ABBREVIATIONS = {
    "al": "Alabama",
    "ak": "Alaska",
    "az": "Arizona",
    "ar": "Arkansas",
    "ca": "California",
    "co": "Colorado",
    "ct": "Connecticut",
    "de": "Delaware",
    "fl": "Florida",
    "ga": "Georgia",
    "hi": "Hawaii",
    "id": "Idaho",
    "il": "Illinois",
    "in": "Indiana",
    "ia": "Iowa",
    "ks": "Kansas",
    "ky": "Kentucky",
    "la": "Louisiana",
    "me": "Maine",
    "md": "Maryland",
    "ma": "Massachusetts",
    "mi": "Michigan",
    "mn": "Minnesota",
    "ms": "Mississippi",
    "mo": "Missouri",
    "mt": "Montana",
    "ne": "Nebraska",
    "nv": "Nevada",
    "nh": "New Hampshire",
    "nj": "New Jersey",
    "nm": "New Mexico",
    "ny": "New York",
    "nc": "North Carolina",
    "nd": "North Dakota",
    "oh": "Ohio",
    "ok": "Oklahoma",
    "or": "Oregon",
    "pa": "Pennsylvania",
    "ri": "Rhode Island",
    "sc": "South Carolina",
    "sd": "South Dakota",
    "tn": "Tennessee",
    "tx": "Texas",
    "ut": "Utah",
    "vt": "Vermont",
    "va": "Virginia",
    "wa": "Washington",
    "wv": "West Virginia",
    "wi": "Wisconsin",
    "wy": "Wyoming",
    "dc": "District of Columbia",
}

_TOOL_PROMPT_TEMPLATE = """Built-in tools are available for this conversation.

When you need external information, respond with ONLY one XML block in exactly this format:
<tool_call>{{"name":"TOOL_NAME","arguments":{{...}}}}</tool_call>

Available tools:
{tool_definitions}

Rules:
- Emit only one tool call at a time.
- Do not answer normally in the same message as a tool call.
- After you receive a system message beginning with TOOL RESULT, either answer directly or emit one more tool call if you still need more information.
- Prefer `time_now` for date/time questions.
- If a user gives a location like a city or state, pass it as `{{"location":"..."}}` to `time_now`.
- For web questions, use `web_search` first. If the search snippets are insufficient, use `web_read` on the most relevant result before answering.
- Do not invent tool results.
"""


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


def _record_llama_output(proc: subprocess.Popen) -> None:
    stream = proc.stdout
    if stream is None:
        return
    try:
        for raw_line in stream:
            line = raw_line.rstrip()
            if not line:
                continue
            _llama_log_tail.append(line)
            log.info("[llama] %s", line)
    except Exception as exc:
        log.warning("Failed to read llama-server output: %s", exc)


def _describe_llama_failure(timeout: Optional[int] = None) -> str:
    proc: Optional[subprocess.Popen] = _state.get("process")
    if proc is not None:
        code = proc.poll()
        if code is not None:
            if _llama_log_tail:
                return f"llama-server exited early with code {code}: {_llama_log_tail[-1]}"
            return f"llama-server exited early with code {code}"
    if timeout is not None:
        if _llama_log_tail:
            return f"llama-server did not become healthy within {timeout}s: {_llama_log_tail[-1]}"
        return f"llama-server did not become healthy within {timeout}s"
    if _llama_log_tail:
        return _llama_log_tail[-1]
    return "llama-server failed to start"


def _get_llama_diagnostics() -> dict:
    diagnostics: dict[str, object] = {
        "offloaded_layers": None,
        "total_layers": None,
        "flash_attn": None,
        "cuda_device": None,
        "cuda_device_name": None,
        "cuda_free_mib_at_load": None,
        "recent_log_tail": list(_llama_log_tail)[-20:],
    }

    for line in _llama_log_tail:
        offload_match = _OFFLOAD_RE.search(line)
        if offload_match:
            diagnostics["offloaded_layers"] = int(offload_match.group(1))
            diagnostics["total_layers"] = int(offload_match.group(2))
            continue

        flash_attn_match = _FLASH_ATTN_RE.search(line)
        if flash_attn_match:
            diagnostics["flash_attn"] = flash_attn_match.group(1) == "1"
            continue

        cuda_device_match = _CUDA_DEVICE_RE.search(line)
        if cuda_device_match:
            diagnostics["cuda_device"] = cuda_device_match.group(1)
            diagnostics["cuda_device_name"] = cuda_device_match.group(2)
            diagnostics["cuda_free_mib_at_load"] = int(cuda_device_match.group(3))

    return diagnostics


def _tool_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("gateway_tools")
    if not isinstance(raw, dict):
        return {"enabled": False, "time": False, "web_search": False, "client_timezone": None}
    enabled = bool(raw.get("enabled"))
    client_timezone = raw.get("client_timezone")
    return {
        "enabled": enabled,
        "time": enabled and bool(raw.get("time")),
        "web_search": enabled and bool(raw.get("web_search")),
        "client_timezone": client_timezone if isinstance(client_timezone, str) and client_timezone.strip() else None,
    }


def _tool_system_message(config: dict[str, bool]) -> str:
    tool_defs: list[str] = []
    if config.get("time"):
        tool_defs.append('- `time_now`: get the current date and time. Optional arguments: {"timezone":"Area/City"} or {"location":"city, state/country"}')
    if config.get("web_search"):
        tool_defs.append('- `web_search`: search the public web. Required arguments: {"query":"search terms"}')
        tool_defs.append('- `web_read`: fetch and read a web page. Required arguments: {"url":"https://..."}')
    return _TOOL_PROMPT_TEMPLATE.format(
        tool_definitions="\n".join(tool_defs) if tool_defs else "- none",
    )


def _extract_text_content(chat_response: dict[str, Any]) -> str:
    choices = chat_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _latest_user_text(messages: list[Any]) -> Optional[str]:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
    return None


def _extract_location_hint(user_text: str) -> Optional[str]:
    match = _TIME_LOCATION_RE.search(user_text.strip())
    if match:
        return match.group(1).strip(" .")
    return None


def _is_simple_time_request(user_text: str) -> bool:
    return bool(_DIRECT_TIME_REQUEST_RE.match(user_text.strip()))


def _merge_usage(total: dict[str, int], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _parse_tool_call(text: str, config: dict[str, bool]) -> Optional[dict[str, Any]]:
    match = _TOOL_CALL_RE.search(text)
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        name = payload.get("name")
        arguments = payload.get("arguments", {})
    else:
        alt_match = _ALT_TOOL_CALL_RE.search(text)
        if not alt_match:
            return None
        name = alt_match.group("name")
        try:
            arguments = json.loads(alt_match.group("arguments"))
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    allowed = set()
    if config.get("time"):
        allowed.add("time_now")
    if config.get("web_search"):
        allowed.add("web_search")
        allowed.add("web_read")
    if name not in allowed:
        return {"name": name, "arguments": arguments, "error": f"Tool '{name}' is not enabled"}
    return {"name": name, "arguments": arguments}


def _strip_html(text: str) -> str:
    plain = re.sub(r"<[^>]+>", "", text)
    return html.unescape(re.sub(r"\s+", " ", plain)).strip()


def _normalise_search_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        query = urllib.parse.parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return urllib.parse.unquote(uddg[0])
    return html.unescape(url)


def _location_search_candidates(location: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", location.strip(" .")).strip()
    if not cleaned:
        return []
    candidates = [cleaned]
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) >= 2:
        state_token = re.sub(r"[^a-z]", "", parts[-1].lower())
        full_state = _US_STATE_ABBREVIATIONS.get(state_token)
        if full_state:
            expanded = ", ".join([*parts[:-1], full_state])
            if expanded not in candidates:
                candidates.append(expanded)
            expanded_us = f"{expanded}, United States"
            if expanded_us not in candidates:
                candidates.append(expanded_us)
    elif cleaned.lower() not in candidates:
        candidates.append(cleaned.lower())
    return candidates


async def _perform_web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0 (llm-service web search)"}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=10.0),
        follow_redirects=True,
    ) as client:
        response = await client.get("https://html.duckduckgo.com/html/", params=params, headers=headers)
        response.raise_for_status()
    html_text = response.text
    title_matches = list(_DDG_RESULT_RE.finditer(html_text))
    snippet_matches = list(_DDG_SNIPPET_RE.finditer(html_text))
    results: list[dict[str, str]] = []
    for idx, title_match in enumerate(title_matches[:max_results]):
        snippet = ""
        if idx < len(snippet_matches):
            snippet = _strip_html(
                snippet_matches[idx].group("snippet") or snippet_matches[idx].group("snippet_div") or ""
            )
        results.append(
            {
                "title": _strip_html(title_match.group("title")),
                "url": _normalise_search_url(title_match.group("url")),
                "snippet": snippet,
            }
        )
    return {"query": query, "results": results}


async def _resolve_location_timezone(location: str) -> dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0 (llm-service geocoding)"}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=10.0),
        follow_redirects=True,
    ) as client:
        for candidate in _location_search_candidates(location):
            params = {"name": candidate, "count": 1, "language": "en", "format": "json"}
            response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results")
            if not isinstance(results, list) or not results:
                continue
            first = results[0]
            timezone = first.get("timezone")
            if not isinstance(timezone, str) or not timezone:
                continue
            parts = [first.get("name"), first.get("admin1"), first.get("country")]
            label = ", ".join([part for part in parts if isinstance(part, str) and part])
            return {"timezone": timezone, "resolved_location": label or candidate}
    raise ValueError(f"Could not resolve location '{location}'")


async def _perform_web_read(url: str) -> dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0 (llm-service web read)"}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=10.0),
        follow_redirects=True,
    ) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    text = response.text
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    title = _strip_html(title_match.group(1)) if title_match else url
    cleaned = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
    cleaned = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", cleaned)
    cleaned = _strip_html(cleaned)
    if len(cleaned) > 4000:
        cleaned = cleaned[:4000].rsplit(" ", 1)[0] + "..."
    return {
        "url": url,
        "content_type": content_type,
        "title": title,
        "content": cleaned,
    }


async def _time_tool_result(arguments: dict[str, Any]) -> dict[str, str]:
    timezone_name = arguments.get("timezone")
    location = arguments.get("location")
    client_timezone = arguments.get("client_timezone")
    now_utc = dt.datetime.now(dt.timezone.utc)
    result = {"utc": now_utc.isoformat()}
    if isinstance(timezone_name, str) and timezone_name.strip():
        tz = ZoneInfo(timezone_name.strip())
        result["timezone"] = timezone_name.strip()
        result["local"] = now_utc.astimezone(tz).isoformat()
    elif isinstance(location, str) and location.strip():
        try:
            resolved = await _resolve_location_timezone(location.strip())
            tz = ZoneInfo(resolved["timezone"])
            result["timezone"] = resolved["timezone"]
            result["resolved_location"] = resolved["resolved_location"]
            result["local"] = now_utc.astimezone(tz).isoformat()
        except ValueError:
            if isinstance(client_timezone, str) and client_timezone.strip():
                tz = ZoneInfo(client_timezone.strip())
                result["timezone"] = client_timezone.strip()
                result["local"] = now_utc.astimezone(tz).isoformat()
                result["resolved_location"] = location.strip()
            else:
                raise
    else:
        local_now = dt.datetime.now().astimezone()
        result["timezone"] = str(local_now.tzinfo) if local_now.tzinfo else "local"
        result["local"] = local_now.isoformat()
    result["date"] = result["local"][:10]
    result["time"] = result["local"][11:19]
    return result


async def _execute_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    if tool_call.get("error"):
        return {"ok": False, "error": tool_call["error"]}
    name = tool_call["name"]
    arguments = tool_call.get("arguments", {})
    try:
        if name == "time_now":
            return {"ok": True, "name": name, "result": await _time_tool_result(arguments)}
        if name == "web_search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return {"ok": False, "name": name, "error": "web_search requires a non-empty 'query' string"}
            return {"ok": True, "name": name, "result": await _perform_web_search(query.strip())}
        if name == "web_read":
            url = arguments.get("url")
            if not isinstance(url, str) or not url.strip():
                return {"ok": False, "name": name, "error": "web_read requires a non-empty 'url' string"}
            return {"ok": True, "name": name, "result": await _perform_web_read(url.strip())}
        return {"ok": False, "name": name, "error": f"Unsupported tool '{name}'"}
    except Exception as exc:
        log.warning("Tool execution failed for %s: %s", name, exc)
        return {"ok": False, "name": name, "error": f"{exc.__class__.__name__}: {exc}"}


async def _call_llama_chat(payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
    target_url = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=REQUEST_TIMEOUT, write=30.0, pool=30.0)
    ) as client:
        response = await client.post(target_url, json=payload)
    try:
        data = response.json()
    except ValueError:
        data = {"error": response.text}
    headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    return response.status_code, headers, data


def _sse_response_from_chat_json(chat_response: dict[str, Any]) -> StreamingResponse:
    content = _extract_text_content(chat_response)
    model = chat_response.get("model", "local")
    response_id = chat_response.get("id", f"chatcmpl-{uuid.uuid4().hex}")
    created = int(chat_response.get("created") or dt.datetime.now().timestamp())
    usage = chat_response.get("usage")

    async def _emit():
        first_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n".encode()
        if isinstance(usage, dict):
            usage_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n".encode()
        done_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_emit(), media_type="text/event-stream")


def _chat_response_from_text(
    content: str,
    *,
    model: str = "local",
    usage: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(dt.datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage:
        response["usage"] = usage
    return response


def _format_time_answer(result: dict[str, str], user_text: str) -> str:
    local_raw = result.get("local")
    if not local_raw:
        return result.get("utc", "")
    local_dt = dt.datetime.fromisoformat(local_raw)
    wants_date = bool(re.search(r"\b(date|today|day)\b", user_text, re.IGNORECASE))
    wants_time = bool(re.search(r"\btime|clock|timezone\b", user_text, re.IGNORECASE))
    time_part = local_dt.strftime("%-I:%M:%S %p")
    tz_name = result.get("timezone") or (local_dt.tzname() or "")
    date_part = local_dt.strftime("%B %-d, %Y")
    if wants_time and wants_date:
        return f"{time_part} {tz_name} on {date_part}".strip()
    if wants_date and not wants_time:
        return date_part
    return f"{time_part} {tz_name}".strip()


async def _maybe_handle_direct_time_request(
    payload: dict[str, Any],
    config: dict[str, bool],
) -> Optional[dict[str, Any]]:
    if not config.get("time"):
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    user_text = _latest_user_text(messages)
    if not user_text or not _TIME_INTENT_RE.search(user_text):
        return None
    if not _is_simple_time_request(user_text):
        return None
    lowered = user_text.lower()
    if "holiday" in lowered or "search" in lowered or "headline" in lowered or "news" in lowered:
        return None
    location = _extract_location_hint(user_text)
    if "my timezone" in lowered and not location:
        if not config.get("client_timezone"):
            return None
        tool_result = await _time_tool_result({"timezone": config["client_timezone"]})
    else:
        tool_args: dict[str, Any] = {}
        if location:
            tool_args["location"] = location
            if config.get("client_timezone"):
                tool_args["client_timezone"] = config["client_timezone"]
        elif config.get("client_timezone") and ("my timezone" in lowered):
            tool_args["timezone"] = config["client_timezone"]
        tool_result = await _time_tool_result(tool_args)
    return _chat_response_from_text(
        _format_time_answer(tool_result, user_text),
        model=str(payload.get("model", "local")),
    )


async def _handle_builtin_tool_chat(payload: dict[str, Any]) -> Response:
    stream_requested = bool(payload.get("stream"))
    config = _tool_config_from_payload(payload)
    stripped_payload = {k: v for k, v in payload.items() if k != "gateway_tools"}
    stripped_payload["stream"] = False

    base_messages = payload.get("messages")
    if not isinstance(base_messages, list):
        return JSONResponse({"error": "messages must be an array"}, status_code=400)

    direct_time_response = await _maybe_handle_direct_time_request(payload, config)
    if direct_time_response is not None:
        return _sse_response_from_chat_json(direct_time_response) if stream_requested else JSONResponse(direct_time_response)

    tool_prompt = {"role": "system", "content": _tool_system_message(config)}
    conversation = [tool_prompt, *base_messages]
    total_usage: dict[str, int] = {}
    last_response: dict[str, Any] | None = None

    for _ in range(5):
        llama_payload = {**stripped_payload, "messages": conversation}
        status_code, _, chat_response = await _call_llama_chat(llama_payload)
        if status_code != 200:
            return JSONResponse(chat_response, status_code=status_code)
        last_response = chat_response
        _merge_usage(total_usage, chat_response.get("usage"))
        content = _extract_text_content(chat_response)
        tool_call = _parse_tool_call(content, config)
        if not tool_call:
            if total_usage:
                chat_response["usage"] = total_usage
            return _sse_response_from_chat_json(chat_response) if stream_requested else JSONResponse(chat_response)

        tool_result = await _execute_tool_call(tool_call)
        conversation.extend(
            [
                {"role": "assistant", "content": content},
                {
                    "role": "system",
                    "content": (
                        f"TOOL RESULT ({tool_call.get('name', 'unknown')}):\n"
                        f"{json.dumps(tool_result, ensure_ascii=True)}\n"
                        "Use this result to answer the user directly."
                    ),
                },
            ]
        )

    fallback = last_response or {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(dt.datetime.now().timestamp()),
        "model": stripped_payload.get("model", "local"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I hit the tool-call limit for this request. Please narrow the question and retry.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    if total_usage:
        fallback["usage"] = total_usage
    return _sse_response_from_chat_json(fallback) if stream_requested else JSONResponse(fallback)


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
    _llama_log_tail.clear()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_record_llama_output, args=(proc,), daemon=True).start()
    return proc


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


async def _wait_for_llama(timeout: int = LLAMA_STARTUP_TIMEOUT) -> tuple[bool, Optional[str]]:
    """Poll llama-server /health until it returns HTTP 200 or *timeout* seconds elapse."""
    url = f"http://127.0.0.1:{LLAMA_PORT}/health"
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    async with httpx.AsyncClient() as client:
        while loop.time() < deadline:
            proc: Optional[subprocess.Popen] = _state.get("process")
            if proc is not None and proc.poll() is not None:
                detail = _describe_llama_failure()
                log.error(detail)
                return False, detail
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200:
                    log.info("llama-server is healthy")
                    return True, None
            except Exception:
                pass
            await asyncio.sleep(1)
    detail = _describe_llama_failure(timeout)
    log.error(detail)
    return False, detail

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

    ok, detail = await _wait_for_llama()
    if ok:
        _state["status"] = "ready"
        _state["error"] = None
    else:
        _state["status"] = "error"
        _state["error"] = detail or "llama-server did not become healthy during startup"


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
        "llama": _get_llama_diagnostics(),
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
            "llama": _get_llama_diagnostics(),
        }

    ok, detail = await _wait_for_llama()
    _state["status"] = "ready" if ok else "error"
    if not ok:
        _state["error"] = detail or "llama-server did not become healthy after model switch"

    return {
        "loaded_model": _state["model"],
        "ctx_size": _state["ctx_size"],
        "n_gpu_layers": _state["n_gpu_layers"],
        "status": _state["status"],
        "error": _state.get("error"),
        "llama": _get_llama_diagnostics(),
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
        "llama": _get_llama_diagnostics(),
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
            "llama": _get_llama_diagnostics(),
        })
    if _state["status"] == "no-model":
        return JSONResponse(
            {
                "status": "no-model",
                "detail": _state.get("error", ""),
                "ctx_size": _state["ctx_size"],
                "n_gpu_layers": _state["n_gpu_layers"],
                "llama": _get_llama_diagnostics(),
            },
        )
    if _state["status"] == "error":
        return JSONResponse(
            {
                "status": "error",
                "detail": _state.get("error", ""),
                "ctx_size": _state["ctx_size"],
                "n_gpu_layers": _state["n_gpu_layers"],
                "llama": _get_llama_diagnostics(),
            },
            status_code=503,
        )
    return JSONResponse({
        "status": "ok",
        "ctx_size": _state["ctx_size"],
        "n_gpu_layers": _state["n_gpu_layers"],
        "llama": _get_llama_diagnostics(),
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
    payload: Optional[dict[str, Any]] = None

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

    if (
        path == "v1/chat/completions"
        and isinstance(payload, dict)
        and _tool_config_from_payload(payload).get("enabled")
    ):
        try:
            return await _handle_builtin_tool_chat(payload)
        except Exception as exc:
            log.exception("Built-in tool chat failed")
            return JSONResponse(
                {
                    "error": "builtin tool execution failed",
                    "detail": f"{exc.__class__.__name__}: {exc}",
                },
                status_code=500,
            )
        finally:
            if is_inference:
                _active_inference -= 1

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
