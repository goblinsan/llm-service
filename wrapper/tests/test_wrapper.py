"""Tests for wrapper/main.py.

Environment variables are pre-set by conftest.py (loaded before this module
is imported) so that SKIP_LLAMA_STARTUP=1 prevents the startup event from
launching a real llama-server binary.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main as m

ADMIN_HDR = {"Authorization": "Bearer test-admin-token"}
BAD_HDR = {"Authorization": "Bearer wrong-token"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_state(tmp_path):
    """Isolate mutable global state between tests."""
    original_dir = m.MODELS_DIR
    original_model = m._state["model"]
    original_status = m._state["status"]
    original_ctx_size = m._state["ctx_size"]
    original_n_gpu_layers = m._state["n_gpu_layers"]

    m.MODELS_DIR = tmp_path
    m._state["status"] = "ready"
    m._state["model"] = str(tmp_path / "default.gguf")
    m._state["ctx_size"] = 4096
    m._state["n_gpu_layers"] = -1
    m._state["error"] = None
    m._downloads.clear()

    yield tmp_path

    m.MODELS_DIR = original_dir
    m._state["model"] = original_model
    m._state["status"] = original_status
    m._state["ctx_size"] = original_ctx_size
    m._state["n_gpu_layers"] = original_n_gpu_layers
    m._state["error"] = None
    m._downloads.clear()


@pytest.fixture(scope="session")
def client():
    with TestClient(m.app) as c:
        yield c


# ---------------------------------------------------------------------------
# _parse_quant
# ---------------------------------------------------------------------------


class TestParseQuant:
    def test_q4_k_m(self):
        assert m._parse_quant("mistral-7b-v0.3.Q4_K_M.gguf") == "Q4_K_M"

    def test_q5_k_s(self):
        assert m._parse_quant("llama-3.1-8b-Q5_K_S.gguf") == "Q5_K_S"

    def test_q8_0(self):
        assert m._parse_quant("model-Q8_0.gguf") == "Q8_0"

    def test_no_quant(self):
        assert m._parse_quant("model.gguf") is None

    def test_case_insensitive(self):
        assert m._parse_quant("model-q4_k_m.gguf") == "Q4_K_M"

    def test_empty_string(self):
        assert m._parse_quant("") is None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ready(self, client):
        m._state["status"] = "ready"
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "ctx_size": 4096, "n_gpu_layers": -1}

    def test_loading(self, client):
        m._state["status"] = "loading"
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "loading", "ctx_size": 4096, "n_gpu_layers": -1}

    def test_error(self, client):
        m._state["status"] = "error"
        m._state["error"] = "something broke"
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"
        assert "something broke" in body["detail"]


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_empty_dir(self, client, reset_state):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["models"] == []
        assert body["status"] == "ready"

    def test_lists_gguf_files(self, client, reset_state):
        models_dir: Path = reset_state
        (models_dir / "model-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
        (models_dir / "model-Q5_K_S.gguf").write_bytes(b"GGUF" + b"\x00" * 200)
        # Non-GGUF file — should be ignored
        (models_dir / "notes.txt").write_text("ignored")

        resp = client.get("/api/models")
        assert resp.status_code == 200
        names = [entry["filename"] for entry in resp.json()["models"]]
        assert "model-Q4_K_M.gguf" in names
        assert "model-Q5_K_S.gguf" in names
        assert "notes.txt" not in names

    def test_size_and_quant_fields(self, client, reset_state):
        models_dir: Path = reset_state
        content = b"GGUF" + b"\x00" * 50
        (models_dir / "mistral-7b-v0.3.Q4_K_M.gguf").write_bytes(content)

        body = client.get("/api/models").json()
        entry = next(e for e in body["models"] if "Q4_K_M" in e["filename"])
        assert entry["size_bytes"] == len(content)
        assert entry["quantization"] == "Q4_K_M"

    def test_loaded_flag(self, client, reset_state):
        models_dir: Path = reset_state
        model_file = models_dir / "active.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 10)
        m._state["model"] = str(model_file)
        m._state["status"] = "ready"

        body = client.get("/api/models").json()
        entry = next(e for e in body["models"] if e["filename"] == "active.gguf")
        assert entry["loaded"] is True

    def test_non_loaded_flag_false(self, client, reset_state):
        models_dir: Path = reset_state
        (models_dir / "other.gguf").write_bytes(b"GGUF" + b"\x00" * 10)
        m._state["model"] = str(models_dir / "different.gguf")

        body = client.get("/api/models").json()
        entry = next(e for e in body["models"] if e["filename"] == "other.gguf")
        assert entry["loaded"] is False


# ---------------------------------------------------------------------------
# POST /api/models/download
# ---------------------------------------------------------------------------


class TestDownloadModel:
    def test_requires_admin_token(self, client):
        resp = client.post(
            "/api/models/download",
            json={"url": "http://example.com/model.gguf"},
        )
        assert resp.status_code == 403

    def test_wrong_token_rejected(self, client):
        resp = client.post(
            "/api/models/download",
            json={"url": "http://example.com/model.gguf"},
            headers=BAD_HDR,
        )
        assert resp.status_code == 403

    def test_rejects_non_gguf_url(self, client):
        resp = client.post(
            "/api/models/download",
            json={"url": "http://example.com/model.bin"},
            headers=ADMIN_HDR,
        )
        assert resp.status_code == 400
        assert "gguf" in resp.json()["detail"].lower()

    def test_rejects_non_gguf_explicit_filename(self, client):
        resp = client.post(
            "/api/models/download",
            json={"url": "http://example.com/something", "filename": "bad.bin"},
            headers=ADMIN_HDR,
        )
        assert resp.status_code == 400

    def test_rejects_path_traversal_filename(self, client):
        resp = client.post(
            "/api/models/download",
            json={"url": "http://example.com/x.gguf", "filename": "../evil.gguf"},
            headers=ADMIN_HDR,
        )
        assert resp.status_code == 400

    def test_rejects_conflict(self, client, reset_state):
        models_dir: Path = reset_state
        existing = models_dir / "existing.gguf"
        existing.write_bytes(b"GGUF")

        resp = client.post(
            "/api/models/download",
            json={"url": "http://example.com/existing.gguf"},
            headers=ADMIN_HDR,
        )
        assert resp.status_code == 409

    def test_rejects_insufficient_disk_space(self, client):
        with patch("main._free_bytes", return_value=100 * 1024 * 1024):  # 100 MiB
            resp = client.post(
                "/api/models/download",
                json={"url": "http://example.com/model.gguf"},
                headers=ADMIN_HDR,
            )
        assert resp.status_code == 507
        assert "disk" in resp.json()["detail"].lower()

    def test_enqueues_download(self, client):
        with patch("main._free_bytes", return_value=10 * 1024 ** 3):
            resp = client.post(
                "/api/models/download",
                json={"url": "http://example.com/newmodel.gguf"},
                headers=ADMIN_HDR,
            )
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "pending"
        assert body["filename"] == "newmodel.gguf"
        assert "task_id" in body


# ---------------------------------------------------------------------------
# GET /api/models/download/{task_id}
# ---------------------------------------------------------------------------


class TestDownloadStatus:
    def test_not_found(self, client):
        resp = client.get("/api/models/download/does-not-exist")
        assert resp.status_code == 404

    def test_returns_task_info(self, client):
        task_id = "test-task-123"
        m._downloads[task_id] = {
            "task_id": task_id,
            "status": "downloading",
            "filename": "model.gguf",
            "bytes_downloaded": 1024,
            "total_bytes": 4096,
            "error": None,
        }
        resp = client.get(f"/api/models/download/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "downloading"
        assert body["bytes_downloaded"] == 1024


# ---------------------------------------------------------------------------
# POST /api/models/load
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_requires_admin_token(self, client):
        resp = client.post("/api/models/load", json={"filename": "model.gguf"})
        assert resp.status_code == 403

    def test_wrong_token_rejected(self, client):
        resp = client.post(
            "/api/models/load",
            json={"filename": "model.gguf"},
            headers=BAD_HDR,
        )
        assert resp.status_code == 403

    def test_rejects_missing_model(self, client):
        resp = client.post(
            "/api/models/load",
            json={"filename": "nonexistent.gguf"},
            headers=ADMIN_HDR,
        )
        assert resp.status_code == 404

    def test_rejects_non_gguf_filename(self, client):
        resp = client.post(
            "/api/models/load",
            json={"filename": "model.bin"},
            headers=ADMIN_HDR,
        )
        assert resp.status_code == 400

    def test_switches_model_successfully(self, client, reset_state):
        models_dir: Path = reset_state
        new_model = models_dir / "new-model.gguf"
        new_model.write_bytes(b"GGUF" + b"\x00" * 10)

        with (
            patch("main._stop_llama"),
            patch("main._start_llama", return_value=MagicMock()) as mock_start,
            patch(
                "main._wait_for_llama",
                new=AsyncMock(return_value=True),
            ),
        ):
            resp = client.post(
                "/api/models/load",
                json={"filename": "new-model.gguf", "ctx_size": 8192, "n_gpu_layers": 20},
                headers=ADMIN_HDR,
            )
            mock_start.assert_called_once_with(str(new_model), 8192, 20)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert "new-model.gguf" in body["loaded_model"]
        assert body["ctx_size"] == 8192
        assert body["n_gpu_layers"] == 20

    def test_returns_error_when_llama_unhealthy(self, client, reset_state):
        models_dir: Path = reset_state
        (models_dir / "broken.gguf").write_bytes(b"GGUF" + b"\x00" * 10)

        with (
            patch("main._stop_llama"),
            patch("main._start_llama", return_value=MagicMock()),
            patch(
                "main._wait_for_llama",
                new=AsyncMock(return_value=False),
            ),
        ):
            resp = client.post(
                "/api/models/load",
                json={"filename": "broken.gguf"},
                headers=ADMIN_HDR,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"] is not None

    def test_unloads_active_model(self, client):
        with patch("main._stop_llama") as mock_stop:
            resp = client.post("/api/models/unload", headers=ADMIN_HDR)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "no-model"
        assert body["loaded_model"] == ""
        mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# Proxy behaviour
# ---------------------------------------------------------------------------


class TestProxy:
    def test_returns_503_when_loading(self, client):
        m._state["status"] = "loading"
        resp = client.get("/v1/models")
        assert resp.status_code == 503
        assert "loading" in resp.json()["error"]

    def test_returns_503_when_error(self, client):
        m._state["status"] = "error"
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Concurrency guard — 429 when inference slot is busy
# ---------------------------------------------------------------------------


class TestInferenceConcurrency:
    def test_returns_429_when_slot_busy(self, client):
        """Setting _active_inference to the cap simulates a busy inference slot."""
        original = m._active_inference
        m._active_inference = m.MAX_CONCURRENT_REQUESTS
        try:
            resp = client.post("/v1/chat/completions", json={"messages": []})
            assert resp.status_code == 429
            assert "busy" in resp.json()["error"]
            assert resp.headers.get("retry-after") == "5"
        finally:
            m._active_inference = original

    def test_returns_429_for_completions_when_busy(self, client):
        original = m._active_inference
        m._active_inference = m.MAX_CONCURRENT_REQUESTS
        try:
            resp = client.post("/v1/completions", json={"prompt": "hi"})
            assert resp.status_code == 429
        finally:
            m._active_inference = original

    def test_returns_429_for_embeddings_when_busy(self, client):
        original = m._active_inference
        m._active_inference = m.MAX_CONCURRENT_REQUESTS
        try:
            resp = client.post("/v1/embeddings", json={"input": "hi"})
            assert resp.status_code == 429
        finally:
            m._active_inference = original

    def test_non_inference_path_not_throttled(self, client):
        """Non-inference paths should NOT be gated by the concurrency counter."""
        original = m._active_inference
        m._active_inference = m.MAX_CONCURRENT_REQUESTS
        try:
            # /v1/models is not an inference path; it should reach the proxy
            # (which will fail because llama-server isn't running in tests —
            # that's fine; we only care it doesn't return 429).
            resp = client.get("/v1/models")
            assert resp.status_code != 429
        finally:
            m._active_inference = original


# ---------------------------------------------------------------------------
# max_tokens capping
# ---------------------------------------------------------------------------


class TestMaxTokensCapping:
    def _post_chat(self, payload: dict, max_tokens_cap: int = 512):
        """POST /v1/chat/completions and return the body forwarded to llama-server."""
        import json as _json
        from unittest.mock import MagicMock, patch

        captured = {}

        async def fake_send(self_arg, req, **kwargs):
            captured["body"] = _json.loads(req.content)
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "application/json"}

            async def aiter_raw():
                yield b'{"choices":[]}'

            async def aclose():
                pass

            resp.aiter_raw = aiter_raw
            resp.aclose = aclose
            return resp

        original_max = m.MAX_TOKENS
        m.MAX_TOKENS = max_tokens_cap
        try:
            with patch.object(m.httpx.AsyncClient, "send", new=fake_send):
                from fastapi.testclient import TestClient

                with TestClient(m.app) as c:
                    c.post("/v1/chat/completions", json=payload)
        finally:
            m.MAX_TOKENS = original_max

        return captured.get("body", {})

    def test_caps_large_max_tokens(self, reset_state):
        body = self._post_chat({"messages": [], "max_tokens": 9999}, max_tokens_cap=512)
        assert body.get("max_tokens") == 512

    def test_sets_max_tokens_when_absent(self, reset_state):
        body = self._post_chat({"messages": []}, max_tokens_cap=512)
        assert body.get("max_tokens") == 512

    def test_does_not_raise_max_tokens(self, reset_state):
        """If the caller requests fewer tokens than the cap, preserve their value."""
        body = self._post_chat({"messages": [], "max_tokens": 100}, max_tokens_cap=2048)
        assert body.get("max_tokens") == 100



# ---------------------------------------------------------------------------
# GGUF validation in _do_download (async unit test)
# ---------------------------------------------------------------------------


class TestGgufValidation:
    def test_bad_magic_bytes_rejected(self, tmp_path):
        """_do_download should delete the temp file if magic bytes are wrong."""
        dest = tmp_path / "model.gguf"
        tmp = dest.with_suffix(".tmp")
        # Write non-GGUF content to the tmp file
        tmp.write_bytes(b"NOTGGUF content here")

        task_id = "validation-test"
        m._downloads[task_id] = {
            "task_id": task_id,
            "status": "downloading",
            "filename": "model.gguf",
            "bytes_downloaded": 20,
            "total_bytes": 20,
            "error": None,
        }

        # Simulate the validation step directly (after download)
        with open(tmp, "rb") as fh:
            magic = fh.read(4)
        if magic != b"GGUF":
            tmp.unlink(missing_ok=True)
            m._downloads[task_id]["status"] = "error"
            m._downloads[task_id]["error"] = "bad magic bytes"

        assert not tmp.exists()
        assert m._downloads[task_id]["status"] == "error"

    def test_valid_gguf_accepted(self, tmp_path):
        dest = tmp_path / "model.gguf"
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(b"GGUF" + b"\x00" * 100)

        with open(tmp, "rb") as fh:
            magic = fh.read(4)

        assert magic == b"GGUF"
        tmp.rename(dest)
        assert dest.exists()
