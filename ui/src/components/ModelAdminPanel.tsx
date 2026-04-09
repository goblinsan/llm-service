import { useEffect, useMemo, useState } from "react";
import type { DownloadTask, LoadResponse, ModelEntry } from "../types";

interface ModelAdminPanelProps {
  models: ModelEntry[];
  loadedModel: string;
  onChanged: () => void;
}

function formatBytes(value: number | null): string {
  if (value === null) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}

function buildAuthHeader(token: string): HeadersInit {
  return token.trim() ? { Authorization: `Bearer ${token.trim()}` } : {};
}

export function ModelAdminPanel({
  models,
  loadedModel,
  onChanged,
}: ModelAdminPanelProps) {
  const [adminToken, setAdminToken] = useState("");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [downloadFilename, setDownloadFilename] = useState("");
  const [downloadTask, setDownloadTask] = useState<DownloadTask | null>(null);
  const [loadingFilename, setLoadingFilename] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("llm_admin_token");
    if (stored) setAdminToken(stored);
  }, []);

  useEffect(() => {
    if (adminToken.trim()) {
      sessionStorage.setItem("llm_admin_token", adminToken.trim());
    } else {
      sessionStorage.removeItem("llm_admin_token");
    }
  }, [adminToken]);

  useEffect(() => {
    if (!downloadTask) return;
    if (downloadTask.status === "done" || downloadTask.status === "error") return;

    const timer = window.setTimeout(async () => {
      try {
        const res = await fetch(`/api/models/download/${downloadTask.task_id}`);
        const data: DownloadTask = await res.json();
        setDownloadTask(data);
        if (data.status === "done") {
          setMessage(`Downloaded ${data.filename}`);
          setError(null);
          onChanged();
        }
        if (data.status === "error") {
          setError(data.error || "Download failed");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to poll download status");
      }
    }, 1500);

    return () => window.clearTimeout(timer);
  }, [downloadTask, onChanged]);

  const activeModelName = useMemo(
    () => (loadedModel ? loadedModel.split("/").pop() || loadedModel : ""),
    [loadedModel]
  );

  async function handleDownload(e: React.FormEvent) {
    e.preventDefault();
    setMessage(null);
    setError(null);
    setDownloadTask(null);

    try {
      const res = await fetch("/api/models/download", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...buildAuthHeader(adminToken),
        },
        body: JSON.stringify({
          url: downloadUrl.trim(),
          filename: downloadFilename.trim() || undefined,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || data.error || `HTTP ${res.status}`);
      }
      setDownloadTask(data as DownloadTask);
      setDownloadUrl("");
      setDownloadFilename("");
      setMessage(`Started download for ${(data as DownloadTask).filename}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start download");
    }
  }

  async function handleLoad(filename: string) {
    setLoadingFilename(filename);
    setMessage(null);
    setError(null);
    try {
      const res = await fetch("/api/models/load", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...buildAuthHeader(adminToken),
        },
        body: JSON.stringify({ filename }),
      });
      const data = (await res.json().catch(() => ({}))) as LoadResponse & {
        detail?: string;
      };
      if (!res.ok) {
        throw new Error(data.detail || data.error || `HTTP ${res.status}`);
      }
      setMessage(`Switched active model to ${filename}`);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load model");
    } finally {
      setLoadingFilename(null);
    }
  }

  return (
    <section className="panel model-admin">
      <div className="panel-header">
        <h2>Model admin</h2>
      </div>

      <div className="form-row">
        <label htmlFor="admin-token">Admin token</label>
        <input
          id="admin-token"
          className="input"
          type="password"
          value={adminToken}
          onChange={(e) => setAdminToken(e.target.value)}
          placeholder="Bearer token for model management"
          autoComplete="off"
        />
      </div>

      <form className="model-admin__download" onSubmit={handleDownload}>
        <div className="form-row">
          <label htmlFor="download-url">Download URL</label>
          <input
            id="download-url"
            className="input"
            value={downloadUrl}
            onChange={(e) => setDownloadUrl(e.target.value)}
            placeholder="https://…/model.gguf"
            required
          />
        </div>
        <div className="form-row">
          <label htmlFor="download-filename">Filename override</label>
          <input
            id="download-filename"
            className="input"
            value={downloadFilename}
            onChange={(e) => setDownloadFilename(e.target.value)}
            placeholder="optional-model-name.gguf"
          />
        </div>
        <div className="form-actions">
          <button className="btn btn-submit" type="submit" disabled={!downloadUrl.trim()}>
            Download model
          </button>
        </div>
      </form>

      {downloadTask && (
        <div className="model-admin__task">
          <strong>{downloadTask.filename}</strong>
          <span className={`badge ${downloadTask.status === "done" ? "badge-active" : "badge-idle"}`}>
            {downloadTask.status}
          </span>
          <span className="dim">
            {formatBytes(downloadTask.bytes_downloaded)} / {formatBytes(downloadTask.total_bytes)}
          </span>
        </div>
      )}

      {message ? <p className="hint">{message}</p> : null}
      {error ? <p className="error-text">{error}</p> : null}

      <div className="model-admin__list">
        <div className="model-admin__list-head">
          <span>Installed models</span>
          <span className="dim">Loaded: {activeModelName || "none"}</span>
        </div>
        {models.length === 0 ? (
          <p className="dim">No models downloaded yet.</p>
        ) : (
          models.map((model) => (
            <div key={model.filename} className="model-admin__row">
              <div className="model-admin__meta">
                <strong>{model.filename}</strong>
                <span className="dim">
                  {model.quantization || "unknown quant"} · {formatBytes(model.size_bytes)}
                </span>
              </div>
              <button
                className="btn-small"
                type="button"
                disabled={loadingFilename === model.filename || activeModelName === model.filename}
                onClick={() => handleLoad(model.filename)}
              >
                {activeModelName === model.filename
                  ? "Active"
                  : loadingFilename === model.filename
                    ? "Loading…"
                    : "Load"}
              </button>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
