import { useEffect, useMemo, useState } from "react";
import type { DownloadTask, LoadResponse, ModelEntry } from "../types";

interface ModelAdminPanelProps {
  models: ModelEntry[];
  loadedModel: string;
  onChanged: () => void;
}

interface StarterModel {
  id: string;
  name: string;
  publisher: string;
  filename: string;
  quantization: string;
  approxSize: string;
  approxVram: string;
  summary: string;
  url: string;
  recommended?: boolean;
}

const STARTER_MODELS: StarterModel[] = [
  {
    id: "mistral-7b-v03-q4km",
    name: "Mistral 7B v0.3",
    publisher: "TheBloke",
    filename: "mistral-7b-v0.3.Q4_K_M.gguf",
    quantization: "Q4_K_M",
    approxSize: "~4.5 GB",
    approxVram: "~5.2 GB @ 4k ctx",
    summary: "Recommended default for a shared 8 GB GPU. Strong general-purpose quality and broad community support.",
    url: "https://huggingface.co/TheBloke/Mistral-7B-v0.3-GGUF/resolve/main/mistral-7b-v0.3.Q4_K_M.gguf",
    recommended: true,
  },
  {
    id: "llama-31-8b-q4km",
    name: "Llama 3.1 8B Instruct",
    publisher: "bartowski",
    filename: "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    quantization: "Q4_K_M",
    approxSize: "~5.0 GB",
    approxVram: "~5.7 GB @ 4k ctx",
    summary: "A good stronger general alternative if you can spare a bit more VRAM.",
    url: "https://huggingface.co/bartowski/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf",
  },
  {
    id: "qwen25-7b-q4km",
    name: "Qwen 2.5 7B Instruct",
    publisher: "bartowski",
    filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    quantization: "Q4_K_M",
    approxSize: "~4.5 GB",
    approxVram: "~5.2 GB @ 4k ctx",
    summary: "Best starter choice when you care about multilingual prompts and output.",
    url: "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
  },
];

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

  async function startDownload(url: string, filename?: string) {
    setMessage(null);
    setError(null);
    setDownloadTask(null);

    const trimmedUrl = url.trim();
    const trimmedFilename = filename?.trim();
    if (!trimmedUrl) {
      setError("Provide a GGUF download URL");
      return;
    }

    try {
      const res = await fetch("/api/models/download", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...buildAuthHeader(adminToken),
        },
        body: JSON.stringify({
          url: trimmedUrl,
          filename: trimmedFilename || undefined,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || data.error || `HTTP ${res.status}`);
      }
      setDownloadTask(data as DownloadTask);
      setMessage(`Started download for ${(data as DownloadTask).filename}`);
      setDownloadUrl("");
      setDownloadFilename("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start download");
    }
  }

  async function handleDownload(e: React.FormEvent) {
    e.preventDefault();
    await startDownload(downloadUrl, downloadFilename);
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
        <p className="hint">
          The token stays in this browser session only. It is required for model downloads and model switching.
        </p>
      </div>

      <div className="model-admin__starter">
        <div className="model-admin__list-head">
          <span>Starter models</span>
          <span className="dim">Quick-install a known GGUF</span>
        </div>
        <div className="model-admin__catalog">
          {STARTER_MODELS.map((model) => (
            <article
              key={model.id}
              className={`model-admin__catalog-card ${model.recommended ? "model-admin__catalog-card--recommended" : ""}`}
            >
              <div className="model-admin__catalog-head">
                <div>
                  <strong>{model.name}</strong>
                  <div className="dim">{model.publisher}</div>
                </div>
                {model.recommended ? <span className="badge badge-active">recommended</span> : null}
              </div>
              <p className="model-admin__catalog-copy">{model.summary}</p>
              <div className="model-admin__catalog-meta">
                <span>{model.quantization}</span>
                <span>{model.approxSize}</span>
                <span>{model.approxVram}</span>
              </div>
              <div className="model-admin__catalog-actions">
                <button
                  className="btn-small"
                  type="button"
                  onClick={() => void startDownload(model.url, model.filename)}
                >
                  Download
                </button>
                <button
                  className="btn-small"
                  type="button"
                  onClick={() => {
                    setDownloadUrl(model.url);
                    setDownloadFilename(model.filename);
                    setMessage(`Prefilled ${model.name}. Review the URL below or download immediately.`);
                    setError(null);
                  }}
                >
                  Fill form
                </button>
              </div>
            </article>
          ))}
        </div>
      </div>

      <form className="model-admin__download" onSubmit={handleDownload}>
        <div className="model-admin__list-head">
          <span>Custom download</span>
          <span className="dim">Paste any direct GGUF URL</span>
        </div>
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
