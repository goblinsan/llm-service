import { useEffect, useMemo, useState } from "react";
import type { DownloadTask, LoadResponse, ModelEntry } from "../types";

interface ModelAdminPanelProps {
  models: ModelEntry[];
  loadedModel: string;
  currentCtxSize: number;
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
  weightVramGb: number;
  kvCache4kGb: number;
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
    weightVramGb: 4.5,
    kvCache4kGb: 0.7,
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
    weightVramGb: 5.0,
    kvCache4kGb: 0.7,
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
    weightVramGb: 4.5,
    kvCache4kGb: 0.7,
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

function formatCtx(value: number): string {
  return `${value.toLocaleString()} ctx`;
}

function estimateInstalledModelVram(model: ModelEntry, ctxSize: number): number {
  const weightGb = model.size_bytes / 1024 ** 3;
  const kvAt4kGb = Math.max(0.4, Math.min(3.5, weightGb * 0.14));
  return weightGb + kvAt4kGb * (ctxSize / 4096);
}

function estimateStarterVram(model: StarterModel, ctxSize: number): number {
  return model.weightVramGb + model.kvCache4kGb * (ctxSize / 4096);
}

function formatVramEstimate(valueGb: number): string {
  return `~${valueGb.toFixed(1)} GB VRAM`;
}

export function ModelAdminPanel({
  models,
  loadedModel,
  currentCtxSize,
  onChanged,
}: ModelAdminPanelProps) {
  const [adminToken, setAdminToken] = useState("");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [downloadFilename, setDownloadFilename] = useState("");
  const [ctxSize, setCtxSize] = useState(currentCtxSize);
  const [downloadTask, setDownloadTask] = useState<DownloadTask | null>(null);
  const [loadingFilename, setLoadingFilename] = useState<string | null>(null);
  const [unloading, setUnloading] = useState(false);
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
    setCtxSize(currentCtxSize);
  }, [currentCtxSize]);

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
        body: JSON.stringify({ filename, ctx_size: ctxSize }),
      });
      const data = (await res.json().catch(() => ({}))) as LoadResponse & {
        detail?: string;
      };
      if (!res.ok) {
        throw new Error(data.detail || data.error || `HTTP ${res.status}`);
      }
      setMessage(`Switched active model to ${filename} at ${ctxSize.toLocaleString()} tokens`);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load model");
    } finally {
      setLoadingFilename(null);
    }
  }

  async function handleUnload() {
    setUnloading(true);
    setMessage(null);
    setError(null);
    try {
      const res = await fetch("/api/models/unload", {
        method: "POST",
        headers: {
          ...buildAuthHeader(adminToken),
        },
      });
      const data = (await res.json().catch(() => ({}))) as LoadResponse & {
        detail?: string;
      };
      if (!res.ok) {
        throw new Error(data.detail || data.error || `HTTP ${res.status}`);
      }
      setMessage("Unloaded active model. The next load will bring it back into VRAM.");
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to unload model");
    } finally {
      setUnloading(false);
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

      <div className="model-admin__runtime">
        <div className="form-row">
          <label htmlFor="ctx-size">Context size</label>
          <input
            id="ctx-size"
            className="input input-number"
            type="number"
            min={256}
            max={262144}
            step={256}
            value={ctxSize}
            onChange={(e) => {
              const parsed = Number.parseInt(e.target.value || String(currentCtxSize), 10);
              setCtxSize(Number.isFinite(parsed) ? Math.max(256, parsed) : currentCtxSize);
            }}
          />
        </div>
        <p className="hint">
          This applies on the next <strong>Load</strong>. VRAM estimates below are rough and scale with the selected context size.
        </p>
      </div>

      <div className="model-admin__starter">
        <div className="model-admin__list-head">
          <span>Starter models</span>
          <span className="dim">Quick-install a known GGUF at {formatCtx(ctxSize)}</span>
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
                <span>{formatVramEstimate(estimateStarterVram(model, ctxSize))}</span>
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
          <span className="dim">Loaded: {activeModelName || "none"} · {formatCtx(ctxSize)}</span>
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
                <span className="dim">
                  {formatVramEstimate(estimateInstalledModelVram(model, ctxSize))}
                </span>
              </div>
              <button
                className="btn-small"
                type="button"
                disabled={loadingFilename === model.filename || unloading}
                onClick={() => handleLoad(model.filename)}
              >
                {activeModelName === model.filename
                  ? loadingFilename === model.filename
                    ? "Reloading…"
                    : "Reload"
                  : loadingFilename === model.filename
                    ? "Loading…"
                    : "Load"}
              </button>
              {activeModelName === model.filename ? (
                <button
                  className="btn-small"
                  type="button"
                  disabled={loadingFilename === model.filename || unloading}
                  onClick={() => void handleUnload()}
                >
                  {unloading ? "Unloading…" : "Unload"}
                </button>
              ) : null}
            </div>
          ))
        )}
      </div>
    </section>
  );
}
