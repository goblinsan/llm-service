import type { HealthResponse, ModelsResponse } from "../types";

interface StatusPanelProps {
  health: HealthResponse | null;
  healthError: string | null;
  models: ModelsResponse | null;
}

const statusColour: Record<string, string> = {
  ok: "#22c55e",
  loading: "#f59e0b",
  "no-model": "#94a3b8",
  error: "#ef4444",
};

function Dot({ colour }: { colour: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 10,
        height: 10,
        borderRadius: "50%",
        background: colour,
        marginRight: 6,
        flexShrink: 0,
      }}
    />
  );
}

export function StatusPanel({ health, healthError, models }: StatusPanelProps) {
  const status = health?.status ?? "loading";
  const colour = statusColour[status] ?? "#94a3b8";
  const loadedModel = models?.loaded_model
    ? models.loaded_model.split("/").pop()
    : "—";
  const ctxSize = models?.ctx_size ?? health?.ctx_size ?? "—";
  const gpuLayers = models?.n_gpu_layers ?? health?.n_gpu_layers ?? "—";
  const llama = models?.llama ?? health?.llama ?? null;
  const offloadedLayers =
    typeof llama?.offloaded_layers === "number" && typeof llama?.total_layers === "number"
      ? `${llama.offloaded_layers}/${llama.total_layers}`
      : "—";
  const flashAttn =
    typeof llama?.flash_attn === "boolean" ? (llama.flash_attn ? "enabled" : "disabled") : "—";
  const loadGpu =
    llama?.cuda_device_name && typeof llama?.cuda_free_mib_at_load === "number"
      ? `${llama.cuda_device_name} (${llama.cuda_free_mib_at_load.toLocaleString()} MiB free at load)`
      : llama?.cuda_device_name || "—";
  const lastLlamaLine =
    Array.isArray(llama?.recent_log_tail) && llama.recent_log_tail.length > 0
      ? llama.recent_log_tail[llama.recent_log_tail.length - 1]
      : null;

  return (
    <section className="panel status-panel">
      <h2>Service status</h2>
      <table className="info-table">
        <tbody>
          <tr>
            <th>Wrapper</th>
            <td>
              <Dot colour={colour} />
              {healthError ? (
                <span className="error-text">{healthError}</span>
              ) : (
                <span>{status}</span>
              )}
              {health?.detail ? (
                <span className="dim"> — {health.detail}</span>
              ) : null}
            </td>
          </tr>
          <tr>
            <th>Active model</th>
            <td title={models?.loaded_model}>{loadedModel}</td>
          </tr>
          <tr>
            <th>Models found</th>
            <td>{models?.models.length ?? "—"}</td>
          </tr>
          <tr>
            <th>Context</th>
            <td>{typeof ctxSize === "number" ? `${ctxSize.toLocaleString()} tokens` : ctxSize}</td>
          </tr>
          <tr>
            <th>GPU layers</th>
            <td>{typeof gpuLayers === "number" ? gpuLayers : gpuLayers}</td>
          </tr>
          <tr>
            <th>GPU offload</th>
            <td>{offloadedLayers}</td>
          </tr>
          <tr>
            <th>Flash attention</th>
            <td>{flashAttn}</td>
          </tr>
          <tr>
            <th>Load GPU</th>
            <td>{loadGpu}</td>
          </tr>
          {lastLlamaLine ? (
            <tr>
              <th>Last llama log</th>
              <td title={lastLlamaLine}>{lastLlamaLine}</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </section>
  );
}
