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
        </tbody>
      </table>
    </section>
  );
}
