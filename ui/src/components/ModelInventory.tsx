import type { ModelEntry } from "../types";

interface ModelInventoryProps {
  models: ModelEntry[];
  loadedModel: string;
  onRefresh: () => void;
}

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function ModelInventory({
  models,
  loadedModel,
  onRefresh,
}: ModelInventoryProps) {
  return (
    <section className="panel model-inventory">
      <div className="panel-header">
        <h2>Model inventory</h2>
        <button className="btn-small" onClick={onRefresh} title="Refresh models">
          ↻
        </button>
      </div>
      {models.length === 0 ? (
        <p className="dim">
          No GGUF files found in the models directory. Use{" "}
          <code>POST /api/models/download</code> to download one.
        </p>
      ) : (
        <table className="model-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Size</th>
              <th>Quant</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.filename} className={m.loaded ? "row-active" : ""}>
                <td title={m.path}>{m.filename}</td>
                <td>{bytes(m.size_bytes)}</td>
                <td>{m.quantization ?? "—"}</td>
                <td>
                  {m.loaded ? (
                    <span className="badge badge-active">active</span>
                  ) : (
                    <span className="badge badge-idle">idle</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {models.length > 0 && (
        <p className="dim hint">
          To switch models use{" "}
          <code>POST /api/models/load {"{"}"filename":"…"{"}"}</code> (requires
          admin token). Reload this page after switching.
        </p>
      )}
      <p className="dim hint" style={{ marginTop: 4 }}>
        Active model path: <code>{loadedModel || "—"}</code>
      </p>
    </section>
  );
}
