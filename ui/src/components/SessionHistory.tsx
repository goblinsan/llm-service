import type { ChatSession } from "../types";

interface SessionHistoryProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelect: (session: ChatSession) => void;
  onDelete: (id: string) => void;
  onClearAll: () => void;
  onNewSession: () => void;
}

export function SessionHistory({
  sessions,
  activeSessionId,
  onSelect,
  onDelete,
  onClearAll,
  onNewSession,
}: SessionHistoryProps) {
  return (
    <aside className="session-history">
      <div className="panel-header">
        <h2>History</h2>
        <button className="btn-small" onClick={onNewSession} title="New session">
          + New
        </button>
      </div>

      {sessions.length === 0 ? (
        <p className="dim">No sessions yet.</p>
      ) : (
        <>
          <ul className="session-list">
            {sessions.map((s) => (
              <li
                key={s.id}
                className={`session-item ${s.id === activeSessionId ? "session-active" : ""}`}
              >
                <button
                  className="session-title"
                  onClick={() => onSelect(s)}
                  title={s.title}
                >
                  <span className="session-name">{s.title}</span>
                  <span className="session-meta dim">
                    {new Date(s.updatedAt).toLocaleDateString()} ·{" "}
                    {s.turns.length} turn{s.turns.length !== 1 ? "s" : ""}
                  </span>
                </button>
                <button
                  className="btn-icon"
                  onClick={() => onDelete(s.id)}
                  title="Delete session"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
          <button className="btn-small btn-danger" onClick={onClearAll}>
            Clear all
          </button>
        </>
      )}
    </aside>
  );
}
