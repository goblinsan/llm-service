import type { Turn } from "../types";
import type { ChatStatus } from "../hooks/useChat";

interface ChatPanelProps {
  turns: Turn[];
  streamingResponse: string;
  chatStatus: ChatStatus;
  errorMessage: string | null;
  onClearHistory: () => void;
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms} ms`;
}

function TurnView({ turn }: { turn: Turn }) {
  const userMsg = turn.messages.find((m) => m.role === "user");
  const sysMsg = turn.messages.find((m) => m.role === "system");
  const { metadata } = turn;

  return (
    <div className="turn">
      {sysMsg && (
        <div className="bubble bubble-system">
          <span className="role-label">system</span>
          <pre className="message-content">{sysMsg.content}</pre>
        </div>
      )}
      {userMsg && (
        <div className="bubble bubble-user">
          <span className="role-label">user</span>
          <pre className="message-content">{userMsg.content}</pre>
        </div>
      )}
      {turn.error ? (
        <div className="bubble bubble-error">
          <span className="role-label">error</span>
          <pre className="message-content">{turn.error}</pre>
        </div>
      ) : (
        <div className="bubble bubble-assistant">
          <span className="role-label">assistant</span>
          <pre className="message-content">{turn.response}</pre>
          <div className="turn-meta">
            <span title="Wall-clock latency">⏱ {formatMs(metadata.latency_ms)}</span>
            {metadata.tokens_per_second !== undefined && (
              <span title="Tokens per second">
                ⚡ {metadata.tokens_per_second.toFixed(1)} t/s
              </span>
            )}
            {metadata.usage?.completion_tokens !== undefined && (
              <span title="Completion tokens">
                🔤 {metadata.usage.completion_tokens} tok
              </span>
            )}
            {metadata.model && (
              <span title="Model" className="dim">
                {metadata.model}
              </span>
            )}
            <span className="dim" title="Sent at">
              {new Date(metadata.timestamp).toLocaleTimeString()}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export function ChatPanel({
  turns,
  streamingResponse,
  chatStatus,
  errorMessage,
  onClearHistory,
}: ChatPanelProps) {
  const empty = turns.length === 0 && chatStatus === "idle";

  return (
    <section className="panel chat-panel">
      <div className="panel-header">
        <h2>Transcript</h2>
        {turns.length > 0 && (
          <button
            className="btn-small"
            onClick={onClearHistory}
            title="Clear current transcript"
          >
            ✕ Clear
          </button>
        )}
      </div>

      {empty ? (
        <p className="dim placeholder">
          Submit a prompt to start a conversation.
        </p>
      ) : (
        <div className="turns-list">
          {turns.map((t) => (
            <TurnView key={t.id} turn={t} />
          ))}

          {/* Live streaming bubble */}
          {(chatStatus === "streaming" || chatStatus === "complete") &&
            streamingResponse && (
              <div className="turn">
                <div className="bubble bubble-assistant">
                  <span className="role-label">
                    assistant
                    {chatStatus === "streaming" && (
                      <span className="streaming-badge"> streaming…</span>
                    )}
                  </span>
                  <pre className="message-content">{streamingResponse}</pre>
                </div>
              </div>
            )}

          {chatStatus === "error" && errorMessage && (
            <div className="turn">
              <div className="bubble bubble-error">
                <span className="role-label">error</span>
                <pre className="message-content">{errorMessage}</pre>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
