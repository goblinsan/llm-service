import { useCallback, useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { StatusPanel } from "./components/StatusPanel";
import { ModelInventory } from "./components/ModelInventory";
import { ModelAdminPanel } from "./components/ModelAdminPanel";
import { ChatPanel } from "./components/ChatPanel";
import { PromptForm } from "./components/PromptForm";
import { SessionHistory } from "./components/SessionHistory";
import { ExportActions } from "./components/ExportActions";
import { useHealth } from "./hooks/useHealth";
import { useModels } from "./hooks/useModels";
import { useChat } from "./hooks/useChat";
import { useSessionHistory } from "./hooks/useSessionHistory";
import type { ChatParams, ChatSession, TurnMetadata, Turn } from "./types";

function makeTitle(userMessage: string): string {
  const clean = userMessage.trim().replace(/\s+/g, " ");
  return clean.length > 50 ? clean.slice(0, 47) + "\u2026" : clean;
}

// ---------------------------------------------------------------------------
// Hook: finalise a turn when chat status becomes "complete" or "error"
// ---------------------------------------------------------------------------

function useFinaliseTurn(
  status: string,
  response: string,
  metadata: TurnMetadata | null,
  errorMessage: string | null,
  pendingRef: React.MutableRefObject<null | {
    session: ChatSession;
    turnId: string;
    pendingTurn: Turn;
  }>,
  onDone: (session: ChatSession) => void
) {
  const doneRef = useRef<string | null>(null);

  useEffect(() => {
    if (status !== "complete" && status !== "error") return;
    if (!pendingRef.current) return;
    const { session, turnId, pendingTurn } = pendingRef.current;
    if (doneRef.current === turnId) return;
    doneRef.current = turnId;

    const finalisedTurn: Turn = {
      ...pendingTurn,
      response: status === "error" ? "" : response,
      metadata: metadata ?? pendingTurn.metadata,
      error: status === "error" ? (errorMessage ?? "Unknown error") : undefined,
    };
    const finalisedSession: ChatSession = {
      ...session,
      updatedAt: Date.now(),
      turns: [...session.turns, finalisedTurn],
    };
    pendingRef.current = null;
    onDone(finalisedSession);
  }, [status]); // eslint-disable-line react-hooks/exhaustive-deps
}

// ---------------------------------------------------------------------------
// Root component
// ---------------------------------------------------------------------------

export default function App() {
  const { health, error: healthError, refresh: refreshHealth } = useHealth(5000);
  const { models, refresh: refreshModels } = useModels(15000);
  const { response, status, metadata, errorMessage, submit, cancel } = useChat();
  const { sessions, upsertSession, deleteSession, clearAll } = useSessionHistory();

  const [activeSession, setActiveSession] = useState<ChatSession | null>(null);

  // Pending turn ref — carries the in-progress submission context
  const pendingRef = useRef<null | {
    session: ChatSession;
    turnId: string;
    pendingTurn: Turn;
  }>(null);

  // Finalise the turn once the chat hook settles
  useFinaliseTurn(status, response, metadata, errorMessage, pendingRef, (s) => {
    setActiveSession({ ...s });
    upsertSession({ ...s });
  });

  // -----------------------------------------------------------------------
  // Session helpers
  // -----------------------------------------------------------------------

  function blankSession(): ChatSession {
    return {
      id: uuidv4(),
      title: "New session",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      systemPrompt: "",
      turns: [],
      params: { model: "local", temperature: 0.7, max_tokens: 512, stream: true },
    };
  }

  // -----------------------------------------------------------------------
  // Prompt submission
  // -----------------------------------------------------------------------

  const handleSubmit = useCallback(
    (userMessage: string, systemPrompt: string, params: ChatParams) => {
      const session: ChatSession = activeSession
        ? { ...activeSession, params, systemPrompt }
        : blankSession();

      if (session.title === "New session") {
        session.title = makeTitle(userMessage);
      }

      // Build full message list for multi-turn context
      const messages = [
        ...(systemPrompt.trim()
          ? [{ role: "system" as const, content: systemPrompt }]
          : []),
        ...session.turns.flatMap((t) =>
          t.error
            ? []
            : [
                ...t.messages.filter((m) => m.role === "user"),
                { role: "assistant" as const, content: t.response },
              ]
        ),
        { role: "user" as const, content: userMessage },
      ];

      submit(messages, params);

      const turnId = uuidv4();
      pendingRef.current = {
        session,
        turnId,
        pendingTurn: {
          id: turnId,
          messages: [
            ...(systemPrompt.trim()
              ? [{ role: "system" as const, content: systemPrompt }]
              : []),
            { role: "user" as const, content: userMessage },
          ],
          response: "",
          metadata: { latency_ms: 0, timestamp: Date.now() },
        },
      };

      setActiveSession({ ...session });
    },
    [activeSession, submit]
  );

  // -----------------------------------------------------------------------
  // Rerun a previous turn with the same user message and current params
  // -----------------------------------------------------------------------

  const handleRerun = useCallback(
    (turn: Turn) => {
      if (!activeSession) return;
      const userMsg = turn.messages.find((m) => m.role === "user");
      const sysMsg = turn.messages.find((m) => m.role === "system");
      if (!userMsg) return;
      handleSubmit(
        userMsg.content,
        sysMsg?.content ?? activeSession.systemPrompt,
        activeSession.params
      );
    },
    [activeSession, handleSubmit]
  );

  // -----------------------------------------------------------------------
  // UI callbacks
  // -----------------------------------------------------------------------

  function handleClearHistory() {
    if (!activeSession) return;
    const cleared = { ...activeSession, turns: [], updatedAt: Date.now() };
    setActiveSession(cleared);
    upsertSession(cleared);
  }

  function handleRefresh() {
    refreshHealth();
    refreshModels();
  }

  const activeModelName = models?.loaded_model
    ? models.loaded_model.split("/").pop() || models.loaded_model
    : "";

  return (
    <div className="app-layout">
      <SessionHistory
        sessions={sessions}
        activeSessionId={activeSession?.id ?? null}
        onSelect={(s) => setActiveSession({ ...s })}
        onDelete={deleteSession}
        onClearAll={clearAll}
        onNewSession={() => setActiveSession(blankSession())}
      />

      <main className="main-content">
        <header className="app-header">
          <div className="header-title">
            <h1>&#129504; LLM Playground</h1>
            <span className="scope-badge" title="This UI is for local operator testing only. Public auth and multi-user isolation belong in gateway-tools-platform.">
              local operator tool
            </span>
          </div>
          <button className="btn-small" onClick={handleRefresh}>
            &#8635; Refresh status
          </button>
        </header>

        <div className="info-row">
          <StatusPanel
            health={health}
            healthError={healthError}
            models={models}
          />
          <ModelInventory
            models={models?.models ?? []}
            loadedModel={models?.loaded_model ?? ""}
            onRefresh={refreshModels}
          />
        </div>

        {health?.status === "no-model" && (
          <section className="panel setup-callout">
            <div className="setup-callout__copy">
              <h2>Load your first model</h2>
              <p>
                The wrapper is healthy, but no GGUF is loaded yet. Use the model setup panel below to
                download a starter model and make it active before sending chat prompts.
              </p>
            </div>
            <ol className="setup-callout__steps">
              <li>Paste the admin token for this node.</li>
              <li>Download a starter GGUF like Mistral 7B Q4_K_M.</li>
              <li>Click <strong>Load</strong> once the download finishes.</li>
            </ol>
          </section>
        )}

        <ModelAdminPanel
          models={models?.models ?? []}
          loadedModel={models?.loaded_model ?? ""}
          currentCtxSize={models?.ctx_size ?? health?.ctx_size ?? 4096}
          currentGpuLayers={models?.n_gpu_layers ?? health?.n_gpu_layers ?? -1}
          onChanged={handleRefresh}
        />

        <ChatPanel
          turns={activeSession?.turns ?? []}
          streamingResponse={response}
          chatStatus={status}
          errorMessage={errorMessage}
          onClearHistory={handleClearHistory}
          onRerun={handleRerun}
        />

        <ExportActions session={activeSession} />

        <PromptForm
          onSubmit={handleSubmit}
          onCancel={cancel}
          status={status}
          healthStatus={health?.status ?? "loading"}
          activeModelName={activeModelName}
        />
      </main>
    </div>
  );
}
