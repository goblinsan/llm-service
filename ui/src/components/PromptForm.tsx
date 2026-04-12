import { useState } from "react";
import type { ChatParams, PromptPreset } from "../types";
import { DEFAULT_PRESETS } from "../types";
import type { ChatStatus } from "../hooks/useChat";
import type { HealthStatus } from "../types";

interface PromptFormProps {
  onSubmit: (userMessage: string, systemPrompt: string, params: ChatParams) => void;
  onCancel: () => void;
  status: ChatStatus;
  healthStatus: HealthStatus;
  activeModelName: string;
}

export function PromptForm({
  onSubmit,
  onCancel,
  status,
  healthStatus,
  activeModelName,
}: PromptFormProps) {
  const [systemPrompt, setSystemPrompt] = useState("Provide concise answers without explanations");
  const [userMessage, setUserMessage] = useState("");
  const [model, setModel] = useState("local");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [stream, setStream] = useState(true);
  const [toolsEnabled, setToolsEnabled] = useState(false);
  const [timeToolEnabled, setTimeToolEnabled] = useState(true);
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);
  const [selectedPreset, setSelectedPreset] = useState<string>("");

  const busy = status === "streaming" || status === "queued";
  const blockedByModel = healthStatus === "no-model" || healthStatus === "loading";
  const busyLabel =
    status === "queued"
      ? stream
        ? "Waiting for first token…"
        : "Waiting for response…"
      : stream
        ? "Streaming…"
        : "Receiving…";

  function applyPreset(preset: PromptPreset) {
    setSelectedPreset(preset.id);
    if (preset.systemPrompt) setSystemPrompt(preset.systemPrompt);
    if (preset.userPrompt) setUserMessage(preset.userPrompt);
    if (preset.params.temperature !== undefined)
      setTemperature(preset.params.temperature);
    if (preset.params.max_tokens !== undefined)
      setMaxTokens(preset.params.max_tokens);
    if (preset.params.stream !== undefined) setStream(preset.params.stream);
    if (preset.params.model !== undefined) setModel(preset.params.model);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!userMessage.trim()) return;
    const clientTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone || undefined;
    onSubmit(userMessage, systemPrompt, {
      model,
      temperature,
      max_tokens: maxTokens,
      stream,
      gateway_tools: {
        enabled: toolsEnabled,
        time: toolsEnabled && timeToolEnabled,
        web_search: toolsEnabled && webSearchEnabled,
        client_timezone: toolsEnabled ? clientTimezone : undefined,
      },
    });
  }

  return (
    <form className="panel prompt-form" onSubmit={handleSubmit}>
      <h2>Prompt</h2>

      {/* Presets */}
      <div className="form-row presets-row">
        <label>Preset</label>
        <div className="preset-chips">
          {DEFAULT_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`chip ${selectedPreset === p.id ? "chip-active" : ""}`}
              onClick={() => applyPreset(p)}
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* System prompt */}
      <div className="form-row">
        <label htmlFor="system-prompt">System prompt</label>
        <textarea
          id="system-prompt"
          className="textarea"
          rows={3}
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="Optional system prompt…"
        />
      </div>

      {/* User message */}
      <div className="form-row">
        <label htmlFor="user-message">Message</label>
        <textarea
          id="user-message"
          className="textarea"
          rows={4}
          value={userMessage}
          onChange={(e) => setUserMessage(e.target.value)}
          placeholder="Type your message…"
          required
        />
      </div>

      {/* Parameters row */}
      <div className="params-grid">
        <div className="param-item">
          <label htmlFor="model-id">Model</label>
          <input
            id="model-id"
            className="input"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="local"
          />
        </div>

        <div className="param-item">
          <label htmlFor="temperature">
            Temperature <span className="dim">({temperature})</span>
          </label>
          <input
            id="temperature"
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
          />
        </div>

        <div className="param-item">
          <label htmlFor="max-tokens">Max tokens</label>
          <input
            id="max-tokens"
            type="number"
            className="input input-number"
            min={1}
            max={4096}
            value={maxTokens}
            onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
          />
        </div>

        <div className="param-item param-checkbox">
          <label htmlFor="stream-toggle">
            <input
              id="stream-toggle"
              type="checkbox"
              checked={stream}
              onChange={(e) => setStream(e.target.checked)}
            />
            Streaming
          </label>
        </div>
      </div>

      <div className="params-grid">
        <div className="param-item param-checkbox">
          <label htmlFor="tools-toggle">
            <input
              id="tools-toggle"
              type="checkbox"
              checked={toolsEnabled}
              onChange={(e) => setToolsEnabled(e.target.checked)}
            />
            Built-in tools
          </label>
        </div>

        <div className="param-item param-checkbox">
          <label htmlFor="time-tool-toggle">
            <input
              id="time-tool-toggle"
              type="checkbox"
              checked={timeToolEnabled}
              disabled={!toolsEnabled}
              onChange={(e) => setTimeToolEnabled(e.target.checked)}
            />
            Date / time
          </label>
        </div>

        <div className="param-item param-checkbox">
          <label htmlFor="search-tool-toggle">
            <input
              id="search-tool-toggle"
              type="checkbox"
              checked={webSearchEnabled}
              disabled={!toolsEnabled}
              onChange={(e) => setWebSearchEnabled(e.target.checked)}
            />
            Web search
          </label>
        </div>
      </div>

      {/* Actions */}
      <div className="form-actions">
        {busy ? (
          <button type="button" className="btn btn-cancel" onClick={onCancel}>
            ✕ Cancel
          </button>
        ) : (
          <button
            type="submit"
            className="btn btn-submit"
            disabled={!userMessage.trim() || blockedByModel}
          >
            {healthStatus === "no-model"
              ? "Load a model first"
              : healthStatus === "loading"
                ? "Model loading…"
                : "Send ↵"}
          </button>
        )}
        {busy && <span className="streaming-indicator">{busyLabel}</span>}
      </div>

      {blockedByModel && (
        <p className="hint">
          {healthStatus === "no-model"
            ? "Use the model setup panel above to download and load a GGUF model before chatting."
            : `The service is loading ${activeModelName || "the selected model"}. Chat will unlock once /health returns ok.`}
        </p>
      )}
    </form>
  );
}
