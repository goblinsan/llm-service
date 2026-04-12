import { useCallback, useRef, useState } from "react";
import type { ChatMessage, ChatParams, TurnMetadata } from "../types";

export type ChatStatus = "idle" | "queued" | "streaming" | "complete" | "error";

interface UseChatReturn {
  response: string;
  status: ChatStatus;
  metadata: TurnMetadata | null;
  errorMessage: string | null;
  submit: (messages: ChatMessage[], params: ChatParams) => void;
  cancel: () => void;
}

export function useChat(): UseChatReturn {
  const [response, setResponse] = useState("");
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [metadata, setMetadata] = useState<TurnMetadata | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
  }, []);

  const submit = useCallback(
    (messages: ChatMessage[], params: ChatParams) => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      setResponse("");
      setMetadata(null);
      setErrorMessage(null);
      setStatus("queued");

      const t0 = Date.now();

      const body = JSON.stringify({
        model: params.model || "local",
        messages,
        temperature: params.temperature,
        max_tokens: params.max_tokens,
        stream: params.stream,
        gateway_tools: params.gateway_tools,
      });

      (async () => {
        try {
          const res = await fetch("/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
            signal: ctrl.signal,
          });

          if (!res.ok) {
            const text = await res.text().catch(() => "");
            let msg = `HTTP ${res.status}`;
            try {
              const j = JSON.parse(text);
              msg = j.error ?? j.detail ?? msg;
            } catch {
              /* empty */
            }
            if (res.status === 503) {
              msg =
                "Model not ready — check status panel and retry when status is ok.";
            } else if (res.status === 429) {
              msg =
                "Inference slot busy — another request is in progress. Retry in a moment.";
            }
            setErrorMessage(msg);
            setStatus("error");
            return;
          }

          if (!params.stream) {
            const data = await res.json();
            const content: string =
              data.choices?.[0]?.message?.content ?? "(empty response)";
            const elapsed = Date.now() - t0;
            const usage = data.usage;
            const tps =
              usage?.completion_tokens && elapsed
                ? (usage.completion_tokens / elapsed) * 1000
                : undefined;
            setResponse(content);
            setMetadata({
              latency_ms: elapsed,
              tokens_per_second: tps,
              usage,
              model: data.model,
              timestamp: t0,
              streamed: false,
            });
            setStatus("complete");
            return;
          }

          // Streaming
          const reader = res.body?.getReader();
          if (!reader) {
            setErrorMessage("No response body for streaming");
            setStatus("error");
            return;
          }

          const decoder = new TextDecoder();
          let buffer = "";
          let fullContent = "";
          let completionTokens = 0;
          let promptTokens = 0;
          let modelName: string | undefined;

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() ?? "";

            for (const line of lines) {
              if (!line.startsWith("data:")) continue;
              const payload = line.slice(5).trim();
              if (payload === "[DONE]") break;
              try {
                const chunk = JSON.parse(payload);
                const delta: string =
                  chunk.choices?.[0]?.delta?.content ?? "";
                if (delta) {
                  fullContent += delta;
                  setResponse(fullContent);
                  // Transition from queued → streaming on first token
                  setStatus("streaming");
                }
                if (chunk.model) modelName = chunk.model;
                if (chunk.usage) {
                  completionTokens =
                    chunk.usage.completion_tokens ?? completionTokens;
                  promptTokens = chunk.usage.prompt_tokens ?? promptTokens;
                }
              } catch {
                /* ignore malformed chunks */
              }
            }
          }

          const elapsed = Date.now() - t0;
          const tps =
            completionTokens && elapsed
              ? (completionTokens / elapsed) * 1000
              : undefined;
          setMetadata({
            latency_ms: elapsed,
            tokens_per_second: tps,
            usage:
              completionTokens || promptTokens
                ? {
                    prompt_tokens: promptTokens,
                    completion_tokens: completionTokens,
                    total_tokens: promptTokens + completionTokens,
                  }
                : undefined,
            model: modelName,
            timestamp: t0,
            streamed: true,
          });
          setStatus("complete");
        } catch (err: unknown) {
          if (err instanceof DOMException && err.name === "AbortError") {
            setStatus("idle");
            return;
          }
          setErrorMessage(
            err instanceof Error ? err.message : "Unknown error"
          );
          setStatus("error");
        }
      })();
    },
    []
  );

  return { response, status, metadata, errorMessage, submit, cancel };
}
