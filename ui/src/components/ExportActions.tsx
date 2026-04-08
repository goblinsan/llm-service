import type { ChatSession, Turn } from "../types";

interface ExportActionsProps {
  session: ChatSession | null;
}

function buildText(session: ChatSession): string {
  const lines: string[] = [
    `Session: ${session.title}`,
    `Created: ${new Date(session.createdAt).toISOString()}`,
    `Model: ${session.params.model}`,
    `Temperature: ${session.params.temperature}  Max tokens: ${session.params.max_tokens}`,
    `System: ${session.systemPrompt || "(none)"}`,
    "---",
    "",
  ];
  session.turns.forEach((t, i) => {
    const user = t.messages.find((m) => m.role === "user");
    lines.push(`[Turn ${i + 1}]`);
    if (user) lines.push(`User: ${user.content}`);
    if (t.error) {
      lines.push(`Error: ${t.error}`);
    } else {
      lines.push(`Assistant: ${t.response}`);
      lines.push(
        `  Latency: ${t.metadata.latency_ms} ms` +
          (t.metadata.tokens_per_second
            ? `  Speed: ${t.metadata.tokens_per_second.toFixed(1)} t/s`
            : "")
      );
    }
    lines.push("");
  });
  return lines.join("\n");
}

function buildJson(session: ChatSession): string {
  return JSON.stringify(session, null, 2);
}

function buildMarkdown(session: ChatSession): string {
  const lines: string[] = [
    `# ${session.title}`,
    "",
    `**Created:** ${new Date(session.createdAt).toISOString()}  `,
    `**Model:** \`${session.params.model}\`  `,
    `**Temperature:** ${session.params.temperature}  **Max tokens:** ${session.params.max_tokens}`,
    "",
  ];
  if (session.systemPrompt) {
    lines.push(`> **System:** ${session.systemPrompt}`, "");
  }
  session.turns.forEach((t: Turn) => {
    const user = t.messages.find((m) => m.role === "user");
    if (user) {
      lines.push(`**User**`, "", "```", user.content, "```", "");
    }
    if (t.error) {
      lines.push(`**Error**`, "", t.error, "");
    } else {
      lines.push(`**Assistant**`, "", "```", t.response, "```", "");
      const meta: string[] = [
        `⏱ ${t.metadata.latency_ms} ms`,
        ...(t.metadata.tokens_per_second !== undefined
          ? [`⚡ ${t.metadata.tokens_per_second.toFixed(1)} t/s`]
          : []),
        ...(t.metadata.usage?.completion_tokens !== undefined
          ? [`🔤 ${t.metadata.usage.completion_tokens} tokens`]
          : []),
      ];
      lines.push(`*${meta.join("  ")}*`, "");
    }
    lines.push("---", "");
  });
  return lines.join("\n");
}

function download(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function slug(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 40);
}

export function ExportActions({ session }: ExportActionsProps) {
  if (!session || session.turns.length === 0) return null;

  const base = slug(session.title);

  return (
    <div className="export-actions">
      <span className="dim">Export:</span>
      <button
        className="btn-small"
        onClick={() => download(buildText(session), `${base}.txt`, "text/plain")}
        title="Export as plain text"
      >
        .txt
      </button>
      <button
        className="btn-small"
        onClick={() =>
          download(buildJson(session), `${base}.json`, "application/json")
        }
        title="Export as JSON"
      >
        .json
      </button>
      <button
        className="btn-small"
        onClick={() =>
          download(buildMarkdown(session), `${base}.md`, "text/markdown")
        }
        title="Export as Markdown"
      >
        .md
      </button>
    </div>
  );
}
