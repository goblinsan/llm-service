export type HealthStatus = "ok" | "loading" | "no-model" | "error";

export interface HealthResponse {
  status: HealthStatus;
  detail?: string;
}

export interface ModelEntry {
  filename: string;
  path: string;
  size_bytes: number;
  quantization: string | null;
  loaded: boolean;
}

export interface ModelsResponse {
  models: ModelEntry[];
  loaded_model: string;
  status: string;
}

export interface DownloadTask {
  task_id: string;
  status: "pending" | "downloading" | "done" | "error";
  filename: string;
  bytes_downloaded: number;
  total_bytes: number | null;
  error: string | null;
}

export interface LoadResponse {
  loaded_model: string;
  status: string;
  error?: string | null;
}

export type MessageRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface TokenUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface TurnMetadata {
  latency_ms: number;
  tokens_per_second?: number;
  usage?: TokenUsage;
  model?: string;
  timestamp: number;
  streamed?: boolean;
}

export interface Turn {
  id: string;
  messages: ChatMessage[];
  response: string;
  metadata: TurnMetadata;
  error?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  systemPrompt: string;
  turns: Turn[];
  params: ChatParams;
}

export interface ChatParams {
  model: string;
  temperature: number;
  max_tokens: number;
  stream: boolean;
}

export interface PromptPreset {
  id: string;
  name: string;
  systemPrompt: string;
  userPrompt: string;
  params: Partial<ChatParams>;
}

export const DEFAULT_PRESETS: PromptPreset[] = [
  {
    id: "general",
    name: "General assistant",
    systemPrompt: "You are a helpful assistant.",
    userPrompt: "",
    params: { temperature: 0.7, max_tokens: 512 },
  },
  {
    id: "code",
    name: "Code review",
    systemPrompt:
      "You are an expert code reviewer. Provide concise, actionable feedback.",
    userPrompt: "",
    params: { temperature: 0.2, max_tokens: 1024 },
  },
  {
    id: "summarise",
    name: "Summarise",
    systemPrompt:
      "Summarise the following text in 3–5 bullet points. Be concise.",
    userPrompt: "",
    params: { temperature: 0.3, max_tokens: 256 },
  },
  {
    id: "classify",
    name: "Classify / route",
    systemPrompt:
      "Classify the user input into one of the provided categories. Reply with only the category name.",
    userPrompt: "",
    params: { temperature: 0.0, max_tokens: 32 },
  },
];
