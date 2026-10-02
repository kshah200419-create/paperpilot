import { SSEParser } from "./sse";
import type {
  AskDone,
  CategoryCount,
  Health,
  PaperDetail,
  SearchMode,
  SearchResponse,
  Source,
  Stats,
  Topic,
  TopicPoint,
  TrendsResponse,
} from "./types";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function get<T>(path: string, params?: Record<string, string | string[] | undefined>) {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v === undefined || v === "") continue;
    for (const item of Array.isArray(v) ? v : [v]) url.searchParams.append(k, item);
  }
  const res = await fetch(url.toString(), { headers: { Accept: "application/json" } });
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
  return (await res.json()) as T;
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return "Invalid request";
  } catch {
    /* ignore */
  }
  return `Request failed (${res.status})`;
}

export interface SearchParams {
  q: string;
  mode: SearchMode;
  categories: string[];
  dateFrom?: string;
  dateTo?: string;
  rerank?: boolean;
  limit?: number;
}

export const api = {
  health: () => get<Health>("/api/health"),
  stats: () => get<Stats>("/api/stats"),
  categories: () => get<CategoryCount[]>("/api/categories"),
  search: (p: SearchParams) =>
    get<SearchResponse>("/api/search", {
      q: p.q,
      mode: p.mode,
      category: p.categories,
      date_from: p.dateFrom ? `${p.dateFrom}T00:00:00Z` : undefined,
      date_to: p.dateTo ? `${p.dateTo}T23:59:59Z` : undefined,
      rerank: p.rerank ? "true" : undefined,
      limit: String(p.limit ?? 15),
    }),
  paper: (arxivId: string) => get<PaperDetail>(`/api/papers/${arxivId}`),
  topics: () => get<Topic[]>("/api/topics"),
  topicMap: () => get<TopicPoint[]>("/api/topics/map"),
  trends: (groupBy: "topic" | "category") =>
    get<TrendsResponse>("/api/trends", { group_by: groupBy, top: "8" }),
};

export interface AskHandlers {
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onDone: (done: AskDone) => void;
  onError: (message: string) => void;
}

/** POST /api/ask/stream and dispatch SSE events as they arrive. */
export async function streamAsk(
  question: string,
  handlers: AskHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/api/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, max_sources: 5 }),
    signal,
  });
  if (!res.ok || !res.body) {
    handlers.onError(await errorMessage(res));
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const msg of parser.push(decoder.decode(value, { stream: true }))) {
      const data = JSON.parse(msg.data) as Record<string, unknown>;
      if (msg.event === "sources") handlers.onSources(data.sources as Source[]);
      else if (msg.event === "token") handlers.onToken(String(data.text ?? ""));
      else if (msg.event === "done") handlers.onDone(data as unknown as AskDone);
      else if (msg.event === "error") handlers.onError(String(data.message ?? "error"));
    }
  }
}
