import { useCallback, useRef, useState } from "react";
import { streamAsk } from "../api/client";
import type { Source } from "../api/types";

export interface Turn {
  id: number;
  question: string;
  answer: string;
  sources: Source[];
  cited: number[];
  provider?: string;
  tookMs?: number;
  status: "retrieving" | "streaming" | "done" | "error";
  error?: string;
}

/** Chat state machine over the SSE /api/ask/stream endpoint. */
export function useAsk() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const nextId = useRef(1);

  const patch = (id: number, fn: (t: Turn) => Partial<Turn>) =>
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...fn(t) } : t)));

  const ask = useCallback(async (question: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const id = nextId.current++;
    setTurns((ts) => [
      ...ts,
      { id, question, answer: "", sources: [], cited: [], status: "retrieving" },
    ]);
    try {
      await streamAsk(
        question,
        {
          onSources: (sources) => patch(id, () => ({ sources, status: "streaming" })),
          onToken: (text) => patch(id, (t) => ({ answer: t.answer + text, status: "streaming" })),
          onDone: (d) =>
            patch(id, () => ({
              status: "done",
              cited: d.cited,
              provider: d.provider,
              tookMs: d.took_ms,
            })),
          onError: (message) => patch(id, () => ({ status: "error", error: message })),
        },
        controller.signal,
      );
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      patch(id, () => ({ status: "error", error: (err as Error).message || "Network error" }));
    }
  }, []);

  const busy = turns.some((t) => t.status === "retrieving" || t.status === "streaming");
  const reset = () => {
    abortRef.current?.abort();
    setTurns([]);
  };
  return { turns, ask, busy, reset };
}
