import { Bot, CornerDownLeft, RotateCcw, User } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { AnswerText } from "../components/AnswerText";
import { CitationCard } from "../components/CitationCard";
import { Spinner } from "../components/States";
import { useAsk, type Turn } from "../hooks/useAsk";

const SUGGESTIONS = [
  "How can mixture-of-experts inference be made cheaper?",
  "What attacks bypass LLM safety alignment?",
  "How do robots fuse lidar and cameras for localization?",
  "Are time series foundation models well calibrated?",
];

export function AskPage() {
  const { turns, ask, busy, reset } = useAsk();
  const [draft, setDraft] = useState("");
  const [params, setParams] = useSearchParams();
  const bottomRef = useRef<HTMLDivElement>(null);

  // Allow deep links like /ask?q=... (used by the paper page "Ask about this paper").
  useEffect(() => {
    const q = params.get("q");
    if (q) {
      setParams({}, { replace: true });
      void ask(q);
    }
  }, [params, setParams, ask]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [turns]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const q = draft.trim();
    if (q.length < 3 || busy) return;
    setDraft("");
    void ask(q);
  };

  return (
    <div className="mx-auto flex max-w-4xl flex-col">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Ask the literature</h1>
          <p className="mt-1 text-sm text-slate-600">
            Grounded answers with inline citations — every claim links back to a retrieved paper.
          </p>
        </div>
        {turns.length > 0 && (
          <button
            onClick={reset}
            className="flex items-center gap-1 rounded-lg px-2 py-1 text-sm text-slate-600 hover:bg-slate-100"
          >
            <RotateCcw className="h-4 w-4" aria-hidden /> New chat
          </button>
        )}
      </div>

      <div className="mt-6 space-y-8">
        {turns.length === 0 && (
          <div className="grid gap-2 sm:grid-cols-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => void ask(s)}
                className="rounded-xl border border-slate-200 bg-white p-4 text-left text-sm text-slate-700 shadow-sm transition hover:border-indigo-300 hover:shadow"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((t) => (
          <TurnView key={t.id} turn={t} />
        ))}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={submit}
        className="sticky bottom-4 mt-8 flex items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-lg"
      >
        <label htmlFor="question" className="sr-only">
          Your question
        </label>
        <textarea
          id="question"
          value={draft}
          rows={1}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
          placeholder="Ask a research question…"
          className="max-h-40 min-h-11 flex-1 resize-none rounded-xl px-3 py-2.5 outline-none"
        />
        <button
          type="submit"
          disabled={busy || draft.trim().length < 3}
          className="flex h-11 items-center gap-1 rounded-xl bg-indigo-600 px-4 font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          Ask <CornerDownLeft className="h-4 w-4" aria-hidden />
        </button>
      </form>
    </div>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  const [active, setActive] = useState<number | null>(null);
  const cardRefs = useRef(new Map<number, HTMLDivElement>());
  const valid = new Set(turn.sources.map((s) => s.index));
  const cited = new Set(turn.cited.length ? turn.cited : [...valid]);

  const focus = (n: number) => {
    setActive(n);
    cardRefs.current.get(n)?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  };

  return (
    <div className="space-y-3">
      <div className="flex items-start gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-200">
          <User className="h-4 w-4 text-slate-600" aria-hidden />
        </div>
        <p className="pt-1 font-medium text-slate-900">{turn.question}</p>
      </div>
      <div className="flex items-start gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-600">
          <Bot className="h-4 w-4 text-white" aria-hidden />
        </div>
        <div className="min-w-0 flex-1 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          {turn.status === "retrieving" && <Spinner label="Searching papers…" />}
          {turn.answer && (
            <div data-testid="answer">
              <AnswerText
                text={turn.answer}
                validIndices={valid}
                onCite={focus}
                activeIndex={active}
              />
              {turn.status === "streaming" && (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-indigo-400 align-middle" />
              )}
            </div>
          )}
          {turn.status === "error" && (
            <p role="alert" className="text-sm text-red-700">
              {turn.error}
            </p>
          )}
          {turn.status === "done" && (
            <div className="mt-3 text-[11px] text-slate-400">
              {turn.provider} · {turn.tookMs?.toFixed(0)} ms · {turn.sources.length} sources
            </div>
          )}
          {turn.sources.length > 0 && (
            <div className="mt-4 grid items-start gap-2 md:grid-cols-2">
              {turn.sources.map((s) => (
                <CitationCard
                  key={s.index}
                  ref={(el) => {
                    if (el) cardRefs.current.set(s.index, el);
                  }}
                  source={s}
                  active={active === s.index}
                  cited={cited.has(s.index)}
                  onSelect={() => setActive(active === s.index ? null : s.index)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
