import { Fragment } from "react";

interface Props {
  text: string;
  validIndices: Set<number>;
  onCite: (index: number) => void;
  activeIndex?: number | null;
}

/** Render an answer, turning inline `[n]` markers into clickable citation chips. */
export function AnswerText({ text, validIndices, onCite, activeIndex }: Props) {
  const lines = text.split("\n");
  return (
    <div className="space-y-1.5 leading-relaxed text-slate-800">
      {lines.map((line, li) => {
        if (!line.trim()) return <div key={li} className="h-1" />;
        const bullet = /^\s*[-*]\s+/.test(line);
        const body = line.replace(/^\s*[-*]\s+/, "");
        const parts = body.split(/(\[\d{1,2}\])/g);
        return (
          <p key={li} className={bullet ? "relative pl-4" : undefined}>
            {bullet && <span className="absolute left-0 text-indigo-400">•</span>}
            {parts.map((part, i) => {
              const m = /^\[(\d{1,2})\]$/.exec(part);
              const n = m ? Number(m[1]) : NaN;
              if (m && validIndices.has(n)) {
                return (
                  <button
                    key={i}
                    type="button"
                    onClick={() => onCite(n)}
                    aria-label={`Show source ${n}`}
                    className={
                      "mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md px-1 align-baseline text-[11px] font-semibold transition " +
                      (activeIndex === n
                        ? "bg-indigo-600 text-white"
                        : "bg-indigo-100 text-indigo-700 hover:bg-indigo-200")
                    }
                  >
                    {n}
                  </button>
                );
              }
              return <Fragment key={i}>{part}</Fragment>;
            })}
          </p>
        );
      })}
    </div>
  );
}
