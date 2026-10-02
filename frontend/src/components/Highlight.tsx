import { Fragment } from "react";

/**
 * Render a server snippet where matches are wrapped in <mark>…</mark> WITHOUT using
 * dangerouslySetInnerHTML: we split on the tags and emit text nodes, so any other markup in
 * the text is shown literally (XSS-safe).
 */
export function Highlight({ text, className }: { text: string; className?: string }) {
  const parts = text.split(/(<mark>.*?<\/mark>)/g).filter(Boolean);
  return (
    <span className={className}>
      {parts.map((part, i) =>
        part.startsWith("<mark>") && part.endsWith("</mark>") ? (
          <mark key={i} className="rounded bg-amber-200/70 px-0.5 text-slate-900">
            {part.slice(6, -7)}
          </mark>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </span>
  );
}
