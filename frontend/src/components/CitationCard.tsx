import clsx from "clsx";
import { ExternalLink } from "lucide-react";
import { forwardRef } from "react";
import { Link } from "react-router-dom";
import type { Source } from "../api/types";
import { authorLine, formatDate } from "../lib/format";
import { CategoryBadge } from "./Badges";

interface Props {
  source: Source;
  active: boolean;
  cited: boolean;
  onSelect: () => void;
}

export const CitationCard = forwardRef<HTMLDivElement, Props>(function CitationCard(
  { source, active, cited, onSelect },
  ref,
) {
  return (
    <div
      ref={ref}
      data-testid={`citation-${source.index}`}
      onClick={onSelect}
      className={clsx(
        "cursor-pointer rounded-xl border bg-white p-3 text-sm transition",
        active
          ? "border-indigo-400 shadow-md ring-2 ring-indigo-200"
          : "border-slate-200 hover:border-slate-300",
        !cited && "opacity-70",
      )}
    >
      <div className="flex items-start gap-2">
        <span className="mt-0.5 flex h-5 min-w-5 items-center justify-center rounded-md bg-indigo-600 px-1 text-[11px] font-bold text-white">
          {source.index}
        </span>
        <div className="min-w-0 flex-1">
          <Link
            to={`/papers/${source.arxiv_id}`}
            onClick={(e) => e.stopPropagation()}
            className="line-clamp-2 font-medium text-slate-900 hover:text-indigo-700"
          >
            {source.title}
          </Link>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500">
            <span>{authorLine(source.authors, 2)}</span>
            <span>·</span>
            <span>{formatDate(source.published_at, { day: undefined })}</span>
            <CategoryBadge category={source.primary_category} primary />
          </div>
          {active && (
            <blockquote className="mt-2 max-h-40 overflow-y-auto border-l-2 border-indigo-200 pl-2 text-xs leading-relaxed text-slate-600">
              {source.passages.join(" … ")}
            </blockquote>
          )}
          <a
            href={source.abs_url}
            target="_blank"
            rel="noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="mt-1.5 inline-flex items-center gap-1 text-xs text-indigo-600 hover:underline"
          >
            arXiv:{source.arxiv_id} <ExternalLink className="h-3 w-3" aria-hidden />
          </a>
        </div>
      </div>
    </div>
  );
});
