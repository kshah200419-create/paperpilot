import { Link } from "react-router-dom";
import type { SearchResult } from "../api/types";
import { authorLine, formatDate } from "../lib/format";
import { CategoryBadge, TopicBadge } from "./Badges";
import { Highlight } from "./Highlight";

export function SearchResultCard({ result, rank }: { result: SearchResult; rank: number }) {
  const { paper } = result;
  return (
    <article className="group rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:border-indigo-200 hover:shadow-md">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 text-sm font-semibold text-slate-400 tabular-nums">{rank}.</span>
        <div className="min-w-0 flex-1">
          <Link
            to={`/papers/${paper.arxiv_id}`}
            className="text-base leading-snug font-semibold text-slate-900 group-hover:text-indigo-700 sm:text-lg"
          >
            {paper.title}
          </Link>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500">
            <span>{authorLine(paper.authors)}</span>
            <span aria-hidden>·</span>
            <time dateTime={paper.published_at}>{formatDate(paper.published_at)}</time>
            {paper.categories.slice(0, 3).map((c) => (
              <CategoryBadge key={c} category={c} primary={c === paper.primary_category} />
            ))}
            {paper.topic_label && <TopicBadge label={paper.topic_label} />}
          </div>
          <p className="mt-3 text-sm leading-relaxed text-slate-600">
            <Highlight text={result.snippet} />
          </p>
          <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-slate-500">
            {result.keyword_rank && (
              <span className="rounded bg-slate-100 px-1.5 py-0.5">
                keyword #{result.keyword_rank}
              </span>
            )}
            {result.vector_rank && (
              <span className="rounded bg-slate-100 px-1.5 py-0.5">
                vector #{result.vector_rank}
              </span>
            )}
            {result.rerank_score !== null && (
              <span className="rounded bg-violet-50 px-1.5 py-0.5 text-violet-700">
                rerank {result.rerank_score.toFixed(2)}
              </span>
            )}
            <span className="ml-auto font-mono">score {result.score.toFixed(4)}</span>
          </div>
        </div>
      </div>
    </article>
  );
}
