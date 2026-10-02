import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ExternalLink, FileText, MessageSquareText } from "lucide-react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { CategoryBadge, TopicBadge } from "../components/Badges";
import { EmptyState, ErrorState, SkeletonCard } from "../components/States";
import { authorLine, formatDate } from "../lib/format";

export function PaperPage() {
  const { "*": arxivId = "" } = useParams();
  const navigate = useNavigate();
  const paper = useQuery({
    queryKey: ["paper", arxivId],
    queryFn: () => api.paper(arxivId),
    retry: 1,
  });

  if (paper.isPending) return <SkeletonCard />;
  if (paper.isError) {
    if (paper.error instanceof ApiError && paper.error.status === 404) {
      return (
        <EmptyState title="Paper not found">
          <Link to="/" className="text-indigo-600 hover:underline">
            Back to search
          </Link>
        </EmptyState>
      );
    }
    return <ErrorState error={paper.error} onRetry={() => paper.refetch()} />;
  }
  const p = paper.data;
  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <article className="lg:col-span-2">
        <button
          onClick={() => navigate(-1)}
          className="mb-4 flex items-center gap-1 text-sm text-slate-500 hover:text-slate-800"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden /> Back
        </button>
        <div className="flex flex-wrap gap-1.5">
          {p.categories.map((c) => (
            <CategoryBadge key={c} category={c} primary={c === p.primary_category} />
          ))}
          {p.topic_label && <TopicBadge label={p.topic_label} />}
        </div>
        <h1 className="mt-3 text-2xl leading-tight font-bold tracking-tight text-slate-900 sm:text-3xl">
          {p.title}
        </h1>
        <p className="mt-2 text-sm text-slate-600">{authorLine(p.authors, 8)}</p>
        <p className="mt-1 text-xs text-slate-500">
          Published {formatDate(p.published_at)} · v{p.version} · arXiv:{p.arxiv_id}
          {p.comment && ` · ${p.comment}`}
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <a
            href={p.abs_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium hover:bg-slate-50"
          >
            <ExternalLink className="h-4 w-4" aria-hidden /> arXiv
          </a>
          {p.pdf_url && (
            <a
              href={p.pdf_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium hover:bg-slate-50"
            >
              <FileText className="h-4 w-4" aria-hidden /> PDF
            </a>
          )}
          <Link
            to={`/ask?q=${encodeURIComponent(`What is the main contribution of "${p.title}"?`)}`}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700"
          >
            <MessageSquareText className="h-4 w-4" aria-hidden /> Ask about this paper
          </Link>
        </div>
        <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold tracking-wide text-slate-500 uppercase">Abstract</h2>
          <p className="mt-2 leading-relaxed text-slate-800">{p.abstract}</p>
        </section>
        <section className="mt-6">
          <h2 className="text-sm font-semibold tracking-wide text-slate-500 uppercase">
            Indexed passages ({p.chunks.length})
          </h2>
          <ol className="mt-2 space-y-2">
            {p.chunks.map((c) => (
              <li
                key={c.id}
                className="rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-700"
              >
                <span className="mr-2 font-mono text-xs text-slate-400">#{c.ordinal}</span>
                {c.text}
              </li>
            ))}
          </ol>
        </section>
      </article>
      <aside className="space-y-4 lg:pt-10">
        {p.topic && (
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="text-sm font-semibold">Topic</h2>
            <p className="mt-1 text-sm text-slate-700">{p.topic.label}</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {p.topic.keywords.map((k) => (
                <span
                  key={k}
                  className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600"
                >
                  {k}
                </span>
              ))}
            </div>
          </div>
        )}
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="text-sm font-semibold">Similar papers</h2>
          <ul className="mt-2 space-y-3">
            {p.similar.map((s) => (
              <li key={s.id}>
                <Link
                  to={`/papers/${s.arxiv_id}`}
                  className="text-sm font-medium text-slate-800 hover:text-indigo-700"
                >
                  {s.title}
                </Link>
                <div className="text-xs text-slate-500">
                  {formatDate(s.published_at, { day: undefined })} · {s.primary_category}
                </div>
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  );
}
