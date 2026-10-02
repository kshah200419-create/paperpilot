import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Search, SlidersHorizontal, Sparkles } from "lucide-react";
import { useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { SearchMode } from "../api/types";
import { SearchResultCard } from "../components/SearchResultCard";
import { EmptyState, ErrorState, SkeletonCard } from "../components/States";

const MODES: { value: SearchMode; label: string; hint: string }[] = [
  { value: "hybrid", label: "Hybrid", hint: "Full-text + vectors fused with RRF" },
  { value: "keyword", label: "Keyword", hint: "Postgres full-text search" },
  { value: "vector", label: "Semantic", hint: "pgvector cosine similarity" },
];

const EXAMPLES = [
  "efficient inference for mixture-of-experts models",
  "jailbreak attacks on LLMs",
  "SLAM for legged robots",
  "foundation models for medical imaging",
];

export function SearchPage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const mode = (params.get("mode") as SearchMode | null) ?? "hybrid";
  const categories = params.getAll("category");
  const dateFrom = params.get("from") ?? "";
  const dateTo = params.get("to") ?? "";
  const rerank = params.get("rerank") === "1";
  const [draft, setDraft] = useState(q);
  const [showFilters, setShowFilters] = useState(categories.length > 0 || !!dateFrom || !!dateTo);

  const cats = useQuery({ queryKey: ["categories"], queryFn: api.categories, staleTime: 300_000 });
  const search = useQuery({
    queryKey: ["search", q, mode, categories, dateFrom, dateTo, rerank],
    queryFn: () => api.search({ q, mode, categories, dateFrom, dateTo, rerank }),
    enabled: q.trim().length > 0,
    placeholderData: (prev) => prev,
  });

  const update = (patch: Record<string, string | string[] | null>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) {
      next.delete(k);
      if (Array.isArray(v)) v.forEach((x) => next.append(k, x));
      else if (v) next.set(k, v);
    }
    setParams(next, { replace: false });
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    update({ q: draft.trim() || null });
  };

  const toggleCategory = (c: string) =>
    update({
      category: categories.includes(c) ? categories.filter((x) => x !== c) : [...categories, c],
    });

  return (
    <div>
      <section className="mx-auto max-w-3xl text-center">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
          Find the right paper, <span className="text-indigo-600">fast</span>.
        </h1>
        <p className="mt-2 text-slate-600">
          Hybrid search over arXiv abstracts — keyword precision plus semantic recall.
        </p>
        <form onSubmit={submit} className="mt-6 flex gap-2" role="search">
          <label htmlFor="q" className="sr-only">
            Search papers
          </label>
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2 text-slate-400" />
            <input
              id="q"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="e.g. retrieval-augmented generation with citations"
              className="w-full rounded-xl border border-slate-300 bg-white py-3 pr-3 pl-10 shadow-sm outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100"
            />
          </div>
          <button
            type="submit"
            className="rounded-xl bg-indigo-600 px-5 font-semibold text-white shadow-sm hover:bg-indigo-700"
          >
            Search
          </button>
        </form>

        <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
          <div
            className="inline-flex rounded-xl bg-slate-100 p-1"
            role="radiogroup"
            aria-label="Search mode"
          >
            {MODES.map((m) => (
              <button
                key={m.value}
                role="radio"
                aria-checked={mode === m.value}
                title={m.hint}
                onClick={() => update({ mode: m.value })}
                className={clsx(
                  "rounded-lg px-3 py-1 text-sm font-medium transition",
                  mode === m.value
                    ? "bg-white text-indigo-700 shadow-sm"
                    : "text-slate-600 hover:text-slate-900",
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-1.5 text-sm text-slate-600">
            <input
              type="checkbox"
              checked={rerank}
              onChange={(e) => update({ rerank: e.target.checked ? "1" : null })}
              className="rounded border-slate-300 text-indigo-600"
            />
            <Sparkles className="h-3.5 w-3.5 text-violet-500" aria-hidden /> Rerank
          </label>
          <button
            onClick={() => setShowFilters((s) => !s)}
            className="flex items-center gap-1 rounded-lg px-2 py-1 text-sm text-slate-600 hover:bg-slate-100"
            aria-expanded={showFilters}
          >
            <SlidersHorizontal className="h-4 w-4" aria-hidden /> Filters
            {categories.length + (dateFrom ? 1 : 0) + (dateTo ? 1 : 0) > 0 && (
              <span className="rounded-full bg-indigo-600 px-1.5 text-[10px] text-white">
                {categories.length + (dateFrom ? 1 : 0) + (dateTo ? 1 : 0)}
              </span>
            )}
          </button>
        </div>

        {showFilters && (
          <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4 text-left shadow-sm">
            <div className="text-xs font-semibold tracking-wide text-slate-500 uppercase">
              Categories
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {(cats.data ?? []).slice(0, 14).map((c) => (
                <button
                  key={c.category}
                  onClick={() => toggleCategory(c.category)}
                  aria-pressed={categories.includes(c.category)}
                  className={clsx(
                    "rounded-full border px-2.5 py-0.5 font-mono text-xs transition",
                    categories.includes(c.category)
                      ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                      : "border-slate-200 text-slate-600 hover:border-slate-300",
                  )}
                >
                  {c.category} <span className="text-slate-400">{c.count}</span>
                </button>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap gap-4 text-sm">
              <label className="flex items-center gap-2 text-slate-600">
                From
                <input
                  type="date"
                  value={dateFrom}
                  onChange={(e) => update({ from: e.target.value || null })}
                  className="rounded-lg border border-slate-300 px-2 py-1"
                />
              </label>
              <label className="flex items-center gap-2 text-slate-600">
                To
                <input
                  type="date"
                  value={dateTo}
                  onChange={(e) => update({ to: e.target.value || null })}
                  className="rounded-lg border border-slate-300 px-2 py-1"
                />
              </label>
              {(categories.length > 0 || dateFrom || dateTo) && (
                <button
                  onClick={() => update({ category: [], from: null, to: null })}
                  className="text-indigo-600 hover:underline"
                >
                  Clear filters
                </button>
              )}
            </div>
          </div>
        )}
      </section>

      <section className="mx-auto mt-8 max-w-3xl" aria-live="polite">
        {!q && (
          <div className="text-center text-sm text-slate-500">
            Try:{" "}
            {EXAMPLES.map((ex, i) => (
              <span key={ex}>
                <button
                  className="text-indigo-600 hover:underline"
                  onClick={() => {
                    setDraft(ex);
                    update({ q: ex });
                  }}
                >
                  {ex}
                </button>
                {i < EXAMPLES.length - 1 && " · "}
              </span>
            ))}
          </div>
        )}
        {q && search.isPending && (
          <div className="space-y-3">
            {[0, 1, 2].map((i) => (
              <SkeletonCard key={i} />
            ))}
          </div>
        )}
        {q && search.isError && (
          <ErrorState error={search.error} onRetry={() => search.refetch()} />
        )}
        {q && search.data && (
          <>
            <div className="mb-3 flex items-center justify-between text-xs text-slate-500">
              <span>
                {search.data.results.length} results for <strong>“{search.data.query}”</strong>
              </span>
              <span className={clsx(search.isFetching && "animate-pulse")}>
                {search.data.mode}
                {search.data.reranked && " + rerank"} · {search.data.took_ms.toFixed(0)} ms
              </span>
            </div>
            {search.data.results.length === 0 ? (
              <EmptyState title="No matching papers">
                Try a broader query, switch to semantic mode, or clear filters.
              </EmptyState>
            ) : (
              <div className="space-y-3">
                {search.data.results.map((r, i) => (
                  <SearchResultCard key={r.paper.id} result={r} rank={i + 1} />
                ))}
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
