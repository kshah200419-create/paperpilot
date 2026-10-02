import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { TrendingDown, TrendingUp } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { api } from "../api/client";
import type { TopicPoint } from "../api/types";
import { EmptyState, ErrorState, Spinner } from "../components/States";
import { colorFor, formatDate, formatMonth } from "../lib/format";

export function TrendsPage() {
  const [groupBy, setGroupBy] = useState<"topic" | "category">("topic");
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const trends = useQuery({ queryKey: ["trends", groupBy], queryFn: () => api.trends(groupBy) });
  const topics = useQuery({ queryKey: ["topics"], queryFn: api.topics });
  const map = useQuery({ queryKey: ["topic-map"], queryFn: api.topicMap });
  const navigate = useNavigate();

  const chartData = useMemo(() => {
    if (!trends.data) return [];
    return trends.data.months.map((m, i) => {
      const row: Record<string, number | string> = { month: formatMonth(m) };
      trends.data.series.forEach((s) => (row[s.label] = s.counts[i] ?? 0));
      return row;
    });
  }, [trends.data]);

  const topicIndex = useMemo(
    () => new Map((topics.data ?? []).map((t, i) => [t.id, i])),
    [topics.data],
  );
  const pointsByTopic = useMemo(() => {
    const groups = new Map<number, TopicPoint[]>();
    for (const p of map.data ?? []) {
      groups.set(p.topic_id, [...(groups.get(p.topic_id) ?? []), p]);
    }
    return groups;
  }, [map.data]);

  if (trends.isError) return <ErrorState error={trends.error} onRetry={() => trends.refetch()} />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Research trends</h1>
        <p className="mt-1 text-sm text-slate-600">
          Topics discovered with KMeans on paper embeddings, labelled with c-TF-IDF keywords.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="Papers" value={stats.data?.papers.toLocaleString()} />
        <StatCard label="Passages" value={stats.data?.chunks.toLocaleString()} />
        <StatCard label="Topics" value={stats.data?.topics} />
        <StatCard
          label="Coverage"
          value={
            stats.data?.first_published && stats.data.last_published
              ? `${formatDate(stats.data.first_published, { day: undefined })} – ${formatDate(stats.data.last_published, { day: undefined })}`
              : undefined
          }
          small
        />
      </div>

      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="font-semibold">Papers per month</h2>
          <div
            className="inline-flex rounded-xl bg-slate-100 p-1 text-sm"
            role="radiogroup"
            aria-label="Group by"
          >
            {(["topic", "category"] as const).map((g) => (
              <button
                key={g}
                role="radio"
                aria-checked={groupBy === g}
                onClick={() => setGroupBy(g)}
                className={clsx(
                  "rounded-lg px-3 py-1 font-medium capitalize",
                  groupBy === g ? "bg-white text-indigo-700 shadow-sm" : "text-slate-600",
                )}
              >
                by {g}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-4 h-80">
          {trends.isPending ? (
            <Spinner />
          ) : chartData.length === 0 ? (
            <EmptyState title="No trend data yet">
              Ingest papers and rebuild topics first.
            </EmptyState>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ left: -20, right: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {trends.data?.series.map((s, i) => (
                  <Area
                    key={s.key}
                    type="monotone"
                    dataKey={s.label}
                    stackId="1"
                    stroke={colorFor(i)}
                    fill={colorFor(i)}
                    fillOpacity={0.55}
                    isAnimationActive={false}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-5">
        <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6 lg:col-span-2">
          <h2 className="font-semibold">Momentum</h2>
          <p className="text-xs text-slate-500">Share of the last 3 months vs. overall share.</p>
          <ul className="mt-3 space-y-2">
            {[...(trends.data?.series ?? [])]
              .sort((a, b) => b.momentum - a.momentum)
              .map((s) => (
                <li key={s.key} className="flex items-center gap-2 text-sm">
                  {s.momentum >= 1 ? (
                    <TrendingUp className="h-4 w-4 shrink-0 text-emerald-600" aria-label="rising" />
                  ) : (
                    <TrendingDown className="h-4 w-4 shrink-0 text-rose-500" aria-label="falling" />
                  )}
                  <span className="min-w-0 flex-1 truncate">{s.label}</span>
                  <span
                    className={clsx(
                      "font-mono text-xs",
                      s.momentum >= 1 ? "text-emerald-700" : "text-rose-600",
                    )}
                  >
                    ×{s.momentum.toFixed(2)}
                  </span>
                </li>
              ))}
          </ul>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6 lg:col-span-3">
          <h2 className="font-semibold">Topic map</h2>
          <p className="text-xs text-slate-500">
            2-D PCA projection of paper embeddings. Click a dot to open the paper.
          </p>
          <div className="mt-2 h-80">
            {map.isPending ? (
              <Spinner />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ left: -20, right: 8, top: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" dataKey="x" tick={false} />
                  <YAxis type="number" dataKey="y" tick={false} />
                  <ZAxis range={[28, 28]} />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    content={({ payload }) => {
                      const p = payload?.[0]?.payload as TopicPoint | undefined;
                      return p ? (
                        <div className="max-w-xs rounded-lg border border-slate-200 bg-white p-2 text-xs shadow">
                          {p.title}
                        </div>
                      ) : null;
                    }}
                  />
                  {[...pointsByTopic.entries()].map(([topicId, pts]) => (
                    <Scatter
                      key={topicId}
                      data={pts}
                      fill={colorFor(topicIndex.get(topicId) ?? 0)}
                      fillOpacity={0.75}
                      isAnimationActive={false}
                      onClick={(d: unknown) => {
                        const p = (d as { payload?: TopicPoint }).payload;
                        if (p) navigate(`/papers/${p.arxiv_id}`);
                      }}
                      className="cursor-pointer"
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
        </section>
      </div>

      <section>
        <h2 className="font-semibold">Topics</h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {(topics.data ?? []).map((t, i) => (
            <div key={t.id} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full" style={{ background: colorFor(i) }} />
                <h3 className="flex-1 truncate font-medium">{t.label}</h3>
                <span className="text-xs text-slate-500">{t.size} papers</span>
              </div>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {t.keywords.map((k) => (
                  <Link
                    key={k}
                    to={`/?q=${encodeURIComponent(k)}`}
                    className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-700 hover:bg-indigo-50 hover:text-indigo-700"
                  >
                    {k}
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  small,
}: {
  label: string;
  value?: string | number;
  small?: boolean;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-medium tracking-wide text-slate-500 uppercase">{label}</div>
      <div className={clsx("mt-1 font-bold text-slate-900", small ? "text-sm" : "text-2xl")}>
        {value ?? "—"}
      </div>
    </div>
  );
}
