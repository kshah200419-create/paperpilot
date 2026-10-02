import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { BarChart3, MessageSquareText, Search } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api/client";

const NAV = [
  { to: "/", label: "Search", icon: Search },
  { to: "/ask", label: "Ask", icon: MessageSquareText },
  { to: "/trends", label: "Trends", icon: BarChart3 },
];

export function Layout() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, staleTime: 60_000 });
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats, staleTime: 60_000 });
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3">
          <NavLink to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span className="text-lg">
              Paper<span className="text-indigo-600">Pilot</span>
            </span>
          </NavLink>
          <nav className="ml-auto flex gap-1" aria-label="Main">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  clsx(
                    "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition",
                    isActive
                      ? "bg-indigo-50 text-indigo-700"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
                  )
                }
              >
                <Icon className="h-4 w-4" aria-hidden />
                <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:py-8">
        <Outlet />
      </main>
      <footer className="border-t border-slate-200 bg-white py-4 text-xs text-slate-500">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-4 gap-y-1 px-4">
          <span>
            {stats.data
              ? `${stats.data.papers.toLocaleString()} papers · ${stats.data.chunks.toLocaleString()} passages · ${stats.data.topics} topics`
              : "—"}
          </span>
          {health.data && (
            <span className="flex items-center gap-1.5">
              <span
                className={clsx(
                  "h-2 w-2 rounded-full",
                  health.data.status === "ok" ? "bg-emerald-500" : "bg-amber-500",
                )}
              />
              API {health.data.status} · embedder {health.data.embedder ?? "—"} · LLM{" "}
              {health.data.llm_provider ?? "—"}
            </span>
          )}
          <a className="ml-auto hover:text-slate-700" href="/docs" target="_blank" rel="noreferrer">
            API docs
          </a>
        </div>
      </footer>
    </div>
  );
}
