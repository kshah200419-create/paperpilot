import { lazy, Suspense } from "react";
import { Link, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { EmptyState, Spinner } from "./components/States";
import { AskPage } from "./pages/AskPage";
import { PaperPage } from "./pages/PaperPage";
import { SearchPage } from "./pages/SearchPage";

// Charts (Recharts) are the heaviest dependency: load the dashboard on demand.
const TrendsPage = lazy(() =>
  import("./pages/TrendsPage").then((m) => ({ default: m.TrendsPage })),
);

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<SearchPage />} />
        <Route path="ask" element={<AskPage />} />
        <Route
          path="trends"
          element={
            <Suspense fallback={<Spinner label="Loading dashboard…" />}>
              <TrendsPage />
            </Suspense>
          }
        />
        <Route path="papers/*" element={<PaperPage />} />
        <Route
          path="*"
          element={
            <EmptyState title="Page not found">
              <Link to="/" className="text-indigo-600 hover:underline">
                Go to search
              </Link>
            </EmptyState>
          }
        />
      </Route>
    </Routes>
  );
}
