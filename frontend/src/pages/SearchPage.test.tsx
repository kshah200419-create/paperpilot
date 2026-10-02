import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SearchResponse } from "../api/types";
import { jsonResponse, renderWithProviders } from "../test/utils";
import { SearchPage } from "./SearchPage";

const RESULT: SearchResponse = {
  query: "mixture of experts",
  mode: "hybrid",
  reranked: false,
  took_ms: 21.4,
  timings: {},
  results: [
    {
      paper: {
        id: 1,
        arxiv_id: "2609.04575",
        title: "Training-Free Halving of Activated Experts",
        authors: ["A. Author", "B. Author"],
        primary_category: "cs.LG",
        categories: ["cs.LG", "cs.CL"],
        published_at: "2026-09-04T10:00:00Z",
        abs_url: "https://arxiv.org/abs/2609.04575",
        pdf_url: null,
        topic_id: 3,
        topic_label: "experts · routing · moe",
      },
      score: 0.0325,
      snippet: "Fine-grained <mark>Mixture-of-Experts</mark> models route each token",
      chunk_id: 10,
      keyword_rank: 1,
      vector_rank: 2,
      rerank_score: null,
    },
  ],
};

function mockFetch(search: () => Response) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.includes("/api/categories")) return jsonResponse([{ category: "cs.LG", count: 10 }]);
    if (url.includes("/api/search")) return search();
    return jsonResponse({});
  });
}

describe("SearchPage", () => {
  afterEach(() => vi.restoreAllMocks());

  it("shows example queries before searching", () => {
    mockFetch(() => jsonResponse(RESULT));
    renderWithProviders(<SearchPage />);
    expect(screen.getByText(/jailbreak attacks on LLMs/)).toBeInTheDocument();
  });

  it("runs a hybrid search and renders highlighted results", async () => {
    const fetchSpy = mockFetch(() => jsonResponse(RESULT));
    renderWithProviders(<SearchPage />);
    await userEvent.type(screen.getByLabelText("Search papers"), "mixture of experts{enter}");
    expect(
      await screen.findByText("Training-Free Halving of Activated Experts"),
    ).toBeInTheDocument();
    expect(screen.getByText("Mixture-of-Experts").tagName).toBe("MARK");
    expect(screen.getByText("keyword #1")).toBeInTheDocument();
    expect(screen.getByText("vector #2")).toBeInTheDocument();
    const searchUrl = fetchSpy.mock.calls
      .map((c) => String(c[0]))
      .find((u) => u.includes("/api/search"));
    expect(searchUrl).toContain("mode=hybrid");
    expect(searchUrl).toContain("q=mixture+of+experts");
  });

  it("switching mode re-queries with the new mode", async () => {
    const fetchSpy = mockFetch(() => jsonResponse({ ...RESULT, mode: "vector" }));
    renderWithProviders(<SearchPage />, { route: "/?q=moe" });
    await screen.findByText("Training-Free Halving of Activated Experts");
    await userEvent.click(screen.getByRole("radio", { name: "Semantic" }));
    await waitFor(() =>
      expect(fetchSpy.mock.calls.some((c) => String(c[0]).includes("mode=vector"))).toBe(true),
    );
  });

  it("shows an empty state when nothing matches", async () => {
    mockFetch(() => jsonResponse({ ...RESULT, results: [] }));
    renderWithProviders(<SearchPage />, { route: "/?q=zzz" });
    expect(await screen.findByText("No matching papers")).toBeInTheDocument();
  });

  it("shows an error state with retry when the API fails", async () => {
    mockFetch(() => jsonResponse({ detail: "database unavailable" }, 503));
    renderWithProviders(<SearchPage />, { route: "/?q=moe" });
    expect(await screen.findByRole("alert")).toHaveTextContent("database unavailable");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
