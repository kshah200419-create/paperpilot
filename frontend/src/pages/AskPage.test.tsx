import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../test/utils";
import { AskPage } from "./AskPage";

function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

const SOURCES = {
  sources: [1, 2].map((i) => ({
    index: i,
    paper_id: i,
    arxiv_id: `2601.0000${i}`,
    title: `Source paper ${i}`,
    authors: ["Ada"],
    published_at: "2026-01-02T00:00:00Z",
    primary_category: "cs.CL",
    abs_url: `https://arxiv.org/abs/2601.0000${i}`,
    passages: [`Passage text ${i}`],
    score: 0.03,
  })),
};

describe("AskPage", () => {
  afterEach(() => vi.restoreAllMocks());

  it("streams an answer and links citations to source cards", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        sseResponse([
          `event: sources\ndata: ${JSON.stringify(SOURCES)}\n\n`,
          'event: token\ndata: {"text":"Grounding reduces "}\n\nevent: tok',
          'en\ndata: {"text":"hallucination [2]."}\n\n',
          'event: done\ndata: {"provider":"extractive","cited":[2],"took_ms":12}\n\n',
        ]),
      );
    renderWithProviders(<AskPage />);
    await userEvent.type(screen.getByLabelText("Your question"), "Does grounding help?");
    await userEvent.click(screen.getByRole("button", { name: /ask/i }));

    const answer = await screen.findByTestId("answer");
    expect(await within(answer).findByText(/hallucination/)).toBeInTheDocument();
    expect(await screen.findByText(/extractive · 12 ms · 2 sources/)).toBeInTheDocument();
    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/ask/stream",
      expect.objectContaining({ method: "POST" }),
    );

    await userEvent.click(within(answer).getByRole("button", { name: "Show source 2" }));
    const card = screen.getByTestId("citation-2");
    expect(within(card).getByText("Passage text 2")).toBeInTheDocument();
    expect(within(card).getByRole("link", { name: "Source paper 2" })).toHaveAttribute(
      "href",
      "/papers/2601.00002",
    );
  });

  it("shows server errors inline", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      sseResponse([
        'event: sources\ndata: {"sources":[]}\n\n',
        'event: error\ndata: {"message":"LLM down"}\n\n',
      ]),
    );
    renderWithProviders(<AskPage />);
    await userEvent.click(screen.getByText("What attacks bypass LLM safety alignment?"));
    expect(await screen.findByRole("alert")).toHaveTextContent("LLM down");
  });
});
