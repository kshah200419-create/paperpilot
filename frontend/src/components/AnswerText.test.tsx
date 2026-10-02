import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AnswerText } from "./AnswerText";

describe("AnswerText", () => {
  it("turns valid [n] markers into clickable citation chips", async () => {
    const onCite = vi.fn();
    render(
      <AnswerText
        text={"Intro:\n- RAG grounds answers [1].\n- Ignored marker [9] stays text [2]."}
        validIndices={new Set([1, 2])}
        onCite={onCite}
      />,
    );
    const chip = screen.getByRole("button", { name: "Show source 1" });
    await userEvent.click(chip);
    expect(onCite).toHaveBeenCalledWith(1);
    expect(screen.getByRole("button", { name: "Show source 2" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show source 9" })).toBeNull();
    expect(screen.getByText(/Ignored marker \[9\]/)).toBeInTheDocument();
  });
});
