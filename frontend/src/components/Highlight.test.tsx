import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Highlight } from "./Highlight";

describe("Highlight", () => {
  it("wraps marked terms in <mark> elements", () => {
    const { container } = render(
      <Highlight text="graph <mark>neural</mark> nets for <mark>RAG</mark>" />,
    );
    const marks = container.querySelectorAll("mark");
    expect([...marks].map((m) => m.textContent)).toEqual(["neural", "RAG"]);
    expect(container.textContent).toBe("graph neural nets for RAG");
  });

  it("renders other markup as literal text (XSS-safe)", () => {
    const { container } = render(
      <Highlight text={'<img src=x onerror="alert(1)"> <mark>ok</mark>'} />,
    );
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<img src=x");
  });
});
