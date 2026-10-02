import { describe, expect, it } from "vitest";
import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("parses complete events", () => {
    const p = new SSEParser();
    const out = p.push('event: token\ndata: {"text":"hi"}\n\nevent: done\ndata: {}\n\n');
    expect(out).toEqual([
      { event: "token", data: '{"text":"hi"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("buffers events split across chunks and handles CRLF", () => {
    const p = new SSEParser();
    expect(p.push("event: tok")).toEqual([]);
    expect(p.push('en\r\ndata: {"a":')).toEqual([]);
    expect(p.push("1}\r\n\r\n")).toEqual([{ event: "token", data: '{"a":1}' }]);
  });

  it("defaults event name, joins multi-line data and skips comments", () => {
    const p = new SSEParser();
    expect(p.push(": keep-alive\n\ndata: a\ndata: b\n\n")).toEqual([
      { event: "message", data: "a\nb" },
    ]);
  });
});
