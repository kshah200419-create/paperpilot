"""Prompt construction for grounded, citation-bearing answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

SYSTEM_PROMPT = (
    "You are PaperPilot, a meticulous research assistant. Answer the user's question using ONLY "
    "the numbered sources provided (excerpts from arXiv abstracts). Cite every claim inline with "
    "the source number in square brackets, e.g. [1] or [2][3]. If the sources do not contain the "
    "answer, say so plainly instead of guessing. Be concise: 3-6 sentences or a short list."
)


@dataclass
class Source:
    index: int
    paper_id: int
    arxiv_id: str
    title: str
    authors: list[str]
    published_at: datetime
    primary_category: str
    abs_url: str
    passages: list[str] = field(default_factory=list)
    score: float = 0.0


def format_sources(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        excerpt = " ".join(s.passages)
        blocks.append(
            f"[{s.index}] {s.title} ({s.published_at:%Y-%m}, arXiv:{s.arxiv_id})\n{excerpt}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, sources: list[Source]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Sources:\n\n{format_sources(sources)}\n\nQuestion: {question}\n\n"
            "Answer with inline citations:",
        },
    ]
