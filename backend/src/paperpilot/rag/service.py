"""RAG orchestration: retrieve -> build numbered sources -> generate (streaming) -> cite."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from paperpilot.db.models import Paper
from paperpilot.logs import get_logger
from paperpilot.rag.prompt import Source
from paperpilot.rag.providers import ExtractiveProvider, LLMError, LLMProvider, cited_indices
from paperpilot.retrieval.search import HybridSearcher, SearchFilters

log = get_logger(__name__)


@dataclass
class AskEvent:
    event: str
    data: dict[str, Any]


async def gather_sources(
    session: AsyncSession,
    searcher: HybridSearcher,
    question: str,
    filters: SearchFilters | None = None,
    max_papers: int = 5,
) -> list[Source]:
    contexts = await searcher.search_chunks(
        session, question, filters=filters, max_papers=max_papers, per_paper=2
    )
    if not contexts:
        return []
    paper_ids = list(dict.fromkeys(c.paper_id for c in contexts))
    rows = (await session.execute(select(Paper).where(Paper.id.in_(paper_ids)))).scalars().all()
    papers = {p.id: p for p in rows}
    sources: dict[int, Source] = {}
    for ctx in contexts:
        p = papers[ctx.paper_id]
        if p.id not in sources:
            sources[p.id] = Source(
                index=len(sources) + 1,
                paper_id=p.id,
                arxiv_id=p.arxiv_id,
                title=p.title,
                authors=list(p.authors),
                published_at=p.published_at,
                primary_category=p.primary_category,
                abs_url=p.abs_url,
                score=ctx.score,
            )
        sources[p.id].passages.append(ctx.text)
    return list(sources.values())


def source_payload(s: Source) -> dict[str, Any]:
    return {
        "index": s.index,
        "paper_id": s.paper_id,
        "arxiv_id": s.arxiv_id,
        "title": s.title,
        "authors": s.authors[:6],
        "published_at": s.published_at.isoformat(),
        "primary_category": s.primary_category,
        "abs_url": s.abs_url,
        "passages": s.passages,
        "score": round(s.score, 5),
    }


async def stream_answer(
    session: AsyncSession,
    searcher: HybridSearcher,
    provider: LLMProvider,
    question: str,
    filters: SearchFilters | None = None,
    max_papers: int = 5,
) -> AsyncIterator[AskEvent]:
    """Yield ``sources`` -> many ``token`` -> ``done`` events (or ``error``)."""
    started = time.perf_counter()
    sources = await gather_sources(session, searcher, question, filters, max_papers)
    yield AskEvent("sources", {"sources": [source_payload(s) for s in sources]})

    if not sources:
        text = "I couldn't find any indexed papers relevant to this question. Try ingesting more papers or rephrasing."
        yield AskEvent("token", {"text": text})
        yield AskEvent("done", {"provider": provider.name, "cited": [], "took_ms": _ms(started)})
        return

    used = provider
    parts: list[str] = []
    try:
        async for piece in provider.stream(question, sources):
            parts.append(piece)
            yield AskEvent("token", {"text": piece})
    except LLMError as exc:
        log.warning("llm.failed", provider=provider.name, error=str(exc))
        if parts:  # partial answer already streamed: report error, don't mix providers
            yield AskEvent("error", {"message": str(exc)})
            return
        used = ExtractiveProvider(searcher.embedder)
        async for piece in used.stream(question, sources):
            parts.append(piece)
            yield AskEvent("token", {"text": piece})
    answer = "".join(parts)
    valid = {s.index for s in sources}
    yield AskEvent(
        "done",
        {
            "provider": used.name if used is provider else f"{used.name} (fallback)",
            "cited": [i for i in cited_indices(answer) if i in valid],
            "took_ms": _ms(started),
        },
    )


async def answer_question(
    session: AsyncSession,
    searcher: HybridSearcher,
    provider: LLMProvider,
    question: str,
    filters: SearchFilters | None = None,
    max_papers: int = 5,
) -> dict[str, Any]:
    """Non-streaming variant: collect the event stream into one JSON payload."""
    out: dict[str, Any] = {"answer": "", "sources": [], "cited": [], "provider": provider.name}
    async for ev in stream_answer(session, searcher, provider, question, filters, max_papers):
        if ev.event == "sources":
            out["sources"] = ev.data["sources"]
        elif ev.event == "token":
            out["answer"] += ev.data["text"]
        elif ev.event == "done":
            out.update(ev.data)
        elif ev.event == "error":
            out["error"] = ev.data["message"]
    return out


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
