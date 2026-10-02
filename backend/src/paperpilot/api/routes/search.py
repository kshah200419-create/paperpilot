from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from paperpilot.api.deps import Services, get_services
from paperpilot.api.routes.serializers import summaries
from paperpilot.api.schemas import SearchResponse, SearchResultOut
from paperpilot.db.models import Paper
from paperpilot.db.session import get_session
from paperpilot.ml.rerank import get_reranker
from paperpilot.retrieval.search import HybridSearcher, SearchFilters, headlines

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse, summary="Hybrid paper search")
async def search(
    q: Annotated[str, Query(min_length=1, max_length=500, description="Free-text query")],
    session: Annotated[AsyncSession, Depends(get_session)],
    svc: Annotated[Services, Depends(get_services)],
    mode: Literal["keyword", "vector", "hybrid"] = "hybrid",
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    category: Annotated[list[str] | None, Query(description="Filter by arXiv category")] = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    rerank: bool = Query(False, description="Apply the cross-encoder reranker (if available)"),
) -> SearchResponse:
    filters = SearchFilters(tuple(category or ()), date_from, date_to)
    searcher = svc.searcher
    if rerank and searcher.reranker is None:
        reranker = get_reranker(svc.settings, force=True)
        searcher = HybridSearcher(svc.embedder, svc.settings, reranker)
    outcome = await searcher.search_papers(
        session, q, mode=mode, filters=filters, limit=limit, rerank=rerank
    )
    hits = outcome.hits
    papers = {
        p.id: p
        for p in (
            await session.execute(select(Paper).where(Paper.id.in_([h.paper_id for h in hits])))
        ).scalars()
    }
    ordered = [papers[h.paper_id] for h in hits]
    summ = {s.id: s for s in await summaries(session, ordered)}
    snippets = await headlines(session, q, [h.chunk_id for h in hits])
    return SearchResponse(
        query=q,
        mode=mode,
        reranked=bool(rerank and searcher.reranker is not None),
        took_ms=round(outcome.took_ms, 1),
        timings={k: round(v, 1) for k, v in outcome.timings.items()},
        results=[
            SearchResultOut(
                paper=summ[h.paper_id],
                score=round(h.score, 6),
                snippet=snippets.get(h.chunk_id, ""),
                chunk_id=h.chunk_id,
                keyword_rank=h.keyword_rank,
                vector_rank=h.vector_rank,
                rerank_score=h.rerank_score,
            )
            for h in hits
        ],
    )
