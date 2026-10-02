"""Hybrid retrieval over Postgres: full-text (tsvector) + pgvector, fused with RRF.

Retrieval happens at *chunk* level; paper-level results take each paper's best chunk.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
from sqlalchemy import ColumnElement, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from paperpilot.config import Settings
from paperpilot.db.models import Chunk, Paper
from paperpilot.ml.embeddings import Embedder
from paperpilot.ml.rerank import Reranker
from paperpilot.retrieval.fusion import rrf_fuse, unique_in_order

SearchMode = Literal["keyword", "vector", "hybrid"]

_TOKEN = re.compile(r"[A-Za-z0-9]+")
HEADLINE_OPTS = (
    "StartSel=<mark>, StopSel=</mark>, MaxWords=45, MinWords=20, "
    'MaxFragments=2, FragmentDelimiter=" … "'
)


@dataclass(frozen=True)
class SearchFilters:
    categories: tuple[str, ...] = ()
    date_from: datetime | None = None
    date_to: datetime | None = None


@dataclass
class ChunkHit:
    chunk_id: int
    paper_id: int
    score: float


@dataclass
class PaperHit:
    paper_id: int
    chunk_id: int
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class ChunkContext:
    chunk_id: int
    paper_id: int
    text: str
    score: float


@dataclass
class SearchOutcome:
    hits: list[PaperHit]
    took_ms: float
    timings: dict[str, float] = field(default_factory=dict)


def to_or_tsquery(query: str, max_terms: int = 32) -> str:
    """Build an OR-of-terms tsquery string from free text (Postgres stems & drops stopwords).

    ``websearch_to_tsquery`` ANDs terms, which is too strict for natural-language questions.
    Tokens are restricted to ``[A-Za-z0-9]`` so the string is always a valid tsquery.
    """
    tokens = unique_in_order(t.lower() for t in _TOKEN.findall(query) if len(t) > 1)
    return " | ".join(tokens[:max_terms])


def _filter_clauses(f: SearchFilters) -> list[ColumnElement[bool]]:
    clauses: list[ColumnElement[bool]] = []
    if f.categories:
        clauses.append(Paper.categories.overlap(list(f.categories)))
    if f.date_from:
        clauses.append(Paper.published_at >= f.date_from)
    if f.date_to:
        clauses.append(Paper.published_at <= f.date_to)
    return clauses


async def keyword_chunks(
    session: AsyncSession, query: str, filters: SearchFilters, limit: int
) -> list[ChunkHit]:
    tsq_text = to_or_tsquery(query)
    if not tsq_text:
        return []
    tsq = func.to_tsquery("english", tsq_text)
    # normalization 1: divide by 1 + log(document length)
    rank = func.ts_rank(Chunk.tsv, tsq, 1).label("score")
    stmt = (
        select(Chunk.id, Chunk.paper_id, rank)
        .join(Paper, Paper.id == Chunk.paper_id)
        .where(Chunk.tsv.op("@@")(tsq), *_filter_clauses(filters))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [ChunkHit(r[0], r[1], float(r[2])) for r in rows]


async def vector_chunks(
    session: AsyncSession,
    query_vec: npt.NDArray[np.float32],
    filters: SearchFilters,
    limit: int,
) -> list[ChunkHit]:
    # Larger candidate list for the HNSW graph walk; iterative scans (pgvector >= 0.8) keep
    # recall when selective filters prune many neighbours.
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {max(40, min(1000, limit * 2))}"))
    if filters != SearchFilters():
        await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    dist = Chunk.embedding.cosine_distance(query_vec.tolist())
    stmt = (
        select(Chunk.id, Chunk.paper_id, (1 - dist).label("score"))
        .join(Paper, Paper.id == Chunk.paper_id)
        .where(Chunk.embedding.is_not(None), *_filter_clauses(filters))
        .order_by(dist)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [ChunkHit(r[0], r[1], float(r[2])) for r in rows]


def best_chunk_per_paper(hits: Sequence[ChunkHit]) -> list[ChunkHit]:
    seen: set[int] = set()
    out: list[ChunkHit] = []
    for h in hits:
        if h.paper_id not in seen:
            seen.add(h.paper_id)
            out.append(h)
    return out


class HybridSearcher:
    def __init__(
        self, embedder: Embedder, settings: Settings, reranker: Reranker | None = None
    ) -> None:
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker

    async def _candidates(
        self,
        session: AsyncSession,
        query: str,
        mode: SearchMode,
        filters: SearchFilters,
        pool: int,
        timings: dict[str, float],
    ) -> tuple[list[ChunkHit], list[ChunkHit]]:
        kw: list[ChunkHit] = []
        vec: list[ChunkHit] = []
        if mode in ("keyword", "hybrid"):
            t0 = time.perf_counter()
            kw = await keyword_chunks(session, query, filters, pool)
            timings["keyword_ms"] = (time.perf_counter() - t0) * 1000
        if mode in ("vector", "hybrid"):
            t0 = time.perf_counter()
            qvec = await asyncio.to_thread(self.embedder.embed_query, query)
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            vec = await vector_chunks(session, qvec, filters, pool)
            timings["vector_ms"] = (time.perf_counter() - t0) * 1000
        return kw, vec

    async def search_papers(
        self,
        session: AsyncSession,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int = 10,
        rerank: bool = False,
    ) -> SearchOutcome:
        start = time.perf_counter()
        filters = filters or SearchFilters()
        timings: dict[str, float] = {}
        pool = max(self.settings.candidate_pool, limit * 3)
        kw, vec = await self._candidates(session, query, mode, filters, pool, timings)

        kw_papers = best_chunk_per_paper(kw)
        vec_papers = best_chunk_per_paper(vec)
        kw_rank = {h.paper_id: i for i, h in enumerate(kw_papers, start=1)}
        vec_rank = {h.paper_id: i for i, h in enumerate(vec_papers, start=1)}
        best_chunk = {h.paper_id: h.chunk_id for h in reversed(kw_papers)}
        best_chunk.update({h.paper_id: h.chunk_id for h in reversed(vec_papers)})

        if mode == "keyword":
            scored = [(h.paper_id, h.score) for h in kw_papers]
        elif mode == "vector":
            scored = [(h.paper_id, h.score) for h in vec_papers]
        else:
            scored = rrf_fuse(
                [[h.paper_id for h in kw_papers], [h.paper_id for h in vec_papers]],
                k=self.settings.rrf_k,
            )
        hits = [
            PaperHit(
                paper_id=pid,
                chunk_id=best_chunk[pid],
                score=score,
                keyword_rank=kw_rank.get(pid),
                vector_rank=vec_rank.get(pid),
            )
            for pid, score in scored
        ]
        if rerank and self.reranker is not None and hits:
            t0 = time.perf_counter()
            hits = await self._rerank(session, query, hits[: max(limit * 3, 30)])
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        return SearchOutcome(hits[:limit], (time.perf_counter() - start) * 1000, timings)

    async def _rerank(
        self, session: AsyncSession, query: str, hits: list[PaperHit]
    ) -> list[PaperHit]:
        assert self.reranker is not None
        rows = (
            await session.execute(
                select(Chunk.id, Chunk.title, Chunk.text).where(
                    Chunk.id.in_([h.chunk_id for h in hits])
                )
            )
        ).all()
        docs = {r.id: f"{r.title}. {r.text}" for r in rows}
        scores = await asyncio.to_thread(
            self.reranker.score, query, [docs[h.chunk_id] for h in hits]
        )
        for h, s in zip(hits, scores, strict=True):
            h.rerank_score = s
        return sorted(hits, key=lambda h: -(h.rerank_score or 0.0))

    async def search_chunks(
        self,
        session: AsyncSession,
        query: str,
        *,
        filters: SearchFilters | None = None,
        max_papers: int = 5,
        per_paper: int = 2,
    ) -> list[ChunkContext]:
        """Chunk-level hybrid retrieval for RAG: fuse chunk rankings, cap chunks per paper."""
        filters = filters or SearchFilters()
        kw, vec = await self._candidates(
            session, query, "hybrid", filters, self.settings.candidate_pool, {}
        )
        fused = rrf_fuse(
            [[h.chunk_id for h in kw], [h.chunk_id for h in vec]], k=self.settings.rrf_k
        )
        paper_of = {h.chunk_id: h.paper_id for h in (*kw, *vec)}
        chosen: list[tuple[int, float]] = []
        per: dict[int, int] = {}
        for cid, score in fused:
            pid = paper_of[cid]
            if pid not in per and len(per) >= max_papers:
                continue
            if per.get(pid, 0) >= per_paper:
                continue
            per[pid] = per.get(pid, 0) + 1
            chosen.append((cid, score))
        if not chosen:
            return []
        rows = (
            await session.execute(
                select(Chunk.id, Chunk.paper_id, Chunk.text).where(
                    Chunk.id.in_([c for c, _ in chosen])
                )
            )
        ).all()
        by_id = {r.id: r for r in rows}
        return [
            ChunkContext(cid, by_id[cid].paper_id, by_id[cid].text, score) for cid, score in chosen
        ]


async def headlines(session: AsyncSession, query: str, chunk_ids: Sequence[int]) -> dict[int, str]:
    """Return ``<mark>``-highlighted snippets for the given chunks."""
    if not chunk_ids:
        return {}
    tsq_text = to_or_tsquery(query)
    if not tsq_text:
        rows = (
            await session.execute(select(Chunk.id, Chunk.text).where(Chunk.id.in_(chunk_ids)))
        ).all()
        return {r.id: r.text for r in rows}
    tsq = func.to_tsquery("english", tsq_text)
    stmt = select(Chunk.id, func.ts_headline("english", Chunk.text, tsq, HEADLINE_OPTS)).where(
        Chunk.id.in_(chunk_ids)
    )
    return {r[0]: r[1] for r in (await session.execute(stmt)).all()}
