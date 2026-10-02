"""Ingestion pipeline: normalise -> validate -> dedupe -> chunk -> embed -> upsert.

Idempotency
-----------
* Papers are upserted on their versionless ``arxiv_id``; unchanged content (same SHA-256 of
  title+abstract *and* same embedding model) is skipped, so re-running costs no embedding work.
* Incremental arXiv runs keep a per-query high-water mark (``ingest_state.last_published_at``)
  and stop paginating once they reach already-ingested submissions.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from paperpilot.config import Settings
from paperpilot.db.models import Chunk, IngestState, Paper
from paperpilot.ingest.arxiv_client import ArxivClient, build_search_query
from paperpilot.ingest.chunking import chunk_text, embedding_input
from paperpilot.ingest.normalize import deduplicate, is_valid, normalize_paper
from paperpilot.ingest.schemas import CleanPaper, RawPaper
from paperpilot.logs import get_logger
from paperpilot.ml.embeddings import Embedder

log = get_logger(__name__)


@dataclass
class IngestStats:
    fetched: int = 0
    invalid: int = 0
    duplicates: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    chunks: int = 0
    pages: int = 0
    stopped_at_watermark: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: IngestStats) -> None:
        for name in (
            "fetched",
            "invalid",
            "duplicates",
            "inserted",
            "updated",
            "unchanged",
            "chunks",
            "pages",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare(raws: Iterable[RawPaper]) -> tuple[list[CleanPaper], IngestStats]:
    stats = IngestStats()
    cleaned: list[CleanPaper] = []
    for raw in raws:
        stats.fetched += 1
        paper = normalize_paper(raw)
        if not is_valid(paper):
            stats.invalid += 1
            continue
        cleaned.append(paper)
    unique = deduplicate(cleaned)
    stats.duplicates += len(cleaned) - len(unique)
    return unique, stats


async def ingest_papers(
    session: AsyncSession,
    raws: Sequence[RawPaper],
    embedder: Embedder,
    settings: Settings,
) -> IngestStats:
    """Upsert a batch of papers (one transaction controlled by the caller)."""
    papers, stats = prepare(raws)
    if not papers:
        return stats

    ids = [p.arxiv_id for p in papers]
    existing_rows = (
        await session.execute(
            select(Paper.arxiv_id, Paper.content_hash, Paper.embedding_model).where(
                Paper.arxiv_id.in_(ids)
            )
        )
    ).all()
    existing = {r.arxiv_id: (r.content_hash, r.embedding_model) for r in existing_rows}

    # Cross-id duplicates (same normalised title already stored under another arXiv id).
    fp_rows = (
        await session.execute(
            select(Paper.title_fingerprint, Paper.arxiv_id).where(
                Paper.title_fingerprint.in_([p.title_fingerprint for p in papers])
            )
        )
    ).all()
    fp_owner = {r.title_fingerprint: r.arxiv_id for r in fp_rows}

    todo: list[CleanPaper] = []
    for p in papers:
        owner = fp_owner.get(p.title_fingerprint)
        if owner is not None and owner != p.arxiv_id:
            stats.duplicates += 1
            continue
        prev = existing.get(p.arxiv_id)
        if prev is not None and prev == (p.content_hash, embedder.name):
            stats.unchanged += 1
            continue
        todo.append(p)
    if not todo:
        return stats

    # Chunk + embed everything in one batch (CPU-bound -> worker thread).
    chunked = [
        chunk_text(p.abstract, settings.chunk_max_words, settings.chunk_overlap_sentences)
        for p in todo
    ]
    texts = [
        embedding_input(p.title, c.text) for p, cs in zip(todo, chunked, strict=True) for c in cs
    ]
    vectors = await asyncio.to_thread(embedder.embed_documents, texts)

    offset = 0
    for paper, chunks in zip(todo, chunked, strict=True):
        vecs = vectors[offset : offset + len(chunks)]
        offset += len(chunks)
        mean = vecs.mean(axis=0)
        paper_vec = (mean / (np.linalg.norm(mean) or 1.0)).astype(np.float32)
        values = paper.model_dump(exclude={"arxiv_id"}) | {
            "arxiv_id": paper.arxiv_id,
            "embedding": paper_vec.tolist(),
            "embedding_model": embedder.name,
        }
        insert_stmt = pg_insert(Paper).values(**values)
        upsert = insert_stmt.on_conflict_do_update(
            index_elements=[Paper.arxiv_id],
            set_={k: insert_stmt.excluded[k] for k in values if k != "arxiv_id"},
        ).returning(Paper.id)
        paper_id = (await session.execute(upsert)).scalar_one()
        await session.execute(delete(Chunk).where(Chunk.paper_id == paper_id))
        session.add_all(
            Chunk(
                paper_id=paper_id,
                ordinal=c.ordinal,
                title=paper.title,
                text=c.text,
                n_words=c.n_words,
                embedding=v.tolist(),
            )
            for c, v in zip(chunks, vecs, strict=True)
        )
        stats.chunks += len(chunks)
        if paper.arxiv_id in existing:
            stats.updated += 1
        else:
            stats.inserted += 1
    await session.flush()
    return stats


async def ingest_from_arxiv(
    sessionmaker: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    settings: Settings,
    *,
    category: str | None = None,
    query: str | None = None,
    max_results: int = 100,
    since: datetime | None = None,
    full: bool = False,
    client: ArxivClient | None = None,
) -> IngestStats:
    """Incrementally ingest the newest papers matching ``category``/``query``.

    Each page is committed separately so a crash loses at most one page; the watermark is
    advanced only at the end of a successful run.
    """
    search_query = build_search_query(category=category, query=query)
    source_key = f"arxiv:{search_query}"
    async with sessionmaker() as session:
        state = await session.get(IngestState, source_key)
    watermark = since or (None if full or state is None else state.last_published_at)

    own_client = client is None
    client = client or ArxivClient(
        base_url=settings.arxiv_api_url,
        delay_seconds=settings.arxiv_delay_seconds,
        max_retries=settings.arxiv_max_retries,
        user_agent=settings.arxiv_user_agent,
    )
    total = IngestStats(
        extra={"source_key": source_key, "watermark": watermark.isoformat() if watermark else None}
    )
    newest: datetime | None = state.last_published_at if state else None
    try:
        async for page in client.iter_papers(search_query, max_results, settings.arxiv_page_size):
            fresh = [p for p in page if watermark is None or p.published_at > watermark]
            if fresh:
                async with sessionmaker() as session, session.begin():
                    page_stats = await ingest_papers(session, fresh, embedder, settings)
                page_stats.pages = 1
                total.merge(page_stats)
                page_max = max(p.published_at for p in fresh)
                newest = page_max if newest is None else max(newest, page_max)
            if len(fresh) < len(page):
                total.stopped_at_watermark = True
                break
    finally:
        if own_client:
            await client.aclose()

    async with sessionmaker() as session, session.begin():
        stmt = pg_insert(IngestState).values(
            source_key=source_key,
            last_published_at=newest,
            total_ingested=total.inserted,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[IngestState.source_key],
            set_={
                "last_published_at": stmt.excluded.last_published_at,
                "total_ingested": IngestState.total_ingested + stmt.excluded.total_ingested,
                "updated_at": datetime.now().astimezone(),
            },
        )
        await session.execute(stmt)
    log.info("ingest.arxiv.done", **total.as_dict())
    return total


def sample_dataset_path() -> Path:
    return Path(str(resources.files("paperpilot.data").joinpath("sample_papers.json")))


def load_fixture(path: Path | None = None) -> list[RawPaper]:
    data = json.loads((path or sample_dataset_path()).read_text(encoding="utf-8"))
    records = data["papers"] if isinstance(data, dict) else data
    return [RawPaper.model_validate(r) for r in records]


async def ingest_fixture(
    sessionmaker: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    settings: Settings,
    path: Path | None = None,
    batch_size: int = 64,
) -> IngestStats:
    raws = load_fixture(path)
    total = IngestStats(extra={"source": str(path or "bundled sample")})
    for start in range(0, len(raws), batch_size):
        async with sessionmaker() as session, session.begin():
            batch_stats = await ingest_papers(
                session, raws[start : start + batch_size], embedder, settings
            )
        total.merge(batch_stats)
    log.info("ingest.fixture.done", **total.as_dict())
    return total
