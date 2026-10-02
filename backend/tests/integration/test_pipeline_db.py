from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from paperpilot.config import get_settings
from paperpilot.db.models import Chunk, IngestState, Paper
from paperpilot.db.session import get_sessionmaker
from paperpilot.ingest.arxiv_client import ArxivClient
from paperpilot.ingest.pipeline import (
    ingest_fixture,
    ingest_from_arxiv,
    ingest_papers,
    load_fixture,
)
from paperpilot.ml.embeddings import HashingEmbedder

pytestmark = pytest.mark.db
FIX = Path(__file__).parents[1] / "fixtures"


async def test_seed_is_idempotent(seeded_db: dict[str, object]) -> None:
    first = seeded_db["ingest"]
    assert first["inserted"] > 0  # type: ignore[index]
    again = await ingest_fixture(
        get_sessionmaker(), HashingEmbedder(), get_settings(), FIX / "mini_papers.json"
    )
    assert again.inserted == 0 and again.updated == 0
    assert again.unchanged == first["inserted"]  # type: ignore[index]
    async with get_sessionmaker()() as s:
        n_papers = (await s.execute(select(func.count(Paper.id)))).scalar_one()
        n_chunks = (await s.execute(select(func.count(Chunk.id)))).scalar_one()
        no_vec = (
            await s.execute(select(func.count(Chunk.id)).where(Chunk.embedding.is_(None)))
        ).scalar_one()
    assert n_papers == first["inserted"]  # type: ignore[index]
    assert n_chunks >= n_papers and no_vec == 0


async def test_changed_abstract_is_reembedded(seeded_db: dict[str, object]) -> None:
    raw = load_fixture(FIX / "mini_papers.json")[0]
    changed = raw.model_copy(
        update={
            "abstract": raw.abstract + " We add one more sentence about evaluation.",
            "version": raw.version + 1,
        }
    )
    sm = get_sessionmaker()
    async with sm() as s, s.begin():
        stats = await ingest_papers(s, [changed], HashingEmbedder(), get_settings())
    assert stats.updated == 1
    async with sm() as s:
        paper = (await s.execute(select(Paper).where(Paper.arxiv_id == raw.arxiv_id))).scalar_one()
        assert paper.version == raw.version + 1
        assert "one more sentence" in paper.abstract
    # restore original so other tests see the fixture as-is
    async with sm() as s, s.begin():
        await ingest_papers(s, [raw], HashingEmbedder(), get_settings())


async def test_cross_id_title_duplicate_is_skipped(seeded_db: dict[str, object]) -> None:
    raw = load_fixture(FIX / "mini_papers.json")[1]
    dup = raw.model_copy(update={"arxiv_id": "9999.99999", "title": raw.title.upper()})
    async with get_sessionmaker()() as s, s.begin():
        stats = await ingest_papers(s, [dup], HashingEmbedder(), get_settings())
    assert stats.duplicates == 1 and stats.inserted == 0


def _feed(entries: list[tuple[str, str]]) -> str:
    body = "".join(
        f"""<entry><id>http://arxiv.org/abs/{aid}v1</id><title>Paper {aid} on incremental ingestion</title>
        <updated>{ts}</updated><published>{ts}</published>
        <summary>{" ".join(["Incremental ingestion keeps a watermark so reruns are cheap."] * 4)} Unique {aid}.</summary>
        <category term="cs.DB"/><arxiv:primary_category term="cs.DB"/><author><name>T</name></author></entry>"""
        for aid, ts in entries
    )
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{len(entries)}</opensearch:totalResults>{body}</feed>"
    )


async def test_incremental_arxiv_ingestion_stops_at_watermark(seeded_db: dict[str, object]) -> None:
    feeds = [
        _feed([("3001.00002", "2030-01-02T00:00:00Z"), ("3001.00001", "2030-01-01T00:00:00Z")]),
        _feed(
            [
                ("3001.00003", "2030-01-03T00:00:00Z"),
                ("3001.00002", "2030-01-02T00:00:00Z"),
                ("3001.00001", "2030-01-01T00:00:00Z"),
            ]
        ),
    ]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["search_query"])
        return httpx.Response(200, text=feeds[min(len(calls), 2) - 1])

    async def no_sleep(_: float) -> None:
        return None

    sm, s = get_sessionmaker(), get_settings()
    client = ArxivClient(transport=httpx.MockTransport(handler), sleep=no_sleep)
    first = await ingest_from_arxiv(
        sm, HashingEmbedder(), s, category="cs.DB", max_results=10, client=client
    )
    assert first.inserted == 2 and not first.stopped_at_watermark
    second = await ingest_from_arxiv(
        sm, HashingEmbedder(), s, category="cs.DB", max_results=10, client=client
    )
    assert second.inserted == 1 and second.stopped_at_watermark
    await client.aclose()
    assert calls == ["cat:cs.DB", "cat:cs.DB"]
    async with sm() as session:
        state = await session.get(IngestState, "arxiv:cat:cs.DB")
        assert state is not None and state.last_published_at == datetime(2030, 1, 3, tzinfo=UTC)
        assert state.total_ingested == 3
        # clean up so search tests see only the fixture corpus
        for p in (
            await session.execute(select(Paper).where(Paper.primary_category == "cs.DB"))
        ).scalars():
            await session.delete(p)
        await session.commit()


def test_mini_fixture_is_valid_json() -> None:
    data = json.loads((FIX / "mini_papers.json").read_text())
    assert len(data["papers"]) >= 40
