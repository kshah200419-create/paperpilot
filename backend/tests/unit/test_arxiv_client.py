from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from paperpilot.ingest.arxiv_client import (
    ArxivAPIError,
    ArxivClient,
    build_search_query,
    parse_arxiv_id,
    parse_feed,
)

FIX = Path(__file__).parents[1] / "fixtures"
FEED = (FIX / "arxiv_feed.xml").read_text()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://arxiv.org/abs/2610.00003v2", ("2610.00003", 2)),
        ("http://arxiv.org/abs/hep-th/9901001v1", ("hep-th/9901001", 1)),
        ("2401.01234", ("2401.01234", 1)),
        ("2401.01234v7", ("2401.01234", 7)),
    ],
)
def test_parse_arxiv_id(raw: str, expected: tuple[str, int]) -> None:
    assert parse_arxiv_id(raw) == expected


def test_parse_feed_extracts_metadata() -> None:
    papers, total = parse_feed(FEED)
    assert total == 3
    assert [p.arxiv_id for p in papers] == ["2610.00003", "2610.00002", "hep-th/9901001"]
    rag = papers[0]
    assert rag.version == 2
    assert rag.primary_category == "cs.CL"
    assert rag.categories == ["cs.CL", "cs.IR"]
    assert rag.authors == ["Ada Lovelace", "Alan Turing"]
    assert rag.pdf_url == "https://arxiv.org/pdf/2610.00003v2"
    assert rag.comment == "12 pages, 4 figures"
    assert rag.published_at == datetime(2026, 9, 30, 17, tzinfo=UTC)
    assert papers[1].doi == "10.1000/example.doi"
    assert papers[2].primary_category == "hep-th"  # falls back to first category


def test_parse_feed_raises_on_api_error_entry() -> None:
    with pytest.raises(ArxivAPIError, match="incorrect id format"):
        parse_feed((FIX / "arxiv_error.xml").read_text())


def test_build_search_query() -> None:
    q = build_search_query(
        category="cs.LG",
        query="graph networks",
        date_from=datetime(2026, 1, 1),
        date_to=datetime(2026, 1, 31),
    )
    assert (
        q == 'cat:cs.LG AND all:"graph networks" AND submittedDate:[202601010000 TO 202601310000]'
    )
    assert build_search_query(query="rag") == "all:rag"
    with pytest.raises(ValueError, match="at least one"):
        build_search_query()


class Recorder:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


async def test_client_retries_on_429_and_honours_retry_after() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        if calls["n"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, text=FEED)

    sleep = Recorder()
    async with ArxivClient(
        delay_seconds=3.0, transport=httpx.MockTransport(handler), sleep=sleep
    ) as client:
        papers, _ = await client.search("cat:cs.CL")
    assert len(papers) == 3
    assert calls["n"] == 3
    assert sleep.sleeps[0] == 7.0  # Retry-After honoured
    assert all(s > 0 for s in sleep.sleeps)


async def test_client_gives_up_after_max_retries() -> None:
    sleep = Recorder()
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    async with ArxivClient(max_retries=2, transport=transport, sleep=sleep) as client:
        with pytest.raises(ArxivAPIError, match="giving up after 3 attempts"):
            await client.search("cat:cs.CL")


async def test_client_does_not_retry_client_errors() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400)

    async with ArxivClient(transport=httpx.MockTransport(handler), sleep=Recorder()) as client:
        with pytest.raises(ArxivAPIError, match="HTTP 400"):
            await client.search("cat:cs.CL")
    assert calls["n"] == 1


async def test_client_throttles_between_requests_and_paginates() -> None:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, text=FEED)

    sleep = Recorder()
    async with ArxivClient(
        delay_seconds=3.0, transport=httpx.MockTransport(handler), sleep=sleep
    ) as client:
        pages = [p async for p in client.iter_papers("cat:cs.CL", max_total=10, page_size=2)]
    # feed reports totalResults=3, so pagination stops after the first page of 3
    assert len(pages) == 1
    assert seen[0]["sortBy"] == "submittedDate"
    assert seen[0]["max_results"] == "2"

    async with ArxivClient(
        delay_seconds=3.0, transport=httpx.MockTransport(handler), sleep=sleep
    ) as client:
        await client.search("a")
        await client.search("b")
    assert any(0 < s <= 3.0 for s in sleep.sleeps), "second call must wait for the polite delay"
