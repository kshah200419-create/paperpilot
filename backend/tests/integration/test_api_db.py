from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from paperpilot.api.deps import reset_services
from paperpilot.api.main import create_app
from paperpilot.config import get_settings
from paperpilot.db.session import get_sessionmaker
from paperpilot.ml.embeddings import HashingEmbedder
from paperpilot.retrieval.search import HybridSearcher, SearchFilters

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
async def client(seeded_db: dict[str, object]) -> AsyncIterator[httpx.AsyncClient]:
    reset_services()
    app = create_app(warmup=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        c.app = app  # type: ignore[attr-defined]
        yield c
    await app.state.jobs.wait_all()


async def _first_paper(client: httpx.AsyncClient) -> dict[str, object]:
    page = (await client.get("/api/papers", params={"limit": 1})).json()
    return page["items"][0]  # type: ignore[no-any-return]


async def test_health_ok(client: httpx.AsyncClient) -> None:
    body = (await client.get("/api/health")).json()
    assert body["status"] == "ok" and body["database"] is True


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid"])
async def test_search_modes_return_ranked_results(client: httpx.AsyncClient, mode: str) -> None:
    resp = await client.get(
        "/api/search", params={"q": "large language models reasoning", "mode": mode, "limit": 5}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == mode and 0 < len(body["results"]) <= 5
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    ids = [r["paper"]["id"] for r in body["results"]]
    assert len(ids) == len(set(ids)), "one result per paper"
    if mode == "hybrid":
        assert any(r["keyword_rank"] and r["vector_rank"] for r in body["results"])
    if mode == "keyword":
        assert any("<mark>" in r["snippet"] for r in body["results"])


async def test_search_exact_title_finds_paper(client: httpx.AsyncClient) -> None:
    paper = await _first_paper(client)
    for mode in ("keyword", "hybrid"):
        body = (await client.get("/api/search", params={"q": paper["title"], "mode": mode})).json()
        assert body["results"][0]["paper"]["arxiv_id"] == paper["arxiv_id"]


async def test_search_filters(client: httpx.AsyncClient) -> None:
    body = (
        await client.get("/api/search", params={"q": "learning", "category": "cs.CV", "limit": 20})
    ).json()
    assert body["results"] and all("cs.CV" in r["paper"]["categories"] for r in body["results"])
    body = (
        await client.get(
            "/api/search", params={"q": "learning", "date_from": "2100-01-01T00:00:00Z"}
        )
    ).json()
    assert body["results"] == []


async def test_search_chunks_caps_per_paper() -> None:
    searcher = HybridSearcher(HashingEmbedder(), get_settings())
    async with get_sessionmaker()() as s:
        ctx = await searcher.search_chunks(
            s, "image segmentation", filters=SearchFilters(), max_papers=3, per_paper=1
        )
    assert 0 < len(ctx) <= 3 and len({c.paper_id for c in ctx}) == len(ctx)


async def test_ask_returns_cited_answer(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/ask",
        json={"question": "How are language models evaluated on reasoning?", "max_sources": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] == "extractive"
    assert 1 <= len(body["sources"]) <= 4
    assert body["cited"] and set(body["cited"]) <= {s["index"] for s in body["sources"]}
    for i in body["cited"]:
        assert f"[{i}]" in body["answer"]


async def test_ask_stream_emits_sse_events(client: httpx.AsyncClient) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    async with client.stream(
        "POST", "/api/ask/stream", json={"question": "robot manipulation policies"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        buf = ""
        async for chunk in resp.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                raw, buf = buf.split("\n\n", 1)
                lines = dict(line.split(": ", 1) for line in raw.splitlines())
                events.append((lines["event"], json.loads(lines["data"])))
    kinds = [e for e, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    assert kinds.count("token") > 3
    answer = "".join(str(d["text"]) for e, d in events if e == "token")
    assert "[1]" in answer


async def test_papers_list_and_detail(client: httpx.AsyncClient) -> None:
    page = (await client.get("/api/papers", params={"limit": 5, "sort": "oldest"})).json()
    assert page["total"] >= 40 and len(page["items"]) == 5
    dates = [p["published_at"] for p in page["items"]]
    assert dates == sorted(dates)
    arxiv_id = page["items"][0]["arxiv_id"]
    detail = (await client.get(f"/api/papers/{arxiv_id}")).json()
    assert detail["arxiv_id"] == arxiv_id
    assert detail["chunks"] and detail["abstract"]
    assert len(detail["similar"]) == 5 and arxiv_id not in [
        p["arxiv_id"] for p in detail["similar"]
    ]
    assert detail["topic"] is not None and detail["topic_label"] == detail["topic"]["label"]
    assert (await client.get("/api/papers/0000.00000")).status_code == 404


async def test_topics_trends_stats(client: httpx.AsyncClient) -> None:
    topics = (await client.get("/api/topics")).json()
    assert len(topics) == 5 and sum(t["size"] for t in topics) >= 40
    assert all(t["keywords"] for t in topics)
    points = (await client.get("/api/topics/map")).json()
    assert len(points) == sum(t["size"] for t in topics)
    for group in ("topic", "category"):
        tr = (await client.get("/api/trends", params={"group_by": group})).json()
        assert tr["months"] == sorted(tr["months"]) and tr["series"]
        assert all(len(s["counts"]) == len(tr["months"]) for s in tr["series"])
    stats = (await client.get("/api/stats")).json()
    assert stats["papers"] >= 40 and stats["topics"] == 5
    cats = (await client.get("/api/categories")).json()
    assert cats[0]["count"] >= cats[-1]["count"]


async def test_ingest_job_lifecycle(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/ingest/jobs", json={"kind": "topics"})
    assert resp.status_code == 202
    job_id = resp.json()["id"]
    for _ in range(100):
        job = (await client.get(f"/api/ingest/jobs/{job_id}")).json()
        if job["status"] in ("succeeded", "failed"):
            break
        await asyncio.sleep(0.1)
    assert job["status"] == "succeeded", job
    assert job["stats"]["topics"]["topics"] >= 2
    jobs = (await client.get("/api/ingest/jobs")).json()
    assert any(j["id"] == job_id for j in jobs)
    assert (await client.get("/api/ingest/jobs/nope")).status_code == 404
    bad = await client.post("/api/ingest/jobs", json={"kind": "arxiv"})
    assert bad.status_code == 422
    # restore the 5-topic model used by other assertions
    await client.app.state.jobs.wait_all()  # type: ignore[attr-defined]


async def test_admin_token_enforced(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "admin_token", "s3cret")
    assert (
        await client.post("/api/ingest/jobs", json={"kind": "topics", "rebuild_topics": False})
    ).status_code == 401
    ok = await client.post(
        "/api/ingest/jobs", headers={"X-Admin-Token": "s3cret"}, json={"kind": "topics"}
    )
    assert ok.status_code == 202
    await client.app.state.jobs.wait_all()  # type: ignore[attr-defined]
