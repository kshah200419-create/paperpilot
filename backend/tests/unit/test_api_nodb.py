from __future__ import annotations

import httpx
import pytest

from paperpilot.api.main import create_app
from paperpilot.config import Settings


@pytest.fixture
async def client() -> httpx.AsyncClient:
    app = create_app(warmup=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_openapi_lists_all_endpoints(client: httpx.AsyncClient) -> None:
    spec = (await client.get("/openapi.json")).json()
    paths = set(spec["paths"])
    for p in [
        "/api/health",
        "/api/search",
        "/api/ask",
        "/api/ask/stream",
        "/api/papers",
        "/api/papers/{arxiv_id}",
        "/api/topics",
        "/api/topics/map",
        "/api/trends",
        "/api/ingest/jobs",
        "/api/ingest/jobs/{job_id}",
        "/api/stats",
        "/api/categories",
    ]:
        assert p in paths, p


async def test_health_degraded_when_db_down(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def down() -> bool:
        return False

    monkeypatch.setattr("paperpilot.api.routes.health.ping", down)
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
    assert resp.headers["x-request-id"]


async def test_request_id_is_propagated(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def up() -> bool:
        return True

    monkeypatch.setattr("paperpilot.api.routes.health.ping", up)
    resp = await client.get("/api/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers["x-request-id"] == "abc123"


async def test_validation_errors(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/search")).status_code == 422
    assert (await client.post("/api/ask", json={"question": "a"})).status_code == 422
    bad = await client.post("/api/ingest/jobs", json={"kind": "arxiv", "category": "cs.LG; DROP"})
    assert bad.status_code == 422


def test_settings_parse_cors_list() -> None:
    s = Settings(cors_origins="http://a.com, http://b.com")  # type: ignore[arg-type]
    assert s.cors_origins == ["http://a.com", "http://b.com"]
    assert (
        Settings(database_url="postgresql+asyncpg://u@h/d").sync_database_url
        == "postgresql://u@h/d"
    )
