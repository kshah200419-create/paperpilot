"""Shared fixtures.

Unit tests never touch the network, a model download, or Postgres. Tests marked ``db`` run
against ``TEST_DATABASE_URL`` (default: local ``paperpilot_test``) and are skipped automatically
when that database is unreachable.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

TEST_DB = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/paperpilot_test"
)
os.environ.update(
    {
        "DATABASE_URL": TEST_DB,
        "DB_NULLPOOL": "true",
        "EMBEDDER": "hashing",
        "LLM_PROVIDER": "extractive",
        "RERANKER": "none",
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "WARNING",
        "ADMIN_TOKEN": "",
    }
)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("OLLAMA_HOST", None)

from paperpilot.config import Settings, get_settings  # noqa: E402
from paperpilot.ml.embeddings import HashingEmbedder  # noqa: E402

get_settings.cache_clear()
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(384)


def _db_reachable(url: str) -> bool:
    import asyncpg

    dsn = url.replace("postgresql+asyncpg://", "postgresql://")

    async def probe() -> bool:
        try:
            conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=3)
        except Exception:
            return False
        await conn.close()
        return True

    return asyncio.run(probe())


_DB_OK: bool | None = None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    global _DB_OK
    if not any("db" in item.keywords for item in items):
        return
    _DB_OK = _db_reachable(TEST_DB)
    if _DB_OK:
        return
    skip = pytest.mark.skip(reason=f"Postgres not reachable at {TEST_DB}")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def migrated_db() -> str:
    from paperpilot.db import migrate

    migrate.downgrade(TEST_DB, "base")
    migrate.upgrade(TEST_DB, "head")
    return TEST_DB


@pytest.fixture(scope="session")
async def seeded_db(migrated_db: str) -> dict[str, object]:
    from paperpilot.db.session import get_sessionmaker
    from paperpilot.ingest.pipeline import ingest_fixture
    from paperpilot.ml.topic_service import rebuild_topics

    s = get_settings()
    sm = get_sessionmaker()
    stats = await ingest_fixture(sm, HashingEmbedder(384), s, FIXTURES / "mini_papers.json")
    async with sm() as session, session.begin():
        topics = await rebuild_topics(session, k=5)
    return {"ingest": stats.as_dict(), "topics": topics}
