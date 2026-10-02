"""Background ingestion jobs with status persisted in Postgres.

Jobs run as asyncio tasks inside the API process (good enough for a single-node deployment;
the ``JobRunner`` interface is where a Celery/Arq/RQ worker would plug in).
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import update

from paperpilot.api.deps import get_services
from paperpilot.api.schemas import IngestJobCreate
from paperpilot.db.models import IngestJob
from paperpilot.db.session import get_sessionmaker
from paperpilot.ingest.pipeline import ingest_fixture, ingest_from_arxiv
from paperpilot.logs import get_logger
from paperpilot.ml.topic_service import rebuild_topics

log = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


class JobRunner:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    async def recover(self) -> int:
        """Mark jobs left queued/running by a previous process as failed."""
        async with get_sessionmaker()() as session, session.begin():
            res = await session.execute(
                update(IngestJob)
                .where(IngestJob.status.in_(["queued", "running"]))
                .values(status="failed", error="interrupted by server restart", finished_at=_now())
            )
        return int(getattr(res, "rowcount", 0) or 0)

    async def submit(self, req: IngestJobCreate) -> IngestJob:
        job = IngestJob(
            id=str(uuid.uuid4()),
            kind=req.kind,
            status="queued",
            params=req.model_dump(),
            stats={},
        )
        async with get_sessionmaker()() as session, session.begin():
            session.add(job)
        task = asyncio.create_task(self._run(job.id, req))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def wait_all(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _set(self, job_id: str, **values: Any) -> None:
        async with get_sessionmaker()() as session, session.begin():
            await session.execute(update(IngestJob).where(IngestJob.id == job_id).values(**values))

    async def _run(self, job_id: str, req: IngestJobCreate) -> None:
        await self._set(job_id, status="running", started_at=_now())
        log.info("job.started", job_id=job_id, kind=req.kind)
        try:
            stats = await self._dispatch(req)
            await self._set(job_id, status="succeeded", stats=stats, finished_at=_now())
            log.info(
                "job.succeeded", job_id=job_id, **{k: v for k, v in stats.items() if k != "extra"}
            )
        except Exception as exc:
            log.error("job.failed", job_id=job_id, error=str(exc), tb=traceback.format_exc())
            await self._set(job_id, status="failed", error=str(exc)[:2000], finished_at=_now())

    async def _dispatch(self, req: IngestJobCreate) -> dict[str, Any]:
        svc = get_services()
        sm = get_sessionmaker()
        stats: dict[str, Any] = {}
        if req.kind == "arxiv":
            if not (req.category or req.query):
                raise ValueError("arxiv jobs need a category or a query")
            result = await ingest_from_arxiv(
                sm,
                svc.embedder,
                svc.settings,
                category=req.category,
                query=req.query,
                max_results=req.max_results,
                full=req.full,
            )
            stats = result.as_dict()
        elif req.kind == "sample":
            stats = (await ingest_fixture(sm, svc.embedder, svc.settings)).as_dict()
        changed = bool(stats.get("inserted") or stats.get("updated"))
        if req.kind == "topics" or (req.rebuild_topics and changed):
            async with sm() as session, session.begin():
                stats["topics"] = await rebuild_topics(session)
        return stats
