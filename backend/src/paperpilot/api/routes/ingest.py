from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from paperpilot.api.deps import require_admin
from paperpilot.api.jobs import JobRunner
from paperpilot.api.schemas import IngestJobCreate, IngestJobOut
from paperpilot.db.models import IngestJob
from paperpilot.db.session import get_session

router = APIRouter(prefix="/ingest", tags=["ingestion"])
Session = Annotated[AsyncSession, Depends(get_session)]


def _runner(request: Request) -> JobRunner:
    runner: JobRunner = request.app.state.jobs
    return runner


@router.post(
    "/jobs",
    response_model=IngestJobOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_admin)],
    summary="Start a background ingestion / topic-rebuild job",
)
async def create_job(
    req: IngestJobCreate, runner: Annotated[JobRunner, Depends(_runner)]
) -> IngestJobOut:
    if req.kind == "arxiv" and not (req.category or req.query):
        raise HTTPException(422, "arxiv jobs need `category` and/or `query`")
    job = await runner.submit(req)
    return IngestJobOut.model_validate(job)


@router.get("/jobs", response_model=list[IngestJobOut], summary="Recent jobs")
async def list_jobs(
    session: Session, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> list[IngestJobOut]:
    rows = (
        (
            await session.execute(
                select(IngestJob).order_by(IngestJob.created_at.desc()).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [IngestJobOut.model_validate(j) for j in rows]


@router.get("/jobs/{job_id}", response_model=IngestJobOut, summary="Job status")
async def get_job(job_id: str, session: Session) -> IngestJobOut:
    job = await session.get(IngestJob, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return IngestJobOut.model_validate(job)
