from __future__ import annotations

from fastapi import APIRouter

from paperpilot import __version__
from paperpilot.api.deps import peek_services
from paperpilot.api.schemas import HealthOut
from paperpilot.db.session import ping

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Liveness/readiness probe")
async def health() -> HealthOut:
    db_ok = await ping()
    svc = peek_services()
    return HealthOut(
        status="ok" if db_ok else "degraded",
        version=__version__,
        database=db_ok,
        embedder=svc.embedder.name if svc else None,
        llm_provider=svc.llm.name if svc else None,
        reranker=svc.reranker.name if svc and svc.reranker else None,
    )
