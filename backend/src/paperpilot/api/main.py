"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from paperpilot import __version__
from paperpilot.api.deps import get_services
from paperpilot.api.jobs import JobRunner
from paperpilot.api.routes import ask, health, ingest, papers, search, topics
from paperpilot.config import Settings, get_settings
from paperpilot.db.session import dispose_engine
from paperpilot.logs import configure_logging, get_logger

log = get_logger(__name__)

DESCRIPTION = """
**PaperPilot** — an AI research assistant over arXiv papers.

* **Search**: hybrid retrieval (Postgres full-text + pgvector, fused with Reciprocal Rank Fusion,
  optional cross-encoder rerank).
* **Ask**: grounded answers with inline `[n]` citations, streamed over Server-Sent Events.
* **Trends**: KMeans + c-TF-IDF topics and monthly publication trends.
* **Ingestion**: background jobs that pull from the arXiv API incrementally.
"""


def create_app(settings: Settings | None = None, warmup: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            recovered = await app.state.jobs.recover()
            if recovered:
                log.warning("jobs.recovered", count=recovered)
        except Exception as exc:  # DB may not be up yet; health will report it
            log.warning("jobs.recover_failed", error=str(exc))
        if warmup:
            await asyncio.to_thread(get_services)  # load the embedding model once, off-loop
        log.info("app.started", version=__version__, env=settings.environment)
        yield
        await app.state.jobs.wait_all()
        await dispose_engine()

    app = FastAPI(
        title="PaperPilot API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.jobs = JobRunner()
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = request_id
        if request.url.path != "/api/health":
            log.info(
                "http.request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
        return response

    for module in (health, search, ask, papers, topics, ingest):
        app.include_router(module.router, prefix="/api")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"name": "PaperPilot API", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
