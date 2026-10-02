from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from paperpilot.api.deps import Services, get_services
from paperpilot.api.schemas import AskRequest, AskResponse
from paperpilot.db.session import get_session, get_sessionmaker
from paperpilot.logs import get_logger
from paperpilot.rag.service import answer_question, stream_answer
from paperpilot.retrieval.search import SearchFilters

router = APIRouter(tags=["ask"])
log = get_logger(__name__)


def _filters(req: AskRequest) -> SearchFilters:
    return SearchFilters(tuple(req.categories), req.date_from, req.date_to)


@router.post("/ask", response_model=AskResponse, summary="Grounded answer with citations")
async def ask(
    req: AskRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    svc: Annotated[Services, Depends(get_services)],
) -> AskResponse:
    out = await answer_question(
        session, svc.searcher, svc.llm, req.question, _filters(req), req.max_sources
    )
    return AskResponse.model_validate(out)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/ask/stream",
    summary="Streamed answer (Server-Sent Events)",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE stream with "
                     "`sources`, `token`*, then `done` (or `error`) events."}},
)
async def ask_stream(
    req: AskRequest, svc: Annotated[Services, Depends(get_services)]
) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        # The session lives inside the generator so it outlives the request handler.
        try:
            async with get_sessionmaker()() as session:
                async for ev in stream_answer(
                    session, svc.searcher, svc.llm, req.question, _filters(req), req.max_sources
                ):
                    yield _sse(ev.event, ev.data)
        except Exception as exc:  # surface failures to the client instead of a broken stream
            log.exception("ask.stream.failed")
            yield _sse("error", {"message": f"internal error: {type(exc).__name__}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
