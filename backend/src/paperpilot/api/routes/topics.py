from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from paperpilot.api.schemas import TopicOut, TopicPoint, TrendsResponse
from paperpilot.db.models import Paper, PaperTopic, Topic
from paperpilot.db.session import get_session
from paperpilot.ml.topic_service import monthly_trends

router = APIRouter(tags=["topics"])
Session = Annotated[AsyncSession, Depends(get_session)]


@router.get("/topics", response_model=list[TopicOut], summary="Discovered topics")
async def topics(session: Session) -> list[TopicOut]:
    rows = (await session.execute(select(Topic).order_by(Topic.size.desc()))).scalars().all()
    return [TopicOut.model_validate(t) for t in rows]


@router.get("/topics/map", response_model=list[TopicPoint], summary="2-D topic map (PCA)")
async def topic_map(session: Session) -> list[TopicPoint]:
    rows = (
        await session.execute(
            select(
                Paper.id,
                Paper.arxiv_id,
                Paper.title,
                Paper.published_at,
                PaperTopic.topic_id,
                PaperTopic.x,
                PaperTopic.y,
            ).join(PaperTopic, PaperTopic.paper_id == Paper.id)
        )
    ).all()
    return [
        TopicPoint(
            paper_id=r[0],
            arxiv_id=r[1],
            title=r[2],
            published_at=r[3],
            topic_id=r[4],
            x=r[5],
            y=r[6],
        )
        for r in rows
    ]


@router.get("/trends", response_model=TrendsResponse, summary="Monthly publication trends")
async def trends(
    session: Session,
    group_by: Literal["topic", "category"] = "topic",
    top: Annotated[int, Query(ge=1, le=30)] = 8,
) -> TrendsResponse:
    return TrendsResponse.model_validate(await monthly_trends(session, group_by, top))
