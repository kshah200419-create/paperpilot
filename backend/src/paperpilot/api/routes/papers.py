from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from paperpilot.api.routes.serializers import summaries
from paperpilot.api.schemas import (
    CategoryCount,
    ChunkOut,
    PaperDetail,
    PaperPage,
    StatsOut,
    TopicOut,
)
from paperpilot.db.models import Chunk, Paper, PaperTopic, Topic
from paperpilot.db.session import get_session

router = APIRouter(tags=["papers"])
Session = Annotated[AsyncSession, Depends(get_session)]


@router.get("/papers", response_model=PaperPage, summary="List papers")
async def list_papers(
    session: Session,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    category: str | None = None,
    topic_id: int | None = None,
    sort: Literal["newest", "oldest"] = "newest",
) -> PaperPage:
    stmt = select(Paper)
    count = select(func.count()).select_from(Paper)
    if category:
        stmt = stmt.where(Paper.categories.contains([category]))
        count = count.where(Paper.categories.contains([category]))
    if topic_id is not None:
        stmt = stmt.join(PaperTopic).where(PaperTopic.topic_id == topic_id)
        count = count.join(PaperTopic).where(PaperTopic.topic_id == topic_id)
    order = Paper.published_at.desc() if sort == "newest" else Paper.published_at.asc()
    rows = (
        (await session.execute(stmt.order_by(order, Paper.id).limit(limit).offset(offset)))
        .scalars()
        .all()
    )
    total = (await session.execute(count)).scalar_one()
    return PaperPage(items=await summaries(session, rows), total=total, limit=limit, offset=offset)


@router.get("/papers/{arxiv_id:path}", response_model=PaperDetail, summary="Paper detail")
async def get_paper(arxiv_id: str, session: Session) -> PaperDetail:
    paper = (
        await session.execute(
            select(Paper)
            .options(selectinload(Paper.chunks), selectinload(Paper.topic_assignment))
            .where(Paper.arxiv_id == arxiv_id)
        )
    ).scalar_one_or_none()
    if paper is None:
        raise HTTPException(404, f"paper {arxiv_id} not found")
    similar: list[Paper] = []
    if paper.embedding is not None:
        similar = list(
            (
                await session.execute(
                    select(Paper)
                    .where(Paper.id != paper.id, Paper.embedding.is_not(None))
                    .order_by(Paper.embedding.cosine_distance(paper.embedding))
                    .limit(5)
                )
            ).scalars()
        )
    topic = None
    if paper.topic_assignment is not None:
        t = await session.get(Topic, paper.topic_assignment.topic_id)
        topic = TopicOut.model_validate(t) if t else None
    summary = (await summaries(session, [paper]))[0]
    return PaperDetail(
        **summary.model_dump(),
        abstract=paper.abstract,
        comment=paper.comment,
        doi=paper.doi,
        version=paper.version,
        updated_at=paper.updated_at,
        chunks=[ChunkOut.model_validate(c) for c in paper.chunks],
        topic=topic,
        similar=await summaries(session, similar),
    )


@router.get("/categories", response_model=list[CategoryCount], summary="Category counts")
async def categories(session: Session) -> list[CategoryCount]:
    cat = func.unnest(Paper.categories).label("category")
    sub = select(cat).subquery()
    rows = (
        await session.execute(
            select(sub.c.category, func.count().label("n"))
            .group_by(sub.c.category)
            .order_by(func.count().desc(), sub.c.category)
        )
    ).all()
    return [CategoryCount(category=r[0], count=r[1]) for r in rows]


@router.get("/stats", response_model=StatsOut, summary="Corpus statistics")
async def stats(session: Session) -> StatsOut:
    papers, first, last = (
        await session.execute(
            select(func.count(Paper.id), func.min(Paper.published_at), func.max(Paper.published_at))
        )
    ).one()
    chunks = (await session.execute(select(func.count(Chunk.id)))).scalar_one()
    topics = (await session.execute(select(func.count(Topic.id)))).scalar_one()
    sub = select(func.unnest(Paper.categories).label("c")).subquery()
    cats = (await session.execute(select(func.count(func.distinct(sub.c.c))))).scalar_one()
    return StatsOut(
        papers=papers,
        chunks=chunks,
        topics=topics,
        categories=cats,
        first_published=first,
        last_published=last,
    )
