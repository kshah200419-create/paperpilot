from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from paperpilot.api.schemas import PaperSummary
from paperpilot.db.models import Paper, PaperTopic, Topic


async def summaries(session: AsyncSession, papers: Sequence[Paper]) -> list[PaperSummary]:
    """Serialise papers, attaching their topic label in one extra query."""
    ids = [p.id for p in papers]
    topic_rows = (
        (
            await session.execute(
                select(PaperTopic.paper_id, Topic.id, Topic.label)
                .join(Topic, Topic.id == PaperTopic.topic_id)
                .where(PaperTopic.paper_id.in_(ids))
            )
        ).all()
        if ids
        else []
    )
    topics = {r[0]: (r[1], r[2]) for r in topic_rows}
    out = []
    for p in papers:
        s = PaperSummary.model_validate(p)
        if p.id in topics:
            s.topic_id, s.topic_label = topics[p.id]
        out.append(s)
    return out
