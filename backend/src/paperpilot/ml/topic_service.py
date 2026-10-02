"""Persist topic models and compute per-month trend series."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Literal

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from paperpilot.db.models import Paper, PaperTopic, Topic
from paperpilot.logs import get_logger
from paperpilot.ml.topics import fit_topics, make_label

log = get_logger(__name__)


async def rebuild_topics(session: AsyncSession, k: int | None = None) -> dict[str, Any]:
    rows = (
        await session.execute(
            select(Paper.id, Paper.title, Paper.abstract, Paper.embedding)
            .where(Paper.embedding.is_not(None))
            .order_by(Paper.id)
        )
    ).all()
    if len(rows) < 4:
        return {"topics": 0, "papers": len(rows), "skipped": "not enough papers"}
    emb = np.asarray([np.asarray(r.embedding, dtype=np.float32) for r in rows])
    docs = [f"{r.title}. {r.abstract}" for r in rows]
    result = await asyncio.to_thread(fit_topics, emb, docs, k)

    await session.execute(delete(PaperTopic))
    await session.execute(delete(Topic))
    topic_ids: list[int] = []
    for c in range(result.k):
        size = int((result.labels == c).sum())
        topic = Topic(
            label=make_label(result.keywords[c]),
            keywords=result.keywords[c],
            size=size,
            x=float(result.centroids_2d[c, 0]),
            y=float(result.centroids_2d[c, 1]),
        )
        session.add(topic)
        await session.flush()
        topic_ids.append(topic.id)
    session.add_all(
        PaperTopic(
            paper_id=r.id,
            topic_id=topic_ids[int(result.labels[i])],
            x=float(result.coords[i, 0]),
            y=float(result.coords[i, 1]),
        )
        for i, r in enumerate(rows)
    )
    await session.flush()
    stats = {"topics": result.k, "papers": len(rows), "silhouette": round(result.silhouette, 4)}
    log.info("topics.rebuilt", **stats)
    return stats


GroupBy = Literal["topic", "category"]


async def monthly_trends(
    session: AsyncSession, group_by: GroupBy = "topic", top: int = 8
) -> dict[str, Any]:
    """Return ``{months: [...], series: [{key, label, counts, total, momentum}]}``.

    ``momentum`` = share of the series in the most recent 3 months divided by its overall share
    (>1 means the topic is rising).
    """
    month = func.date_trunc("month", Paper.published_at, "UTC").label("month")
    if group_by == "topic":
        stmt = (
            select(month, Topic.id, Topic.label, func.count().label("n"))
            .select_from(Paper)
            .join(PaperTopic, PaperTopic.paper_id == Paper.id)
            .join(Topic, Topic.id == PaperTopic.topic_id)
            .group_by(month, Topic.id, Topic.label)
        )
    else:
        cat = func.unnest(Paper.categories).label("cat")
        sub = select(month, cat).subquery()
        stmt = select(sub.c.month, sub.c.cat, sub.c.cat, func.count().label("n")).group_by(
            sub.c.month, sub.c.cat
        )
    rows = (await session.execute(stmt)).all()
    months = sorted({r[0] for r in rows})
    month_idx = {m: i for i, m in enumerate(months)}
    counts: dict[Any, list[int]] = defaultdict(lambda: [0] * len(months))
    labels: dict[Any, str] = {}
    for m, key, label, n in rows:
        counts[key][month_idx[m]] += int(n)
        labels[key] = str(label)

    totals_per_month = [sum(c[i] for c in counts.values()) for i in range(len(months))]
    recent = slice(max(0, len(months) - 3), len(months))
    grand = sum(totals_per_month) or 1
    recent_total = sum(totals_per_month[recent]) or 1
    series: list[dict[str, Any]] = []
    for key, c in counts.items():
        total = sum(c)
        share_all = total / grand
        share_recent = sum(c[recent]) / recent_total
        series.append(
            {
                "key": str(key),
                "label": labels[key],
                "counts": c,
                "total": total,
                "momentum": round(share_recent / share_all, 3) if share_all else 0.0,
            }
        )
    series.sort(key=lambda s: -int(s["total"]))
    return {
        "group_by": group_by,
        "months": [_fmt_month(m) for m in months],
        "totals": totals_per_month,
        "series": series[:top],
    }


def _fmt_month(m: datetime) -> str:
    return f"{m:%Y-%m}"
