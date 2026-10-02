"""Offline retrieval evaluation: keyword vs vector vs hybrid (vs hybrid+rerank).

The labelled query set lives in ``paperpilot/data/eval_queries.json``. Each query lists the
arXiv ids judged relevant within the bundled sample corpus.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from paperpilot.db.models import Paper
from paperpilot.eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from paperpilot.retrieval.search import HybridSearcher, SearchMode


@dataclass
class EvalQuery:
    id: str
    query: str
    relevant: set[str]
    kind: str = "paraphrase"


def load_queries(path: Path | None = None) -> list[EvalQuery]:
    src = path or Path(str(resources.files("paperpilot.data").joinpath("eval_queries.json")))
    raw = json.loads(src.read_text(encoding="utf-8"))
    return [
        EvalQuery(q["id"], q["query"], set(q["relevant"]), q.get("kind", "paraphrase"))
        for q in raw["queries"]
    ]


async def evaluate(
    sessionmaker: async_sessionmaker[AsyncSession],
    searcher: HybridSearcher,
    queries: list[EvalQuery],
    k: int = 10,
    include_rerank: bool = False,
) -> dict[str, Any]:
    configs: list[tuple[str, SearchMode, bool]] = [
        ("keyword (Postgres FTS)", "keyword", False),
        ("vector (pgvector)", "vector", False),
        ("hybrid (RRF)", "hybrid", False),
    ]
    if include_rerank and searcher.reranker is not None:
        configs.append(("hybrid + cross-encoder rerank", "hybrid", True))

    async with sessionmaker() as session:
        id_rows = (await session.execute(select(Paper.id, Paper.arxiv_id))).all()
    arxiv_of = {r.id: r.arxiv_id for r in id_rows}
    corpus = set(arxiv_of.values())
    missing = sorted({a for q in queries for a in q.relevant} - corpus)
    if missing:
        raise ValueError(
            f"relevant ids missing from the corpus (ingest the sample first): {missing[:5]}"
        )

    results: list[dict[str, Any]] = []
    by_kind: dict[str, list[dict[str, Any]]] = {}
    per_query: dict[str, dict[str, float]] = {}
    kinds = sorted({q.kind for q in queries})
    for name, mode, rerank in configs:
        rows: list[tuple[EvalQuery, dict[str, float]]] = []
        for q in queries:
            async with sessionmaker() as session:
                out = await searcher.search_papers(
                    session, q.query, mode=mode, limit=k, rerank=rerank
                )
            ranked = [arxiv_of[h.paper_id] for h in out.hits]
            m = {
                "recall@5": recall_at_k(ranked, q.relevant, 5),
                f"recall@{k}": recall_at_k(ranked, q.relevant, k),
                f"mrr@{k}": reciprocal_rank(ranked, q.relevant, k),
                f"ndcg@{k}": ndcg_at_k(ranked, q.relevant, k),
                "latency": out.took_ms,
            }
            rows.append((q, m))
            per_query.setdefault(q.id, {})[name] = round(m[f"ndcg@{k}"], 4)
        results.append({"system": name, **_aggregate([m for _, m in rows])})
        for kind in kinds:
            by_kind.setdefault(kind, []).append(
                {"system": name, **_aggregate([m for q, m in rows if q.kind == kind])}
            )
    return {
        "k": k,
        "n_queries": len(queries),
        "n_by_kind": {kind: sum(q.kind == kind for q in queries) for kind in kinds},
        "corpus_size": len(corpus),
        "embedder": searcher.embedder.name,
        "results": results,
        "results_by_kind": by_kind,
        "per_query_ndcg": per_query,
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in rows[0] if key != "latency"] if rows else []
    out = {key: round(mean([r[key] for r in rows]), 4) for key in keys}
    out["p50_latency_ms"] = (
        round(statistics.median([r["latency"] for r in rows]), 1) if rows else 0.0
    )
    return out


def to_markdown(report: dict[str, Any], results: list[dict[str, Any]] | None = None) -> str:
    k = report["k"]
    results = results if results is not None else report["results"]
    cols = ["recall@5", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}", "p50_latency_ms"]
    head = f"| System | Recall@5 | Recall@{k} | MRR@{k} | nDCG@{k} | p50 latency (ms) |"
    lines = [head, "|---|---:|---:|---:|---:|---:|"]
    best = {c: max(r[c] for r in results) for c in cols[:-1]}
    for r in results:
        cells = [
            f"**{r[c]:.3f}**" if c in best and r[c] == best[c] else f"{r[c]:.3f}" for c in cols[:-1]
        ]
        lines.append(f"| {r['system']} | " + " | ".join(cells) + f" | {r['p50_latency_ms']:.1f} |")
    return "\n".join(lines)
