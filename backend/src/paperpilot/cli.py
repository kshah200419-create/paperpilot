"""`paperpilot` command-line interface."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer

from paperpilot.config import get_settings
from paperpilot.logs import configure_logging

app = typer.Typer(help="PaperPilot — AI research assistant over arXiv.", no_args_is_help=True)
db_app = typer.Typer(help="Database migrations.", no_args_is_help=True)
app.add_typer(db_app, name="db")

T = TypeVar("T")


def _run(coro: Coroutine[Any, Any, T]) -> T:
    async def wrapper() -> T:
        from paperpilot.db.session import dispose_engine

        try:
            return await coro
        finally:
            await dispose_engine()

    return asyncio.run(wrapper())


def _setup() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


@db_app.command("upgrade")
def db_upgrade(revision: str = "head") -> None:
    """Apply Alembic migrations."""
    from paperpilot.db import migrate

    migrate.upgrade(revision=revision)
    typer.echo(f"database upgraded to {revision}")


@db_app.command("wait")
def db_wait(timeout: float = 60.0) -> None:
    """Block until the database accepts connections (used by the Docker entrypoint)."""
    import time

    from paperpilot.db.session import ping

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _run(ping()):
            typer.echo("database is up")
            return
        time.sleep(1.0)
    typer.echo("database not reachable", err=True)
    raise typer.Exit(1)


@db_app.command("downgrade")
def db_downgrade(revision: str = "base") -> None:
    """Revert Alembic migrations."""
    from paperpilot.db import migrate

    migrate.downgrade(revision=revision)
    typer.echo(f"database downgraded to {revision}")


@app.command()
def ingest(
    category: Annotated[str | None, typer.Option(help="arXiv category, e.g. cs.LG")] = None,
    query: Annotated[str | None, typer.Option(help="Free-text arXiv query")] = None,
    max_results: Annotated[int, typer.Option("--max", help="Maximum papers to fetch")] = 100,
    since: Annotated[
        datetime | None, typer.Option(help="Override the incremental watermark")
    ] = None,
    full: Annotated[bool, typer.Option(help="Ignore the watermark (full refresh)")] = False,
    topics: Annotated[bool, typer.Option(help="Rebuild topics afterwards")] = True,
) -> None:
    """Fetch papers from the arXiv API and index them (incremental & idempotent)."""
    _setup()
    if not (category or query):
        raise typer.BadParameter("provide --category and/or --query")

    async def go() -> dict[str, Any]:
        from paperpilot.db.session import get_sessionmaker
        from paperpilot.ingest.pipeline import ingest_from_arxiv
        from paperpilot.ml.embeddings import build_embedder
        from paperpilot.ml.topic_service import rebuild_topics

        s = get_settings()
        sm = get_sessionmaker()
        stats = (
            await ingest_from_arxiv(
                sm,
                build_embedder(s),
                s,
                category=category,
                query=query,
                max_results=max_results,
                since=since,
                full=full,
            )
        ).as_dict()
        if topics and (stats["inserted"] or stats["updated"]):
            async with sm() as session, session.begin():
                stats["topics"] = await rebuild_topics(session)
        return stats

    typer.echo(json.dumps(_run(go()), indent=2, default=str))


@app.command()
def seed(
    path: Annotated[
        Path | None, typer.Option(help="JSON fixture (defaults to bundled sample)")
    ] = None,
    topics: Annotated[bool, typer.Option(help="Rebuild topics afterwards")] = True,
    k: Annotated[int | None, typer.Option(help="Number of topics (default: auto)")] = None,
    if_empty: Annotated[bool, typer.Option(help="Only seed when no papers exist yet")] = False,
) -> None:
    """Load the bundled offline sample dataset (idempotent)."""
    _setup()

    async def go() -> dict[str, Any]:
        from sqlalchemy import func, select

        from paperpilot.db.models import Paper
        from paperpilot.db.session import get_sessionmaker
        from paperpilot.ingest.pipeline import ingest_fixture
        from paperpilot.ml.embeddings import build_embedder
        from paperpilot.ml.topic_service import rebuild_topics

        s = get_settings()
        sm = get_sessionmaker()
        if if_empty:
            async with sm() as session:
                existing = (await session.execute(select(func.count(Paper.id)))).scalar_one()
            if existing:
                return {"skipped": f"database already has {existing} papers"}
        stats = (await ingest_fixture(sm, build_embedder(s), s, path)).as_dict()
        if topics:
            async with sm() as session, session.begin():
                stats["topics"] = await rebuild_topics(session, k)
        return stats

    typer.echo(json.dumps(_run(go()), indent=2, default=str))


@app.command("topics")
def topics_cmd(
    k: Annotated[int | None, typer.Option(help="Number of topics (default: auto)")] = None,
) -> None:
    """(Re)build topic clusters and labels."""
    _setup()

    async def go() -> dict[str, Any]:
        from paperpilot.db.session import get_sessionmaker
        from paperpilot.ml.topic_service import rebuild_topics

        async with get_sessionmaker()() as session, session.begin():
            return await rebuild_topics(session, k)

    typer.echo(json.dumps(_run(go()), indent=2))


@app.command("eval")
def eval_cmd(
    k: int = 10,
    rerank: Annotated[bool, typer.Option(help="Also evaluate cross-encoder reranking")] = False,
    queries: Annotated[Path | None, typer.Option(help="Labelled queries JSON")] = None,
    output: Annotated[Path | None, typer.Option(help="Write the JSON report here")] = None,
) -> None:
    """Evaluate keyword vs vector vs hybrid retrieval (Recall@k, MRR, nDCG@k)."""
    _setup()

    async def go() -> dict[str, Any]:
        from paperpilot.db.session import get_sessionmaker
        from paperpilot.eval.harness import evaluate, load_queries
        from paperpilot.ml.embeddings import build_embedder
        from paperpilot.ml.rerank import get_reranker
        from paperpilot.retrieval.search import HybridSearcher

        s = get_settings()
        reranker = get_reranker(s, force=True) if rerank else None
        searcher = HybridSearcher(build_embedder(s), s, reranker)
        return await evaluate(get_sessionmaker(), searcher, load_queries(queries), k, rerank)

    from paperpilot.eval.harness import to_markdown

    report = _run(go())
    typer.echo(
        f"\nEmbedder: {report['embedder']} · queries: {report['n_queries']} · "
        f"corpus: {report['corpus_size']} papers\n"
    )
    typer.echo(to_markdown(report))
    for kind, rows in report["results_by_kind"].items():
        typer.echo(f"\n{kind} queries (n={report['n_by_kind'][kind]}):\n")
        typer.echo(to_markdown(report, rows))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        typer.echo(f"\nreport written to {output}")


@app.command()
def ask(question: str, max_sources: int = 5) -> None:
    """Ask a question from the terminal (streams the answer)."""
    _setup()

    async def go() -> None:
        from paperpilot.db.session import get_sessionmaker
        from paperpilot.ml.embeddings import build_embedder
        from paperpilot.rag.providers import build_provider
        from paperpilot.rag.service import stream_answer
        from paperpilot.retrieval.search import HybridSearcher

        s = get_settings()
        emb = build_embedder(s)
        sources: list[dict[str, Any]] = []
        async with get_sessionmaker()() as session:
            async for ev in stream_answer(
                session,
                HybridSearcher(emb, s),
                build_provider(s, emb),
                question,
                max_papers=max_sources,
            ):
                if ev.event == "token":
                    typer.echo(ev.data["text"], nl=False)
                elif ev.event == "sources":
                    sources = ev.data["sources"]
                elif ev.event == "done":
                    typer.echo(f"\n\n[{ev.data['provider']} · {ev.data['took_ms']} ms]")
        for src in sources:
            typer.echo(f"  [{src['index']}] {src['title']} — arXiv:{src['arxiv_id']}")

    _run(go())


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:  # noqa: S104
    """Run the API server (uvicorn)."""
    import uvicorn

    uvicorn.run("paperpilot.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
