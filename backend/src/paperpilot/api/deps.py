"""Lazily-initialised shared services (embedder, searcher, LLM provider)."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from paperpilot.config import Settings, get_settings
from paperpilot.ml.embeddings import Embedder, build_embedder
from paperpilot.ml.rerank import Reranker, get_reranker
from paperpilot.rag.providers import LLMProvider, build_provider
from paperpilot.retrieval.search import HybridSearcher


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    reranker: Reranker | None
    searcher: HybridSearcher
    llm: LLMProvider


_services: Services | None = None


def get_services() -> Services:
    global _services
    if _services is None:
        settings = get_settings()
        embedder = build_embedder(settings)
        reranker = get_reranker(settings)
        _services = Services(
            settings=settings,
            embedder=embedder,
            reranker=reranker,
            searcher=HybridSearcher(embedder, settings, reranker),
            llm=build_provider(settings, embedder),
        )
    return _services


def reset_services() -> None:
    global _services
    _services = None


def peek_services() -> Services | None:
    return _services


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    token = get_settings().admin_token
    if token and x_admin_token != token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-Admin-Token")
