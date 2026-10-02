"""Pydantic v2 request/response models (these drive the OpenAPI schema)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PaperSummary(ORMModel):
    id: int
    arxiv_id: str
    title: str
    authors: list[str]
    primary_category: str
    categories: list[str]
    published_at: datetime
    abs_url: str
    pdf_url: str | None = None
    topic_id: int | None = None
    topic_label: str | None = None


class ChunkOut(ORMModel):
    id: int
    ordinal: int
    text: str
    n_words: int


class TopicOut(ORMModel):
    id: int
    label: str
    keywords: list[str]
    size: int
    x: float
    y: float


class PaperDetail(PaperSummary):
    abstract: str
    comment: str | None = None
    doi: str | None = None
    version: int
    updated_at: datetime
    chunks: list[ChunkOut] = Field(default_factory=list)
    topic: TopicOut | None = None
    similar: list[PaperSummary] = Field(default_factory=list)


class PaperPage(BaseModel):
    items: list[PaperSummary]
    total: int
    limit: int
    offset: int


class SearchResultOut(BaseModel):
    paper: PaperSummary
    score: float
    snippet: str = Field(description="Best-matching passage; query terms wrapped in <mark>.")
    chunk_id: int
    keyword_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: Literal["keyword", "vector", "hybrid"]
    reranked: bool
    took_ms: float
    timings: dict[str, float]
    results: list[SearchResultOut]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    max_sources: int = Field(default=5, ge=1, le=10)
    categories: list[str] = Field(default_factory=list, max_length=10)
    date_from: datetime | None = None
    date_to: datetime | None = None


class SourceOut(BaseModel):
    index: int
    paper_id: int
    arxiv_id: str
    title: str
    authors: list[str]
    published_at: datetime
    primary_category: str
    abs_url: str
    passages: list[str]
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    cited: list[int]
    provider: str
    took_ms: float
    error: str | None = None


class TopicPoint(BaseModel):
    paper_id: int
    arxiv_id: str
    title: str
    topic_id: int
    x: float
    y: float
    published_at: datetime


class TrendSeries(BaseModel):
    key: str
    label: str
    counts: list[int]
    total: int
    momentum: float = Field(description="Recent-3-month share / overall share (>1 = rising)")


class TrendsResponse(BaseModel):
    group_by: Literal["topic", "category"]
    months: list[str]
    totals: list[int]
    series: list[TrendSeries]


class CategoryCount(BaseModel):
    category: str
    count: int


class StatsOut(BaseModel):
    papers: int
    chunks: int
    topics: int
    categories: int
    first_published: datetime | None
    last_published: datetime | None


JobKind = Literal["arxiv", "sample", "topics"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class IngestJobCreate(BaseModel):
    kind: JobKind = "arxiv"
    category: str | None = Field(default=None, pattern=r"^[a-zA-Z\-]+(\.[a-zA-Z\-]+)?$")
    query: str | None = Field(default=None, max_length=200)
    max_results: int = Field(default=100, ge=1, le=2000)
    full: bool = Field(default=False, description="Ignore the incremental watermark")
    rebuild_topics: bool = True


class IngestJobOut(ORMModel):
    id: str
    kind: str
    status: JobStatus
    params: dict[str, object]
    stats: dict[str, object]
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: bool
    embedder: str | None
    llm_provider: str | None
    reranker: str | None
