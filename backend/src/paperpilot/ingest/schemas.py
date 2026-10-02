"""Plain data records flowing through the ingestion pipeline."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RawPaper(BaseModel):
    """A paper as parsed from an arXiv Atom entry (or a bundled fixture)."""

    arxiv_id: str = Field(description="Versionless arXiv identifier, e.g. 2401.01234")
    version: int = 1
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    primary_category: str
    published_at: datetime
    updated_at: datetime
    abs_url: str
    pdf_url: str | None = None
    comment: str | None = None
    doi: str | None = None


class CleanPaper(RawPaper):
    title_fingerprint: str
    content_hash: str


class ChunkRecord(BaseModel):
    ordinal: int
    text: str
    n_words: int
