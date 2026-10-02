"""Cleaning, fingerprinting and de-duplication of raw paper records."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Iterable

from paperpilot.ingest.schemas import CleanPaper, RawPaper

_WS = re.compile(r"\s+")
_LATEX_CMD = re.compile(r"\\(?:emph|textit|textbf|mathrm|mathbf|texttt)\{([^{}]*)\}")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def clean_text(text: str) -> str:
    """Normalise unicode, unescape HTML entities, unwrap simple LaTeX and collapse whitespace."""
    text = unicodedata.normalize("NFKC", html.unescape(text or ""))
    text = _LATEX_CMD.sub(r"\1", text)
    text = text.replace("``", '"').replace("''", '"')
    return _WS.sub(" ", text).strip()


def title_fingerprint(title: str) -> str:
    """Case/punctuation-insensitive fingerprint used to catch cross-id duplicates."""
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    return hashlib.sha1(_NON_ALNUM.sub(" ", folded).strip().encode()).hexdigest()  # noqa: S324


def content_hash(title: str, abstract: str) -> str:
    return hashlib.sha256(f"{title}\n{abstract}".encode()).hexdigest()


def normalize_paper(raw: RawPaper) -> CleanPaper:
    title = clean_text(raw.title)
    abstract = clean_text(raw.abstract)
    authors = [clean_text(a) for a in raw.authors if clean_text(a)]
    categories = list(dict.fromkeys(c.strip() for c in raw.categories if c.strip()))
    primary = raw.primary_category.strip() or (categories[0] if categories else "unknown")
    if primary not in categories:
        categories.insert(0, primary)
    return CleanPaper(
        **raw.model_dump(exclude={"title", "abstract", "authors", "categories", "primary_category"}),
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        primary_category=primary,
        title_fingerprint=title_fingerprint(title),
        content_hash=content_hash(title, abstract),
    )


def is_valid(paper: CleanPaper, min_abstract_words: int = 20) -> bool:
    """Drop withdrawn / empty records."""
    if not paper.title or len(paper.abstract.split()) < min_abstract_words:
        return False
    return "this paper has been withdrawn" not in paper.abstract.lower()


def deduplicate(papers: Iterable[CleanPaper]) -> list[CleanPaper]:
    """Keep the latest version per arXiv id, then drop exact-title duplicates across ids.

    Order of first appearance is preserved for determinism.
    """
    by_id: dict[str, CleanPaper] = {}
    for p in papers:
        cur = by_id.get(p.arxiv_id)
        if cur is None or (p.version, p.updated_at) > (cur.version, cur.updated_at):
            by_id[p.arxiv_id] = p
    seen_titles: set[str] = set()
    out: list[CleanPaper] = []
    for p in by_id.values():
        if p.title_fingerprint in seen_titles:
            continue
        seen_titles.add(p.title_fingerprint)
        out.append(p)
    return out
