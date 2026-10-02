"""Async client for the public arXiv API (Atom feed).

Politeness: arXiv asks clients to wait >= 3 seconds between requests and to use a single
connection. We serialise calls through a lock, enforce a minimum interval, and retry transient
failures (timeouts, 429, 5xx) with exponential backoff + jitter, honouring ``Retry-After``.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime

import httpx
from defusedxml import ElementTree

from paperpilot.ingest.schemas import RawPaper
from paperpilot.logs import get_logger

log = get_logger(__name__)

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v(?P<v>\d+))?$")


class ArxivAPIError(RuntimeError):
    pass


def parse_arxiv_id(url_or_id: str) -> tuple[str, int]:
    """Split ``http://arxiv.org/abs/2401.01234v2`` (or ``hep-th/9901001v1``) into id + version."""
    match = _ID_RE.search(url_or_id.strip())
    raw = match.group("id") if match else url_or_id.strip()
    version = int(match.group("v")) if match and match.group("v") else 1
    if not match:
        m2 = re.match(r"^(?P<id>.+?)v(?P<v>\d+)$", raw)
        if m2:
            raw, version = m2.group("id"), int(m2.group("v"))
    return raw, version


def _text(el: object | None) -> str:
    if el is None:
        return ""
    return str(getattr(el, "text", "") or "")


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_feed(xml_text: str) -> tuple[list[RawPaper], int]:
    """Parse an arXiv Atom feed into ``RawPaper`` records. Returns (papers, total_results)."""
    root = ElementTree.fromstring(xml_text)
    total = int(_text(root.find(f"{OPENSEARCH}totalResults")) or 0)
    papers: list[RawPaper] = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = _text(entry.find(f"{ATOM}id"))
        title = _text(entry.find(f"{ATOM}title"))
        if "api/errors" in entry_id or title.strip().lower() == "error":
            raise ArxivAPIError(_text(entry.find(f"{ATOM}summary")) or "arXiv API error")
        arxiv_id, version = parse_arxiv_id(entry_id)
        pdf_url = None
        abs_url = entry_id
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
            elif link.get("rel") == "alternate":
                abs_url = link.get("href") or abs_url
        primary = entry.find(f"{ARXIV}primary_category")
        categories = [c.get("term") for c in entry.findall(f"{ATOM}category") if c.get("term")]
        primary_category = (primary.get("term") if primary is not None else None) or (
            categories[0] if categories else "unknown"
        )
        papers.append(
            RawPaper(
                arxiv_id=arxiv_id,
                version=version,
                title=title,
                abstract=_text(entry.find(f"{ATOM}summary")),
                authors=[_text(a.find(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")],
                categories=[c for c in categories if c],
                primary_category=primary_category,
                published_at=_parse_dt(_text(entry.find(f"{ATOM}published"))),
                updated_at=_parse_dt(
                    _text(entry.find(f"{ATOM}updated")) or _text(entry.find(f"{ATOM}published"))
                ),
                abs_url=abs_url,
                pdf_url=pdf_url,
                comment=_text(entry.find(f"{ARXIV}comment")) or None,
                doi=_text(entry.find(f"{ARXIV}doi")) or None,
            )
        )
    return papers, total


def build_search_query(
    category: str | None = None,
    query: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> str:
    """Compose an arXiv ``search_query`` expression from simple filters."""
    parts: list[str] = []
    if category:
        parts.append(f"cat:{category}")
    if query:
        terms = " ".join(re.findall(r"[\w\-]+", query))
        parts.append(f'all:"{terms}"' if " " in terms else f"all:{terms}")
    if date_from or date_to:
        lo = (date_from or datetime(1991, 1, 1)).strftime("%Y%m%d%H%M")
        hi = (date_to or datetime(2100, 1, 1)).strftime("%Y%m%d%H%M")
        parts.append(f"submittedDate:[{lo} TO {hi}]")
    if not parts:
        raise ValueError("at least one of category/query/date range is required")
    return " AND ".join(parts)


class ArxivClient:
    def __init__(
        self,
        base_url: str = "https://export.arxiv.org/api/query",
        delay_seconds: float = 3.0,
        max_retries: int = 4,
        backoff_base: float = 5.0,
        timeout: float = 30.0,
        user_agent: str = "PaperPilot/0.1",
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.base_url = base_url
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last_request = 0.0
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": user_agent}, transport=transport
        )

    async def __aenter__(self) -> ArxivClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        wait = self.delay_seconds - (time.monotonic() - self._last_request)
        if wait > 0 and self._last_request > 0:
            await self._sleep(wait)
        self._last_request = time.monotonic()

    async def _get(self, params: dict[str, str | int]) -> str:
        attempt = 0
        async with self._lock:
            while True:
                await self._throttle()
                try:
                    resp = await self._client.get(self.base_url, params=params)
                    if resp.status_code == 200:
                        return resp.text
                    retryable = resp.status_code == 429 or resp.status_code >= 500
                    if not retryable:
                        raise ArxivAPIError(f"arXiv returned HTTP {resp.status_code}")
                    retry_after = resp.headers.get("Retry-After")
                    err: Exception = ArxivAPIError(f"arXiv returned HTTP {resp.status_code}")
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    retry_after, err = None, exc
                attempt += 1
                if attempt > self.max_retries:
                    raise ArxivAPIError(f"giving up after {attempt} attempts: {err}") from err
                backoff = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(120.0, self.backoff_base * 2 ** (attempt - 1)) + random.uniform(0, 1)  # noqa: S311 - jitter, not crypto
                )
                log.warning(
                    "arxiv.retry", attempt=attempt, backoff_s=round(backoff, 2), error=str(err)
                )
                await self._sleep(backoff)

    async def search(
        self,
        search_query: str,
        start: int = 0,
        max_results: int = 100,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> tuple[list[RawPaper], int]:
        xml_text = await self._get(
            {
                "search_query": search_query,
                "start": start,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }
        )
        return parse_feed(xml_text)

    async def iter_papers(
        self, search_query: str, max_total: int, page_size: int = 100
    ) -> AsyncIterator[list[RawPaper]]:
        """Yield pages of papers (newest first) until ``max_total`` or the feed is exhausted."""
        start = 0
        while start < max_total:
            size = min(page_size, max_total - start)
            papers, total = await self.search(search_query, start=start, max_results=size)
            log.info("arxiv.page", query=search_query, start=start, got=len(papers), total=total)
            if not papers:
                return
            yield papers
            start += len(papers)
            if start >= total:
                return
