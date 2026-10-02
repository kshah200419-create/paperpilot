"""Fetch a small, time-spread sample of real arXiv metadata into a JSON fixture.

Used once to build ``src/paperpilot/data/sample_papers.json`` so the demo, tests and the
evaluation harness work fully offline. arXiv metadata is distributed under CC0.

    python scripts/build_sample_dataset.py --months 12 --per-window 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from paperpilot.ingest.arxiv_client import ArxivClient, build_search_query
from paperpilot.ingest.normalize import deduplicate, is_valid, normalize_paper
from paperpilot.ingest.schemas import RawPaper

DEFAULT_CATEGORIES = ["cs.CL", "cs.LG", "cs.CV", "cs.IR", "cs.RO"]
OUT = Path(__file__).resolve().parents[1] / "src/paperpilot/data/sample_papers.json"


def month_starts(end: datetime, months: int) -> list[datetime]:
    starts = []
    y, m = end.year, end.month
    for _ in range(months):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        starts.append(datetime(y, m, 1, tzinfo=UTC))
    return sorted(starts)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--per-window", type=int, default=20)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".sample_cache"),
        help="per-window cache so an interrupted/rate-limited run can resume",
    )
    ap.add_argument("--delay", type=float, default=6.0)
    args = ap.parse_args()

    now = datetime.now(UTC)
    raws = []
    async with ArxivClient(
        delay_seconds=args.delay, user_agent="PaperPilot-sample-builder/0.1"
    ) as client:
        for start in month_starts(now, args.months):
            for day in (4, 17):  # two windows per month for a better spread
                lo = start.replace(day=day)
                hi = lo + timedelta(days=2)
                cache = args.cache_dir / f"{lo:%Y-%m-%d}.json"
                if cache.exists():
                    raws.extend(RawPaper.model_validate(r) for r in json.loads(cache.read_text()))
                    continue
                # One OR-ed query per window keeps us at ~2 requests/month (arXiv rate limits).
                cats = " OR ".join(f"cat:{c}" for c in args.categories)
                window = build_search_query(date_from=lo, date_to=hi)
                papers, _ = await client.search(
                    f"({cats}) AND {window}",
                    max_results=args.per_window,
                    sort_by="submittedDate",
                    sort_order="ascending",
                )
                print(f"{lo:%Y-%m-%d}: {len(papers)}", flush=True)
                args.cache_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps([p.model_dump(mode="json") for p in papers]))
                raws.extend(papers)

    cleaned = deduplicate(p for p in (normalize_paper(r) for r in raws) if is_valid(p))
    records = [
        r.model_dump(mode="json", exclude={"title_fingerprint", "content_hash"}) for r in cleaned
    ]
    records.sort(key=lambda r: r["published_at"])
    args.out.write_text(
        json.dumps(
            {
                "meta": {
                    "source": "arXiv API (export.arxiv.org/api/query)",
                    "license": "arXiv metadata is CC0 1.0",
                    "fetched_at": now.isoformat(timespec="seconds"),
                    "categories": args.categories,
                    "months": args.months,
                    "count": len(records),
                },
                "papers": records,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} papers -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
