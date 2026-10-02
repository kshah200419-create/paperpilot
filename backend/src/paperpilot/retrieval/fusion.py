"""Rank fusion utilities."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def rrf_fuse(
    rankings: Sequence[Sequence[K]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[K, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks. Items missing from a ranking
    contribute nothing for it. Ties are broken by first appearance to keep results deterministic.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    scores: dict[K, float] = {}
    first_seen: dict[K, int] = {}
    order = 0
    for i, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[i]
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def unique_in_order(items: Iterable[K]) -> list[K]:
    return list(dict.fromkeys(items))
