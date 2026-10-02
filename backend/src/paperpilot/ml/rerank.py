"""Optional cross-encoder reranking (local ONNX via fastembed)."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Protocol

from paperpilot.config import Settings
from paperpilot.logs import get_logger

log = get_logger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)
        self.name = model_name
        self._lock = threading.Lock()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        with self._lock:
            return [float(s) for s in self._model.rerank(query, list(documents))]


_reranker: Reranker | None = None
_lock = threading.Lock()


def get_reranker(settings: Settings, force: bool = False) -> Reranker | None:
    """Return the configured reranker, or None if disabled/unavailable.

    ``force`` loads the fastembed reranker even when ``RERANKER=none`` (per-request opt-in).
    """
    global _reranker
    if settings.reranker == "none" and not force:
        return None
    with _lock:
        if _reranker is None:
            try:
                _reranker = FastEmbedReranker(settings.reranker_model, settings.fastembed_cache_dir)
            except Exception as exc:
                log.warning("reranker.unavailable", reason=str(exc))
                return None
        return _reranker
