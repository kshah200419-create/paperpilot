"""Embedding backends.

* ``FastEmbedEmbedder`` — local ONNX model (default ``BAAI/bge-small-en-v1.5``, 384-d), no API key.
* ``HashingEmbedder`` — deterministic signed feature hashing over stemmed uni/bi-grams. Zero
  downloads, used in tests/CI and as an automatic fallback when the model is unavailable.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from paperpilot.config import Settings
from paperpilot.logs import get_logger
from paperpilot.ml.text import tokenize

log = get_logger(__name__)

Matrix = npt.NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> npt.NDArray[np.float32]: ...


def _l2_normalize(m: Matrix) -> Matrix:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-ngrams embedder (the "hashing trick").

    Uses blake2b rather than ``hash()`` so vectors are stable across processes and machines.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> Counter[str]:
        toks = tokenize(text)
        feats: Counter[str] = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False))
        return feats

    def _vector(self, text: str) -> npt.NDArray[np.float32]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, tf in self._features(text).items():
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            weight = (1.0 + math.log(tf)) * (0.6 if "_" in feat else 1.0)
            vec[h % self.dim] += weight if (h >> 63) & 1 else -weight
        return vec

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    def __init__(
        self, model_name: str, dim: int, cache_dir: str | None = None, batch_size: int = 64
    ) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._lock = threading.Lock()  # ONNX sessions are thread-safe, but keep batches serial
        probe = self.embed_query("probe")
        if probe.shape[0] != dim:
            raise ValueError(f"{model_name} produces {probe.shape[0]}-d vectors, expected {dim}")

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            vecs = list(self._model.passage_embed(list(texts), batch_size=self.batch_size))
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        with self._lock:
            vec = next(iter(self._model.query_embed([text])))
        return _l2_normalize(np.asarray([vec], dtype=np.float32))[0]


_cache: dict[tuple[str, str, int], Embedder] = {}
_cache_lock = threading.Lock()


def build_embedder(settings: Settings) -> Embedder:
    """Return a (cached) embedder according to ``settings.embedder``.

    ``auto`` tries fastembed and silently falls back to hashing (logged) if the package or the
    model weights are unavailable, so the app always starts.
    """
    key = (settings.embedder, settings.embedding_model, settings.embedding_dim)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
        embedder: Embedder
        if settings.embedder == "hashing":
            embedder = HashingEmbedder(settings.embedding_dim)
        else:
            try:
                embedder = FastEmbedEmbedder(
                    settings.embedding_model, settings.embedding_dim, settings.fastembed_cache_dir
                )
            except Exception as exc:
                if settings.embedder == "fastembed":
                    raise
                log.warning("embedder.fallback", reason=str(exc), fallback="hashing")
                embedder = HashingEmbedder(settings.embedding_dim)
        log.info("embedder.ready", name=embedder.name, dim=embedder.dim)
        _cache[key] = embedder
        return embedder
