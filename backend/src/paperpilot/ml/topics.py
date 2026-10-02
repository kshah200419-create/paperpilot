"""Topic discovery: KMeans on paper embeddings + class-based TF-IDF (c-TF-IDF) labels.

c-TF-IDF (as popularised by BERTopic) treats all documents of a cluster as one "class document"
and scores term t in class c as ``tf(t,c) * log(1 + A / f(t))`` where A is the mean number of
words per class and f(t) the term's frequency across all classes. This surfaces words that are
frequent *within* a topic but not everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import silhouette_score

ACADEMIC_STOPWORDS = frozenset(
    """paper propose proposed proposes approach approaches method methods methodology result
    results show shows shown demonstrate demonstrates achieve achieves achieved performance
    state art existing novel new framework based using use used task tasks experiments
    experimental extensive significantly significant improve improves improved improvement
    outperforms outperform introduce introduces present presents work recent study studies
    evaluate evaluation code available github com https www model models datasets dataset data
    benchmark benchmarks large various different effective efficient address problem key
    furthermore however additionally moreover specifically finally including lack remains
    challenging challenge challenges limited high low better best leading learning paradigm
    tested providing provide provides enable enables allowing allows et al ii iii""".split()
)


@dataclass
class TopicModelResult:
    labels: npt.NDArray[np.int64]
    keywords: list[list[str]]
    coords: npt.NDArray[np.float32]
    centroids_2d: npt.NDArray[np.float32]
    k: int
    silhouette: float


def choose_k(x: npt.NDArray[np.float32], k_min: int = 4, k_max: int = 14, seed: int = 42) -> tuple[int, float]:
    """Pick k by cosine silhouette (embeddings are L2-normalised, so euclidean KMeans ≈ cosine)."""
    n = x.shape[0]
    k_max = max(k_min, min(k_max, n // 8))
    best_k, best_s = k_min, -1.0
    for k in range(k_min, k_max + 1):
        if k >= n:
            break
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(x)
        s = float(silhouette_score(x, labels, metric="cosine"))
        if s > best_s:
            best_k, best_s = k, s
    return best_k, best_s


def ctfidf_keywords(docs: list[str], labels: npt.NDArray[np.int64], k: int, top_n: int = 8) -> list[list[str]]:
    class_docs = [" ".join(d for d, lab in zip(docs, labels, strict=True) if lab == c) for c in range(k)]
    vectorizer = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | ACADEMIC_STOPWORDS),
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )
    counts = vectorizer.fit_transform(class_docs).toarray().astype(np.float64)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    avg_words = counts.sum() / max(k, 1)
    idf = np.log(1.0 + avg_words / np.maximum(counts.sum(axis=0), 1.0))
    scores = tf * idf
    out: list[list[str]] = []
    for c in range(k):
        picked: list[str] = []
        for idx in np.argsort(-scores[c]):
            term = str(vocab[idx])
            # skip unigrams already covered by a chosen bigram and vice-versa
            if any(term in p or p in term for p in picked):
                continue
            picked.append(term)
            if len(picked) == top_n:
                break
        out.append(picked)
    return out


def fit_topics(
    embeddings: npt.NDArray[np.float32], docs: list[str], k: int | None = None, seed: int = 42
) -> TopicModelResult:
    if embeddings.shape[0] < 4:
        raise ValueError("need at least 4 papers to build topics")
    if k is None:
        k, sil = choose_k(embeddings, seed=seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(embeddings)
    labels = km.labels_.astype(np.int64)
    sil = float(silhouette_score(embeddings, labels, metric="cosine")) if k > 1 else 0.0
    pca = PCA(n_components=2, random_state=seed).fit(embeddings)
    coords = pca.transform(embeddings).astype(np.float32)
    centroids = pca.transform(km.cluster_centers_).astype(np.float32)
    keywords = ctfidf_keywords(docs, labels, k)
    return TopicModelResult(labels, keywords, coords, centroids, k, sil)


def make_label(keywords: list[str], n: int = 3) -> str:
    return " · ".join(keywords[:n]) if keywords else "misc"
