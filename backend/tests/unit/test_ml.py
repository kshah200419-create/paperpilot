from __future__ import annotations

import numpy as np
import pytest

from paperpilot.config import Settings
from paperpilot.eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from paperpilot.ml.embeddings import HashingEmbedder, build_embedder
from paperpilot.ml.text import light_stem, tokenize
from paperpilot.ml.topics import ctfidf_keywords, fit_topics, make_label
from paperpilot.retrieval.fusion import rrf_fuse
from paperpilot.retrieval.search import to_or_tsquery


def test_tokenize_stems_and_drops_stopwords() -> None:
    assert tokenize("The transformers are learning graph-based embeddings") == [
        "transformer",
        "learn",
        "graph-based",
        "graph",
        "embedd",
    ]
    assert light_stem("studies") == "study"


def test_hashing_embedder_is_deterministic_and_normalised(embedder: HashingEmbedder) -> None:
    a = embedder.embed_documents(["graph neural networks for molecules"])
    b = HashingEmbedder(384).embed_documents(["graph neural networks for molecules"])
    assert a.shape == (1, 384)
    np.testing.assert_allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert embedder.embed_documents([]).shape == (0, 384)


def test_hashing_embedder_similarity_is_meaningful(embedder: HashingEmbedder) -> None:
    q = embedder.embed_query("graph neural network molecule property")
    docs = embedder.embed_documents(
        [
            "message passing graph neural networks predict molecule properties",
            "a survey of reinforcement learning for robot locomotion",
        ]
    )
    sims = docs @ q
    assert sims[0] > sims[1]


def test_build_embedder_hashing() -> None:
    e = build_embedder(Settings(embedder="hashing"))
    assert e.name == "hashing-384" and e.dim == 384


def test_rrf_fuse_matches_formula_and_is_deterministic() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60)
    scores = dict(fused)
    assert fused[0][0] == "b"
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    # a (first list rank 1) and d (second list rank 2) - a has higher score
    assert [k for k, _ in fused] == ["b", "a", "c", "d"] or [k for k, _ in fused] == [
        "b",
        "a",
        "d",
        "c",
    ]
    weighted = dict(rrf_fuse([["a"], ["b"]], weights=[1.0, 2.0]))
    assert weighted["b"] == pytest.approx(2 * weighted["a"])
    with pytest.raises(ValueError, match="weights"):
        rrf_fuse([["a"]], weights=[1, 2])


def test_rrf_ties_broken_by_first_appearance() -> None:
    assert [k for k, _ in rrf_fuse([["x", "y"], ["y", "x"]])] == ["x", "y"]


def test_to_or_tsquery_sanitises_input() -> None:
    assert to_or_tsquery("RAG: what's (new) & cool? RAG!") == "rag | what | new | cool"
    assert to_or_tsquery("!!! ' ") == ""


def test_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d", "z"}
    assert recall_at_k(ranked, rel, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, set(), 2) == 0.0
    assert reciprocal_rank(ranked, rel) == 0.5
    assert reciprocal_rank(ranked, {"q"}) == 0.0
    assert reciprocal_rank(ranked, {"d"}, k=3) == 0.0
    assert ndcg_at_k(["b", "x"], {"b"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "b"], {"b"}, 10) == pytest.approx(1 / np.log2(3))
    assert mean([]) == 0.0


def _clustered(n_per: int = 12, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    centers = np.eye(384, dtype=np.float32)[:3] * 5
    emb, docs = [], []
    words = [
        ["robot", "grasping", "manipulation"],
        ["protein", "folding", "molecule"],
        ["translation", "language", "multilingual"],
    ]
    for c in range(3):
        for _ in range(n_per):
            v = centers[c] + rng.normal(0, 0.3, 384).astype(np.float32)
            emb.append(v / np.linalg.norm(v))
            docs.append(" ".join(words[c] * 3) + " shared common text")
    return np.asarray(emb, dtype=np.float32), docs


def test_fit_topics_recovers_clusters_and_labels() -> None:
    emb, docs = _clustered()
    res = fit_topics(emb, docs, k=3)
    assert res.k == 3
    assert len(set(res.labels[:12])) == 1 and len(set(res.labels[12:24])) == 1
    assert res.coords.shape == (36, 2)
    assert res.silhouette > 0.2
    label_of_robots = res.keywords[int(res.labels[0])]
    assert {"robot", "grasping", "manipulation"} & set(" ".join(label_of_robots).split())
    assert "shared" not in label_of_robots  # c-TF-IDF suppresses cross-topic words


def test_fit_topics_auto_k_and_guards() -> None:
    emb, docs = _clustered(n_per=16)
    res = fit_topics(emb, docs)
    assert res.k >= 3
    with pytest.raises(ValueError, match="at least 4"):
        fit_topics(emb[:3], docs[:3])


def test_ctfidf_and_label() -> None:
    labels = np.array([0, 0, 1, 1])
    kw = ctfidf_keywords(["cats purr", "cats meow", "dogs bark", "dogs fetch"], labels, 2, top_n=2)
    assert kw[0][0].startswith("cats") and kw[1][0].startswith("dogs")
    assert make_label(["a", "b", "c", "d"]) == "a · b · c"
    assert make_label([]) == "misc"
