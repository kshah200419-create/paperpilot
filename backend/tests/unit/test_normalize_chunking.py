from __future__ import annotations

from datetime import UTC, datetime

from paperpilot.ingest.chunking import chunk_text, embedding_input, split_sentences
from paperpilot.ingest.normalize import (
    clean_text,
    content_hash,
    deduplicate,
    is_valid,
    normalize_paper,
    title_fingerprint,
)
from paperpilot.ingest.schemas import RawPaper

LONG = " ".join(["word"] * 30)


def raw(
    arxiv_id: str = "2401.00001", version: int = 1, title: str = "A Title", **kw: object
) -> RawPaper:
    base = {
        "arxiv_id": arxiv_id,
        "version": version,
        "title": title,
        "abstract": f"An abstract with enough words. {LONG}",
        "authors": ["  A.  Author "],
        "categories": ["cs.LG"],
        "primary_category": "cs.CL",
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, version, tzinfo=UTC),
        "abs_url": "https://arxiv.org/abs/x",
    }
    base.update(kw)
    return RawPaper.model_validate(base)


def test_clean_text_collapses_whitespace_and_unwraps_latex() -> None:
    assert clean_text("  Deep\n   \\emph{learning} &amp; ``RAG''  ") == 'Deep learning & "RAG"'


def test_title_fingerprint_ignores_case_and_punctuation() -> None:
    assert title_fingerprint("Attention Is All You Need!") == title_fingerprint(
        "attention is all you need"
    )
    assert title_fingerprint("A") != title_fingerprint("B")


def test_normalize_paper_fills_primary_category_and_hashes() -> None:
    p = normalize_paper(raw())
    assert p.categories == ["cs.CL", "cs.LG"]
    assert p.authors == ["A. Author"]
    assert p.content_hash == content_hash(p.title, p.abstract)


def test_is_valid_rejects_withdrawn_and_short() -> None:
    assert is_valid(normalize_paper(raw()))
    assert not is_valid(normalize_paper(raw(abstract="too short")))
    assert not is_valid(normalize_paper(raw(abstract=f"This paper has been withdrawn. {LONG}")))


def test_deduplicate_keeps_latest_version_and_drops_title_dupes() -> None:
    papers = [
        normalize_paper(raw("1", 1)),
        normalize_paper(raw("1", 3)),
        normalize_paper(raw("1", 2)),
        normalize_paper(raw("2", 1, title="a title")),  # same title, different id
        normalize_paper(raw("3", 1, title="Another")),
    ]
    out = deduplicate(papers)
    assert [(p.arxiv_id, p.version) for p in out] == [("1", 3), ("3", 1)]


def test_split_sentences_respects_abbreviations_and_decimals() -> None:
    text = (
        "We use e.g. transformers et al. for this. Accuracy rose to 93.5 percent! Is it good? Yes."
    )
    assert split_sentences(text) == [
        "We use e.g. transformers et al. for this.",
        "Accuracy rose to 93.5 percent!",
        "Is it good?",
        "Yes.",
    ]


def test_chunk_text_respects_budget_and_overlap() -> None:
    sents = [f"Sentence number {i} has exactly seven words." for i in range(10)]
    chunks = chunk_text(" ".join(sents), max_words=21, overlap=1)
    assert all(c.n_words <= 21 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    # consecutive chunks share one sentence
    first_last = chunks[0].text.split(". ")[-1]
    assert first_last.rstrip(".") in chunks[1].text
    assert "Sentence number 9" in chunks[-1].text


def test_chunk_text_single_long_sentence_and_empty() -> None:
    long = " ".join(["token"] * 200) + "."
    chunks = chunk_text(long, max_words=50)
    assert len(chunks) == 1 and chunks[0].n_words == 200
    assert chunk_text("") == []


def test_embedding_input_prefixes_title() -> None:
    assert embedding_input("T", "body") == "T. body"
