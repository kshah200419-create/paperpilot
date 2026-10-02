"""Sentence-aware chunking with overlap.

Abstracts are short (~150-300 words), so we split into sentence windows of at most
``max_words`` words, overlapping by ``overlap`` sentences. Each chunk is later embedded together
with the paper title ("contextual chunk embedding") so short passages keep their topic signal.
"""

from __future__ import annotations

import re

from paperpilot.ingest.schemas import ChunkRecord

_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "etc.", "vs.", "Fig.", "fig.", "Eq.", "eq.", "cf.")
_PLACEHOLDER = "\u2063"  # invisible separator used to protect abbreviation dots
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])")


def split_sentences(text: str) -> list[str]:
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _PLACEHOLDER))
    # protect decimals like 3.5 and version numbers
    protected = re.sub(r"(?<=\d)\.(?=\d)", _PLACEHOLDER, protected)
    parts = _SENT_SPLIT.split(protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


def chunk_text(text: str, max_words: int = 90, overlap: int = 1) -> list[ChunkRecord]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: list[ChunkRecord] = []
    i = 0
    while i < len(sentences):
        window: list[str] = []
        words = 0
        j = i
        while j < len(sentences):
            n = len(sentences[j].split())
            if window and words + n > max_words:
                break
            window.append(sentences[j])
            words += n
            j += 1
        body = " ".join(window)
        chunks.append(ChunkRecord(ordinal=len(chunks), text=body, n_words=len(body.split())))
        if j >= len(sentences):
            break
        i = max(i + 1, j - overlap)
    return chunks


def embedding_input(title: str, chunk: str) -> str:
    return f"{title}. {chunk}"
