"""Tiny, dependency-free text utilities shared by the hashing embedder and the extractive LLM."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with you your yours yourself yourselves also using use used via we
    paper propose proposed show shows study approach new based novel work method methods""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def light_stem(token: str) -> str:
    """Very small suffix stripper (keeps the hashing embedder dependency-free)."""
    for suffix, min_len in (("ations", 7), ("ation", 6), ("ings", 6), ("ing", 5), ("ies", 5),
                            ("es", 5), ("s", 4), ("ed", 5)):
        if token.endswith(suffix) and len(token) >= min_len:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        parts = [tok, *tok.split("-")] if "-" in tok else [tok]
        for part in parts:
            if drop_stopwords and (part in STOPWORDS or len(part) < 2):
                continue
            tokens.append(light_stem(part) if stem and "-" not in part else part)
    return tokens
