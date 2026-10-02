"""Pluggable answer generators.

* ``OpenAIProvider`` — any OpenAI-compatible Chat Completions endpoint (``OPENAI_API_KEY``).
* ``OllamaProvider`` — local models via Ollama (``OLLAMA_HOST``).
* ``ExtractiveProvider`` — keyless fallback: selects the most query-relevant sentences from the
  retrieved passages (embedding similarity + term overlap, MMR for diversity) and stitches them
  into a cited answer. Deterministic, fast, and never hallucinates beyond the sources.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from paperpilot.config import Settings
from paperpilot.ingest.chunking import split_sentences
from paperpilot.logs import get_logger
from paperpilot.ml.embeddings import Embedder
from paperpilot.ml.text import tokenize
from paperpilot.rag.prompt import Source, build_messages

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def stream(self, question: str, sources: list[Source]) -> AsyncIterator[str]:
        """Yield answer text fragments."""


class OpenAIProvider(LLMProvider):
    def __init__(
        self, api_key: str, model: str, base_url: str, timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = f"openai:{model}"
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        self._transport = transport

    async def stream(self, question: str, sources: list[Source]) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": build_messages(question, sources),
            "temperature": 0.1,
            "stream": True,
        }
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client,
                client.stream("POST", self._url, json=payload, headers=self._headers) as resp,
            ):
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise LLMError(f"OpenAI HTTP {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc


class OllamaProvider(LLMProvider):
    def __init__(
        self, host: str, model: str, timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = f"ollama:{model}"
        self._model = model
        self._url = host.rstrip("/") + "/api/chat"
        self._timeout = timeout
        self._transport = transport

    async def stream(self, question: str, sources: list[Source]) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": build_messages(question, sources),
            "stream": True,
            "options": {"temperature": 0.1},
        }
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client,
                client.stream("POST", self._url, json=payload) as resp,
            ):
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise LLMError(f"Ollama HTTP {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    content = msg.get("message", {}).get("content")
                    if content:
                        yield content
                    if msg.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc


@dataclass
class _Candidate:
    text: str
    source: int
    score: float = 0.0


class ExtractiveProvider(LLMProvider):
    name = "extractive"

    def __init__(
        self, embedder: Embedder, max_sentences: int = 4, mmr_lambda: float = 0.7,
        token_delay: float = 0.012,
    ) -> None:
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.mmr_lambda = mmr_lambda
        self.token_delay = token_delay

    def compose(self, question: str, sources: list[Source]) -> str:
        cands: list[_Candidate] = []
        seen: set[str] = set()
        for s in sources:
            for passage in s.passages:
                for sent in split_sentences(passage):
                    key = sent.lower()
                    if len(sent.split()) < 6 or key in seen:
                        continue
                    seen.add(key)
                    cands.append(_Candidate(sent, s.index))
        if not cands:
            return "I couldn't find anything in the indexed papers that answers this question."

        q_terms = set(tokenize(question))
        q_vec = self.embedder.embed_query(question)
        s_vecs = self.embedder.embed_documents([c.text for c in cands])
        sims = s_vecs @ q_vec
        for c, sim in zip(cands, sims, strict=True):
            terms = set(tokenize(c.text))
            overlap = len(q_terms & terms) / len(q_terms) if q_terms else 0.0
            rank_prior = 1.0 / (1.0 + 0.08 * (c.source - 1))
            c.score = (0.7 * float(sim) + 0.3 * overlap) * rank_prior

        # Maximal Marginal Relevance to avoid near-duplicate sentences.
        chosen: list[int] = []
        remaining = list(range(len(cands)))
        while remaining and len(chosen) < self.max_sentences:
            def mmr(i: int) -> float:
                redundancy = max((float(s_vecs[i] @ s_vecs[j]) for j in chosen), default=0.0)
                return self.mmr_lambda * cands[i].score - (1 - self.mmr_lambda) * redundancy

            best = max(remaining, key=mmr)
            chosen.append(best)
            remaining.remove(best)

        top = [cands[i] for i in chosen]
        n_papers = len({c.source for c in top})
        lead = (
            f"Here is what the {n_papers} most relevant indexed paper"
            f"{'s' if n_papers != 1 else ''} say:"
        )
        bullets = "\n".join(f"- {c.text.rstrip()} [{c.source}]" for c in top)
        return f"{lead}\n\n{bullets}"

    async def stream(self, question: str, sources: list[Source]) -> AsyncIterator[str]:
        answer = await asyncio.to_thread(self.compose, question, sources)
        for piece in re.findall(r"\S+\s*", answer):
            yield piece
            if self.token_delay:
                await asyncio.sleep(self.token_delay)


def build_provider(settings: Settings, embedder: Embedder) -> LLMProvider:
    choice = settings.llm_provider
    if choice == "openai" or (choice == "auto" and settings.openai_api_key):
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAIProvider(
            settings.openai_api_key, settings.openai_model, settings.openai_base_url,
            settings.llm_timeout_seconds,
        )
    if choice == "ollama" or (choice == "auto" and settings.ollama_host):
        if not settings.ollama_host:
            raise ValueError("LLM_PROVIDER=ollama requires OLLAMA_HOST")
        return OllamaProvider(settings.ollama_host, settings.ollama_model, settings.llm_timeout_seconds)
    return ExtractiveProvider(embedder)


_CITE = re.compile(r"\[(\d{1,2})\]")


def cited_indices(answer: str) -> list[int]:
    return sorted({int(m) for m in _CITE.findall(answer)})

