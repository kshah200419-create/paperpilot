from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from paperpilot.config import Settings
from paperpilot.ml.embeddings import HashingEmbedder
from paperpilot.rag.prompt import Source, build_messages, format_sources
from paperpilot.rag.providers import (
    ExtractiveProvider,
    LLMError,
    OllamaProvider,
    OpenAIProvider,
    build_provider,
    cited_indices,
)


def sources() -> list[Source]:
    when = datetime(2026, 5, 1, tzinfo=UTC)
    return [
        Source(
            1,
            10,
            "2605.00001",
            "Grounded RAG",
            ["A"],
            when,
            "cs.CL",
            "u",
            [
                "Retrieval-augmented generation grounds answers in retrieved passages. "
                "Citation grounding reduces hallucination in long-form answers."
            ],
        ),
        Source(
            2,
            11,
            "2605.00002",
            "Robot grasping",
            ["B"],
            when,
            "cs.RO",
            "u",
            ["We train a robot arm to grasp unseen household objects with tactile sensing."],
        ),
    ]


async def collect(provider: object, q: str = "how does retrieval reduce hallucination?") -> str:
    return "".join([p async for p in provider.stream(q, sources())])  # type: ignore[attr-defined]


def test_prompt_contains_numbered_sources() -> None:
    text = format_sources(sources())
    assert text.startswith("[1] Grounded RAG (2026-05, arXiv:2605.00001)")
    msgs = build_messages("q?", sources())
    assert msgs[0]["role"] == "system" and "cite" in msgs[0]["content"].lower()
    assert "Question: q?" in msgs[1]["content"]


async def test_extractive_provider_cites_most_relevant_source() -> None:
    answer = await collect(ExtractiveProvider(HashingEmbedder(), token_delay=0))
    assert "[1]" in answer
    first_bullet = answer.split("\n- ")[1]
    assert "hallucination" in first_bullet.lower() or "retrieval" in first_bullet.lower()
    assert first_bullet.rstrip().endswith("[1]")


async def test_extractive_provider_without_passages() -> None:
    p = ExtractiveProvider(HashingEmbedder(), token_delay=0)
    s = sources()[:1]
    s[0].passages = ["tiny."]
    out = "".join([x async for x in p.stream("q", s)])
    assert "couldn't find" in out


def test_cited_indices() -> None:
    assert cited_indices("A [2] b [1][2] c [10].") == [1, 2, 10]


async def test_openai_provider_parses_sse_stream() -> None:
    body = (
        "".join(
            f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}\n\n"
            for t in ["RAG ", "works [1]."]
        )
        + "data: [DONE]\n\n"
    )
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers["authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    p = OpenAIProvider(
        "sk-test", "gpt-x", "https://api.example.com/v1", transport=httpx.MockTransport(handler)
    )
    assert await collect(p) == "RAG works [1]."
    assert captured["auth"] == "Bearer sk-test"
    assert captured["payload"]["stream"] is True  # type: ignore[index]


async def test_openai_provider_raises_llm_error_on_http_error() -> None:
    p = OpenAIProvider(
        "k",
        "m",
        "https://x/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")),
    )
    with pytest.raises(LLMError, match="401"):
        await collect(p)


async def test_ollama_provider_parses_ndjson() -> None:
    lines = [{"message": {"content": "Hello "}}, {"message": {"content": "[2]"}}, {"done": True}]
    body = "\n".join(json.dumps(x) for x in lines)
    p = OllamaProvider(
        "http://ollama:11434",
        "llama3.2",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body)),
    )
    assert await collect(p) == "Hello [2]"


def test_build_provider_selection() -> None:
    e = HashingEmbedder()
    assert build_provider(Settings(llm_provider="auto"), e).name == "extractive"
    assert build_provider(Settings(llm_provider="auto", openai_api_key="k"), e).name.startswith(
        "openai:"
    )
    assert build_provider(
        Settings(llm_provider="auto", ollama_host="http://h:1"), e
    ).name.startswith("ollama:")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_provider(Settings(llm_provider="openai", openai_api_key=None), e)
