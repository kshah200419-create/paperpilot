"""Application configuration, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

EmbedderName = Literal["auto", "fastembed", "hashing"]
LLMProviderName = Literal["auto", "openai", "ollama", "extractive"]
RerankerName = Literal["none", "fastembed"]


class Settings(BaseSettings):
    """Typed settings. Every field can be overridden with an env var of the same name."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- core -----------------------------------------------------------------------------
    app_name: str = "PaperPilot"
    environment: Literal["dev", "test", "prod"] = "dev"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/paperpilot"
    db_pool_size: int = 5
    db_nullpool: bool = Field(
        default=False, description="Disable connection pooling (useful for tests / serverless)."
    )
    log_level: str = "INFO"
    log_json: bool = False
    # NoDecode: accept a plain comma-separated env value instead of requiring JSON.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:8080"]
    )
    admin_token: str | None = Field(
        default=None, description="If set, ingestion endpoints require the X-Admin-Token header."
    )

    # --- embeddings / retrieval -----------------------------------------------------------
    embedder: EmbedderName = "auto"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    fastembed_cache_dir: str | None = None
    reranker: RerankerName = "none"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60
    candidate_pool: int = Field(default=50, description="Candidates per retriever before fusion.")

    # --- LLM providers ----------------------------------------------------------------------
    llm_provider: LLMProviderName = "auto"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    ollama_host: str | None = None
    ollama_model: str = "llama3.2"
    llm_timeout_seconds: float = 60.0

    # --- ingestion ------------------------------------------------------------------------
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    arxiv_delay_seconds: float = Field(default=3.0, description="arXiv asks for >=3s between calls")
    arxiv_max_retries: int = 4
    arxiv_page_size: int = 100
    arxiv_user_agent: str = "PaperPilot/0.1 (research-assistant demo; +https://github.com/kshah200419-create/paperpilot)"
    chunk_max_words: int = 90
    chunk_overlap_sentences: int = 1

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            if value.strip().startswith("["):
                return json.loads(value)
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @property
    def sync_database_url(self) -> str:
        """URL usable by sync drivers (psycopg) — Alembic uses the async URL directly."""
        return self.database_url.replace("+asyncpg", "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
