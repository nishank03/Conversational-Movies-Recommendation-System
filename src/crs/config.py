"""App-wide settings (pydantic-settings). Env vars override defaults."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables in one place. Reads from env / .env with CRS_ prefix."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        env_prefix="CRS_",
        extra="ignore",
    )

    # -- paths --
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    vector_store_dir: Path = Field(default=Path("data/vector_store"))

    # dataset file names
    final_data_file: str = "final_data.jsonl"
    conversation_file: str = "Conversation.txt"
    item_map_file: str = "item_map.json"
    user_ids_file: str = "user_ids.json"

    # -- llm --
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-sonnet-4-5"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 60.0

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    elevenlabs_api_key: str | None = None

    # -- retrieval --
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    retrieval_top_k: int = 20
    rerank_top_k: int = 10

    # -- engine --
    default_engine: Literal["few_shot", "rag", "agent"] = "rag"

    # -- eval --
    eval_split_ratio: float = 0.1
    eval_seed: int = 42
    eval_k_values: tuple[int, ...] = (1, 3, 5, 10)

    # -- api --
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # -- prompt version (v1/v2/v3 for ablation runs) --
    prompt_version: Literal["v1", "v2", "v3"] = "v3"

    def raw_path(self, filename: str) -> Path:
        """Resolve a raw data file to an absolute path."""
        return (self.project_root / self.raw_data_dir / filename).resolve()

    def processed_path(self, filename: str) -> Path:
        return (self.project_root / self.processed_data_dir / filename).resolve()

    def vector_store_path(self, filename: str) -> Path:
        return (self.project_root / self.vector_store_dir / filename).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
