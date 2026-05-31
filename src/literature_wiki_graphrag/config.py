from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LWGRAG_", extra="ignore")

    data_dir: Path = Field(default=Path("data"))
    raw_responses_dir: Path = Field(default=Path("data/raw"))
    output_dir: Path = Field(default=Path("data/outputs"))
    log_level: str = Field(default="INFO")

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_gemini_model: str = Field(default="gemini-2.5-flash", alias="GOOGLE_GEMINI_MODEL")
    google_deep_research_model: str = Field(
        default="deep-research-preview-04-2026",
        alias="GOOGLE_DEEP_RESEARCH_MODEL",
    )
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="google/gemini-2.0-flash-001",
        alias="OPENROUTER_MODEL",
    )
    openalex_mailto: str | None = Field(default=None, alias="OPENALEX_MAILTO")
    semantic_scholar_api_key: str | None = Field(default=None, alias="SEMANTIC_SCHOLAR_API_KEY")


@lru_cache
def get_settings() -> Settings:
    load_dotenv(override=True)
    return Settings()
