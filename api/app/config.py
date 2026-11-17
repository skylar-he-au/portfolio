"""Settings read from environment variables and an optional `.env` file.

Lookup order (later wins): repo-root `.env`, `api/.env`, real environment variables.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Every module that may import LangChain imports this one first (API and ingest).
# Assign, don't setdefault: a `true` already in the environment must be overridden too.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

_API_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_API_DIR.parent / ".env", _API_DIR / ".env"),
        env_file_encoding="utf-8",
        # The shared `.env` also holds variables for the web app and compose.
        extra="ignore",
    )

    # SecretStr keeps the password out of reprs, tracebacks and logs.
    database_url: SecretStr
    # Defaults to "prod" so a missing variable never exposes dev-only routes.
    app_env: Literal["dev", "prod"] = "prod"

    # Model keys. Optional so tests run without them (they use fake models).
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    answer_model: str = "claude-haiku-5-5"
    check_model: str = "gemini-2.5-flash-lite"
    embed_model: str = "text-embedding-3-small"
    embed_dims: int = 1536

    # Markdown pages loaded by `python -m app.ingest`.
    content_dir: Path = _API_DIR.parent / "content"


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment
