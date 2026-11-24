"""Settings read from environment variables and an optional `.env` file.

Lookup order (later wins): repo-root `.env`, `api/.env`, real environment variables.
"""

import os
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Every module that may import LangChain imports this one first (API and ingest).
# Assign, don't setdefault: a `true` already in the environment must be overridden too.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

_API_DIR = Path(__file__).resolve().parent.parent
# Latest dated GA version of the Azure OpenAI data-plane API (2026-11-24); the
# versioned route is what `AzureOpenAIEmbeddings` calls.
AZURE_OPENAI_API_VERSION = "2024-10-21"


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
    # Only the evals still use OpenAI directly (RAGAS embeddings, model comparisons).
    openai_api_key: SecretStr | None = None

    # Embeddings go through Azure OpenAI. Cloud Run and local use the key; the
    # deploy job's ingest uses a Microsoft Entra token instead. With both, the key wins.
    azure_openai_endpoint: str | None = None  # https://<resource>.openai.azure.com/
    azure_openai_deployment: str | None = None  # the deployment name, not the model
    azure_openai_api_key: SecretStr | None = None
    azure_openai_ad_token: SecretStr | None = None
    # The deployment's tokens-per-minute limit; ingest sends half of it per minute.
    azure_openai_tpm: int = Field(default=5000, ge=1000)

    origin_secret: SecretStr | None = None
    session_secret: SecretStr | None = None
    turnstile_secret: SecretStr | None = None
    turnstile_hostnames: Annotated[list[str], NoDecode] = Field(default_factory=list)
    patrol_secret: SecretStr | None = None
    patrol_limit_day: int = Field(default=10, ge=1)

    cleanup_audience: str | None = None
    cleanup_invoker_email: str | None = None

    ask_limit_token: int = Field(default=20, ge=1)
    ask_limit_ip: int = Field(default=20, ge=1)
    ask_limit_site: int = Field(default=300, ge=1)
    match_limit_token: int = Field(default=3, ge=1)
    match_limit_ip: int = Field(default=3, ge=1)
    match_limit_site: int = Field(default=30, ge=1)
    # Fetching a job ad by link (SPEC "职位对照" 输入): per day, and per minute in-process.
    fetch_limit_token: int = Field(default=10, ge=1)
    fetch_limit_ip: int = Field(default=10, ge=1)
    fetch_limit_site: int = Field(default=100, ge=1)
    fetch_limit_minute: int = Field(default=3, ge=1)

    spend_daily_usd_anthropic: Decimal | None = None
    spend_daily_usd_google: Decimal | None = None
    spend_daily_usd_openai: Decimal | None = None
    spend_monthly_usd_anthropic: Decimal | None = None
    spend_monthly_usd_google: Decimal | None = None
    spend_monthly_usd_openai: Decimal | None = None
    spend_input_tokens_estimate: int = 12000

    @field_validator(
        "anthropic_api_key",
        "gemini_api_key",
        "openai_api_key",
        "azure_openai_endpoint",
        "azure_openai_deployment",
        "azure_openai_api_key",
        "azure_openai_ad_token",
        "origin_secret",
        "session_secret",
        "turnstile_secret",
        "patrol_secret",
        "cleanup_audience",
        "cleanup_invoker_email",
        "spend_daily_usd_anthropic",
        "spend_daily_usd_google",
        "spend_daily_usd_openai",
        "spend_monthly_usd_anthropic",
        "spend_monthly_usd_google",
        "spend_monthly_usd_openai",
        mode="before",
    )
    @classmethod
    def empty_is_none(cls, value: Any) -> Any:
        return None if value == "" else value

    @field_validator("turnstile_hostnames", mode="before")
    @classmethod
    def split_hostnames(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [name.strip() for name in value.split(",") if name.strip()]
        return value

    answer_model: str = "claude-haiku-5-5"
    check_model: str = "gemini-3.5-flash-lite"
    embed_model: str = "text-embedding-3-small"
    embed_dims: int = 1536

    # Markdown pages loaded by `python -m app.ingest`.
    content_dir: Path = _API_DIR.parent / "content"


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment
