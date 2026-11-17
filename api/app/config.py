"""Settings read from environment variables and an optional `.env` file.

Lookup order (later wins): repo-root `.env`, `api/.env`, real environment variables.
"""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment
