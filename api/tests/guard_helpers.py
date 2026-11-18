"""Explicit trusted headers and a no-database admission fake for SSE tests."""

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import Settings
from app.guard.limits import SYDNEY, Admission
from app.guard.session import Session, sign_token


def guard_settings(url: str, **overrides: Any) -> Settings:
    values = {
        "database_url": url,
        "app_env": "dev",
        "origin_secret": "test-origin",
        "session_secret": "test-session",
        "turnstile_secret": "1x0000000000000000000000000000000AA",
        "turnstile_hostnames": [],
        **{
            f"spend_{period}_usd_{provider}": cap
            for period, cap in (("daily", "1"), ("monthly", "10"))
            for provider in ("anthropic", "google", "openai")
        },
    }
    return Settings(_env_file=None, **(values | overrides))


@dataclass
class Clock:
    now: datetime = datetime(2026, 1, 1, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


def guard_headers(
    settings: Settings, *, ip: str = "127.0.0.1", token: str | None = None
) -> dict[str, str]:
    assert settings.origin_secret is not None
    assert settings.session_secret is not None
    if token is None:
        token, _ = sign_token(settings.session_secret.get_secret_value(), time.time())
    return {
        "X-Origin-Secret": settings.origin_secret.get_secret_value(),
        "X-Client-IP": ip,
        "Authorization": f"Bearer {token}",
    }


class AdmittingLimits:
    def __init__(self) -> None:
        self.calls = 0

    async def admit(self, kind: str, session: Session) -> Admission:
        self.calls += 1
        return Admission(
            kind,
            session.token_key,
            hashlib.sha256(session.ip.encode()).digest(),
            datetime.now(UTC).astimezone(SYDNEY).date(),
        )
