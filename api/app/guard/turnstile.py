"""Cloudflare verification, with a replaceable HTTP transport for tests."""

import httpx

from app.config import Settings
from app.guard.session import Rejected

TEST_SECRETS = frozenset(
    {
        "1x0000000000000000000000000000000AA",
        "2x0000000000000000000000000000000AA",
        "3x0000000000000000000000000000000AA",
    }
)
SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


class TurnstileVerifier:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client

    async def verify(self, token: str, ip: str) -> None:
        settings = self._settings
        if settings.session_secret is None or settings.turnstile_secret is None:
            raise Rejected("forbidden")
        if not token or len(token) > 2048:
            raise Rejected("auth")
        secret = settings.turnstile_secret.get_secret_value()
        data = {"secret": secret, "response": token, "remoteip": ip}
        try:
            if self._client is None:
                async with httpx.AsyncClient() as client:
                    response = await client.post(SITEVERIFY_URL, data=data, timeout=5.0)
            else:
                response = await self._client.post(SITEVERIFY_URL, data=data, timeout=5.0)
            if response.status_code != 200:
                raise Rejected("unavailable")
            result = response.json()
        except (httpx.RequestError, OSError, ValueError):
            raise Rejected("unavailable") from None
        if not isinstance(result, dict) or result.get("success") is not True:
            raise Rejected("auth")
        if secret not in TEST_SECRETS and (
            result.get("hostname") not in settings.turnstile_hostnames
            or result.get("action") != "ask"
        ):
            raise Rejected("auth")
