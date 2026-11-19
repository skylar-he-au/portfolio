"""Exchange a Turnstile response for a stateless visitor session."""

import hmac
import re
import time
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from app.guard.session import Rejected, raw_client_ip, require_origin, sign_token


def _require_patrol(header: str, secret: str | None, now: float) -> None:
    if secret is None or re.fullmatch(r"[0-9]{1,20}\.[0-9a-f]{64}", header) is None:
        raise Rejected("auth")
    timestamp, signature = header.split(".")
    expected = hmac.digest(secret.encode(), timestamp.encode("ascii"), "sha256").hex()
    if not hmac.compare_digest(signature, expected) or abs(now - int(timestamp)) > 300:
        raise Rejected("auth")


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turnstile: str


def session_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/session")
    async def issue_session(
        request: Request,
        ip: Annotated[str, Depends(require_origin)],
        body: SessionRequest,
    ) -> JSONResponse:
        settings = request.app.state.settings
        if settings.session_secret is None or settings.turnstile_secret is None:
            raise Rejected("forbidden")
        header = request.headers.get("X-Patrol")
        patrol = header is not None
        now = time.time()
        if header is not None:
            _require_patrol(
                header,
                settings.patrol_secret.get_secret_value() if settings.patrol_secret else None,
                now,
            )
        if not request.app.state.limits.allow_new_session(ip):
            raise Rejected("slow_down")
        if not patrol:
            await request.app.state.turnstile_verifier.verify(
                body.turnstile, raw_client_ip(request)
            )
        secret = settings.session_secret.get_secret_value()
        token, exp = sign_token(secret, time.time(), patrol=patrol)
        expires_at = datetime.fromtimestamp(exp, UTC).isoformat().replace("+00:00", "Z")
        return JSONResponse(
            {"token": token, "expires_at": expires_at}, headers={"Cache-Control": "no-store"}
        )

    return router
