"""Exchange a Turnstile response for a stateless visitor session."""

import time
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from app.guard.session import Rejected, require_origin, sign_token


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
        if not request.app.state.limits.allow_new_session(ip):
            raise Rejected("slow_down")
        await request.app.state.turnstile_verifier.verify(body.turnstile, ip)
        secret = settings.session_secret.get_secret_value()
        token, exp = sign_token(secret, time.time())
        expires_at = datetime.fromtimestamp(exp, UTC).isoformat().replace("+00:00", "Z")
        return JSONResponse(
            {"token": token, "expires_at": expires_at}, headers={"Cache-Control": "no-store"}
        )

    return router
