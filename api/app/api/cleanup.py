"""Cloud Scheduler cleanup, authorized by its service account's Google ID token."""

import hmac
from functools import partial

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from app.cleanup import CheckpointStore, run_cleanup
from app.guard.session import Rejected


def cleanup_router(checkpoints: CheckpointStore) -> APIRouter:
    router = APIRouter()

    @router.post("/api/cleanup")
    async def cleanup(request: Request) -> JSONResponse:
        settings = request.app.state.settings
        authorization = request.headers.get("Authorization", "").split()
        if (
            settings.cleanup_audience is None
            or settings.cleanup_invoker_email is None
            or len(authorization) != 2
            or authorization[0].lower() != "bearer"
        ):
            raise Rejected("auth")
        try:
            claims = await anyio.to_thread.run_sync(
                partial(
                    id_token.verify_oauth2_token,
                    authorization[1],
                    google_requests.Request(),
                    audience=settings.cleanup_audience,
                )
            )
        except Exception:  # noqa: BLE001 - all verification failures reject authentication
            raise Rejected("auth") from None
        if not isinstance(claims, dict):
            raise Rejected("auth")
        email = claims.get("email")
        verified = claims.get("email_verified")
        if (
            claims.get("aud") != settings.cleanup_audience
            or claims.get("iss") not in ("https://accounts.google.com", "accounts.google.com")
            or not isinstance(email, str)
            or not hmac.compare_digest(
                email.lower().encode(), settings.cleanup_invoker_email.lower().encode()
            )
            or (verified is not True and verified != "true")
        ):
            raise Rejected("auth")
        result = await run_cleanup(request.app.state.db, checkpoints)
        if result.failed:
            return JSONResponse(
                {"deleted": result.deleted, "failed": result.failed}, status_code=500
            )
        return JSONResponse({"deleted": result.deleted})

    return router
