"""FastAPI app. Run with `uvicorn app.main:app --no-access-log`."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.api.ask import AskService, ask_router
from app.api.cleanup import cleanup_router
from app.api.feedback import feedback_router
from app.api.match import MatchService, match_router
from app.api.session import session_router
from app.config import Settings, get_settings
from app.db import Database, make_pool
from app.guard.ledger import Ledger, SpendLimits
from app.guard.limits import Limits, PauseSource
from app.guard.session import Rejected, require_origin
from app.guard.turnstile import TurnstileVerifier
from app.logging_setup import configure_logging
from app.qa.memory import Checkpoints, make_checkpoint_pool
from app.qa.models import MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)
# Probes and the scheduler skip the Worker secret; cleanup verifies Google OIDC instead.
ORIGIN_EXEMPT_PATHS = frozenset({"/api/ping", "/api/health", "/api/cleanup"})
REQUIRED_SECRETS = ("origin_secret", "session_secret", "turnstile_secret")
REQUIRED_PRODUCTION = (
    "turnstile_hostnames",
    "anthropic_api_key",
    "gemini_api_key",
    "azure_openai_api_key",
    "azure_openai_endpoint",
    "azure_openai_deployment",
    "patrol_secret",
)
REQUIRED_CAPS = tuple(
    f"spend_{period}_usd_{provider}"
    for period in ("daily", "monthly")
    for provider in ("anthropic", "google", "openai")
)


def create_app(
    settings: Settings | None = None,
    pool: Any | None = None,
    *,
    checkpoint_pool: Any | None = None,
    cleanup_checkpoints: Any | None = None,
    ask_service: AskService | None = None,
    match_service: MatchService | None = None,
    limits: Limits | None = None,
    ledger: Ledger | None = None,
    pause_source: PauseSource | None = None,
    turnstile_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Build the app. Tests can inject pools or services; all default pools start closed."""
    configure_logging()
    settings = settings or get_settings()
    if settings.app_env != "dev":
        missing = [
            name.upper()
            for name in (*REQUIRED_SECRETS, *REQUIRED_PRODUCTION, *REQUIRED_CAPS)
            if getattr(settings, name) is None
            or (name == "turnstile_hostnames" and not settings.turnstile_hostnames)
        ]
        if missing:
            raise RuntimeError(f"Missing required variables: {', '.join(missing)}")
    database_url = settings.database_url.get_secret_value()
    db = Database(pool if pool is not None else make_pool(database_url))
    cleanup_checkpoints = (
        cleanup_checkpoints
        if cleanup_checkpoints is not None
        else Checkpoints(make_checkpoint_pool(database_url))
    )
    ledger = (
        ledger
        if ledger is not None
        else Ledger(db, SpendLimits.from_settings(settings), MAX_OUTPUT_TOKENS)
    )
    pause_source = pause_source if pause_source is not None else ledger
    limits = limits if limits is not None else Limits(db, settings, pause_source)
    matches = (
        match_service
        if match_service is not None
        else MatchService.for_app(settings, db, ledger, limits)
    )
    if ask_service is None:
        ask_service = AskService.for_app(
            settings,
            db,
            checkpoint_pool if checkpoint_pool is not None else make_checkpoint_pool(database_url),
            ledger,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if settings.app_env == "dev":
            for name in REQUIRED_SECRETS:
                if getattr(settings, name) is None:
                    logger.warning("%s is not configured", name.upper())
        # The pools still open only on first use.
        yield
        await ask_service.close()
        await matches.close()
        try:
            await cleanup_checkpoints.close()
        finally:
            await db.close()

    app = FastAPI(
        title="portfolio-api",
        lifespan=lifespan,
        # No public schema or docs pages outside dev.
        openapi_url="/api/openapi.json" if settings.app_env == "dev" else None,
        docs_url="/api/docs" if settings.app_env == "dev" else None,
        redoc_url=None,
    )

    app.state.settings = settings
    app.state.db = db
    app.state.ledger = ledger
    app.state.limits = limits
    app.state.turnstile_verifier = TurnstileVerifier(settings, turnstile_client)

    @app.exception_handler(Rejected)
    async def rejected(_request: Request, error: Rejected) -> JSONResponse:
        return JSONResponse(
            {"code": error.code, "message": error.message}, status_code=error.status_code
        )

    @app.middleware("http")
    async def trusted_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Also protect dev docs/schema; unknown paths keep their normal 404 response.
        guarded = request.url.path not in ORIGIN_EXEMPT_PATHS and any(
            getattr(route, "path", None) == request.url.path for route in app.routes
        )
        if guarded:
            try:
                require_origin(request)
            except Rejected as error:
                return await rejected(request, error)
        return await call_next(request)

    app.include_router(ask_router(ask_service))
    app.include_router(match_router(matches))
    app.include_router(session_router())
    app.include_router(feedback_router())
    app.include_router(cleanup_router(cleanup_checkpoints))

    @app.get("/api/ping")
    async def ping() -> dict[str, str]:
        """Liveness and warm-up. Never touches the database."""
        return {"status": "ok"}

    if settings.app_env == "dev":

        @app.get("/api/health")
        async def health() -> JSONResponse:
            """Dev only: is the database reachable, and which pgvector is installed."""
            result = await db.health()
            if result.db_ok and result.pgvector:
                return JSONResponse({"status": "ok", "db": "ok", "pgvector": result.pgvector})
            return JSONResponse(
                {
                    "status": "degraded",
                    "db": "ok" if result.db_ok else "down",
                    "pgvector": result.pgvector,
                },
                status_code=503,
            )

    return app


app = create_app()
