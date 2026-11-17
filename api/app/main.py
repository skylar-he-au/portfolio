"""FastAPI app. Run with `uvicorn app.main:app --no-access-log`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.ask import AskService, ask_router
from app.config import Settings, get_settings
from app.db import Database, make_pool
from app.qa.memory import make_checkpoint_pool


def create_app(
    settings: Settings | None = None,
    pool: Any | None = None,
    *,
    checkpoint_pool: Any | None = None,
    ask_service: AskService | None = None,
) -> FastAPI:
    """Build the app. Tests can inject fake pools or a whole `AskService`; by default both
    pools are made closed (the app's, and the checkpointer's for `/api/ask`)."""
    settings = settings or get_settings()
    database_url = settings.database_url.get_secret_value()
    db = Database(pool if pool is not None else make_pool(database_url))
    if ask_service is None:
        ask_service = AskService.for_app(
            settings,
            db,
            checkpoint_pool if checkpoint_pool is not None else make_checkpoint_pool(database_url),
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Nothing to do at startup: the pools open on first use.
        yield
        await ask_service.close()
        await db.close()

    app = FastAPI(
        title="portfolio-api",
        lifespan=lifespan,
        # No public schema or docs pages outside dev.
        openapi_url="/api/openapi.json" if settings.app_env == "dev" else None,
        docs_url="/api/docs" if settings.app_env == "dev" else None,
        redoc_url=None,
    )

    app.include_router(ask_router(ask_service))

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
