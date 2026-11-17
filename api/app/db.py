"""Database access for the web app.

The pool is created closed and opened on first use, so starting the app (and
`/api/ping`, which Cloud Run and page-load warm-ups hit) never wakes the database.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

# Limits for the dev health check; each one alone keeps a hung database from
# holding the request (contract: 503 within 3 seconds).
CONNECT_TIMEOUT_S = 2
POOL_WAIT_S = 2.0
STATEMENT_TIMEOUT = "2s"
HEALTH_DEADLINE_S = 2.5


def make_pool(database_url: str) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        conninfo=database_url,
        open=False,
        # No idle connections kept: Neon can scale to zero between visits.
        min_size=0,
        max_size=5,
        max_idle=60,
        timeout=POOL_WAIT_S,
        # Stop background reconnect attempts quickly when the database is down.
        reconnect_timeout=30,
        check=AsyncConnectionPool.check_connection,
        kwargs={"connect_timeout": CONNECT_TIMEOUT_S},
    )


@dataclass(frozen=True)
class HealthResult:
    db_ok: bool
    pgvector: str | None


class Database:
    """Owns the pool and opens it lazily."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._opened = False
        self._open_lock = asyncio.Lock()

    async def _ensure_open(self) -> None:
        if self._opened:
            return
        async with self._open_lock:
            if not self._opened:
                # wait=False: with min_size=0 there is nothing to wait for.
                await self._pool.open(wait=False)
                self._opened = True

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    async def health(self) -> HealthResult:
        try:
            async with asyncio.timeout(HEALTH_DEADLINE_S):
                return await self._health()
        except Exception as exc:  # noqa: BLE001 - any failure means "down"
            # Type name only: messages can contain hosts or user names.
            logger.warning("health check failed: %s", type(exc).__name__)
            return HealthResult(db_ok=False, pgvector=None)

    async def _health(self) -> HealthResult:
        await self._ensure_open()
        async with self._pool.connection(timeout=POOL_WAIT_S) as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                cur = await conn.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = await cur.fetchone()
        return HealthResult(db_ok=True, pgvector=row[0] if row else None)
