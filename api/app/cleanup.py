"""Bounded retention for the scheduler and `python -m app.cleanup`."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from app import retention
from app.config import get_settings
from app.db import Database, make_pool
from app.guard.ledger import _add_spent, _lock_provider
from app.logging_setup import configure_logging
from app.qa.memory import Checkpoints, make_checkpoint_pool

logger = logging.getLogger(__name__)


class CheckpointStore(Protocol):
    async def saver(self) -> Any: ...


@dataclass(frozen=True)
class CleanupResult:
    deleted: dict[str, int]
    failed: list[str]


async def _delete_reservations(db: Database) -> int:
    deleted = 0
    # Book and delete together, under the same locks as normal settlement.
    async with db.connection() as conn, conn.transaction():
        for provider in ("anthropic", "google", "openai"):
            await _lock_provider(conn, provider)
            cursor = await conn.execute(
                "DELETE FROM spend_reservations WHERE provider = %s "
                "AND expires_at < now() - interval '1 day' RETURNING day, usd",
                (provider,),
            )
            for day, usd in await cursor.fetchall():
                await _add_spent(conn, provider, day, usd)
                deleted += 1
    return deleted


async def run_cleanup(db: Database, checkpoints: CheckpointStore) -> CleanupResult:
    result = CleanupResult(
        dict.fromkeys(
            ("questions", "matches", "conversations", "daily_keys", "counters", "reservations"), 0
        ),
        [],
    )

    def batched(name: str, sweep: Callable[..., Awaitable[int]]) -> Callable[[], Awaitable[int]]:
        async def step() -> int:
            for _ in range(20):
                count = await sweep(db, limit=500)
                result.deleted[name] += count
                if count < 500:
                    break
            return result.deleted[name]

        return step

    async def conversations() -> int:
        saver = await checkpoints.saver()
        for _ in range(20):
            count = await retention.delete_expired_conversations(db, saver, limit=500)
            result.deleted["conversations"] += count
            if count < 500:
                break
        return result.deleted["conversations"]

    # Each sweep has its own boundary so a failed saver cannot stop key cleanup.
    for name, step in (
        ("questions", batched("questions", retention.delete_old_question_rows)),
        ("matches", batched("matches", retention.delete_old_matches)),
        ("conversations", conversations),
        ("daily_keys", lambda: retention.delete_old_daily_keys(db)),
        ("counters", lambda: retention.delete_old_counters(db)),
        ("reservations", lambda: _delete_reservations(db)),
    ):
        try:
            result.deleted[name] = await step()
        except Exception as error:  # noqa: BLE001 - each cleanup step must be isolated
            logger.warning("cleanup step %s failed: %s", name, type(error).__name__)
            result.failed.append(name)
    return result


async def main() -> int:
    configure_logging()
    settings = get_settings()
    url = settings.database_url.get_secret_value()
    db = Database(make_pool(url))
    checkpoints = Checkpoints(make_checkpoint_pool(url))
    try:
        result = await run_cleanup(db, checkpoints)
        print(json.dumps(asdict(result)))
        return 1 if result.failed else 0
    finally:
        try:
            await checkpoints.close()
        finally:
            await db.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
