"""Small retention sweeps run after each question or job match is recorded.

Daily keys and counters keep today and yesterday (the privacy page says "two days").
Question records and job-match records keep 30 days and are deleted together
(`delete_old_questions` sweeps both, so the cleanup endpoint covers `match_log` too)."""

from typing import Any

from app.db import Database


async def delete_old_questions(db: Database, limit: int) -> int:
    """Delete up to `limit` question records and up to `limit` job-match records older than
    30 days, oldest first; returns how many rows went in all."""
    async with db.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM question_log WHERE id IN ("
            "SELECT id FROM question_log WHERE created_at < now() - interval '30 days' "
            "ORDER BY created_at LIMIT %s)",
            (limit,),
        )
        await conn.commit()
        deleted = cursor.rowcount
    return deleted + await delete_old_matches(db, limit)


async def delete_old_matches(db: Database, limit: int) -> int:
    async with db.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM match_log WHERE id IN ("
            "SELECT id FROM match_log WHERE created_at < now() - interval '30 days' "
            "ORDER BY created_at LIMIT %s)",
            (limit,),
        )
        await conn.commit()
        return cursor.rowcount


async def delete_expired_conversations(db: Database, saver: Any, limit: int) -> int:
    deleted = 0
    async with db.connection() as conn:
        cursor = await conn.execute(
            "SELECT id FROM conversations WHERE created_at < now() - interval '24 hours' "
            "AND (running_until IS NULL OR running_until < now()) "
            "ORDER BY created_at LIMIT %s",
            (limit,),
        )
        rows = await cursor.fetchall()
        for (conversation_id,) in rows:
            # Keep the row for retry if checkpoint deletion fails.
            await saver.adelete_thread(str(conversation_id))
            cursor = await conn.execute(
                "DELETE FROM conversations WHERE id = %s "
                "AND created_at < now() - interval '24 hours' "
                "AND (running_until IS NULL OR running_until < now())",
                (conversation_id,),
            )
            deleted += cursor.rowcount
        await conn.commit()
    return deleted


async def delete_old_daily_keys(db: Database) -> int:
    async with db.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM daily_keys WHERE day < (now() AT TIME ZONE 'Australia/Sydney')::date - 1"
        )
        await conn.commit()
        return cursor.rowcount


async def delete_old_counters(db: Database) -> int:
    async with db.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM usage_counters "
            "WHERE day < (now() AT TIME ZONE 'Australia/Sydney')::date - 1"
        )
        await conn.commit()
        return cursor.rowcount
