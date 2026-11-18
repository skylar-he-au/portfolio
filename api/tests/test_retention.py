"""Bounded, Sydney-day retention (today and yesterday kept) and checkpoint-first deletion."""

import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from app import retention
from app.db import Database, make_pool
from app.migrate import migrate
from app.qa.memory import Checkpoints, Conversations, make_checkpoint_pool
from app.records import QuestionRecord, Records

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def database(db_url: str) -> AsyncIterator[Database]:
    migrate(db_url)
    db = Database(make_pool(db_url))
    try:
        yield db
    finally:
        await db.close()


async def rows(db, query, params=()):
    async with db.connection() as conn:
        cursor = await conn.execute(query, params)
        return await cursor.fetchall()


async def seed_checkpoint(saver, cid):
    config = {"configurable": {"thread_id": cid, "checkpoint_ns": ""}}
    await saver.aput(config, empty_checkpoint(), {}, {})
    assert await saver.aget_tuple(config) is not None
    return config


def record() -> QuestionRecord:
    return QuestionRecord(uuid.uuid4(), "question", "en", "v-test", [], Decimal(0), b"hash")


async def test_questions_delete_oldest_up_to_limit(database):
    ids = [uuid.uuid4() for _ in range(3)]
    async with database.connection() as conn:
        for cid, days in zip(ids, (32, 31, 29), strict=True):
            await conn.execute(
                "INSERT INTO question_log (id, question, prompt_version, ip_hash, created_at) "
                "VALUES (%s, 'old question', 'v-test', %s, now() - %s * interval '1 day')",
                (cid, b"hash", days),
            )
        await conn.commit()
    assert await retention.delete_old_questions(database, 1) == 1
    assert {r[0] for r in await rows(database, "SELECT id FROM question_log")} == set(ids[1:])
    assert await retention.delete_old_questions(database, 20) == 1
    assert await retention.delete_old_questions(database, 20) == 0
    assert await rows(database, "SELECT id FROM question_log") == [(ids[2],)]


async def test_conversations_limit_skip_live_locks_and_retry_checkpoint_failure(database):
    conversations = Conversations(database)
    ids = [await conversations.resolve(None) for _ in range(4)]
    async with database.connection() as conn:
        for cid, hours in zip(ids, (27, 26, 25, 1), strict=True):
            await conn.execute(
                "UPDATE conversations SET created_at = now() - %s * interval '1 hour' "
                "WHERE id = %s",
                (hours, cid),
            )
        await conn.execute(
            "UPDATE conversations SET running_until = now() - interval '1 second' WHERE id = %s",
            (ids[1],),
        )
        await conn.commit()
    assert await conversations.acquire(ids[2], uuid.uuid4())
    saver = AsyncMock()
    saver.adelete_thread.side_effect = RuntimeError("checkpoint failed")
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        await retention.delete_expired_conversations(database, saver, 1)
    assert len(await rows(database, "SELECT id FROM conversations")) == 4
    saver.adelete_thread.side_effect = None
    assert await retention.delete_expired_conversations(database, saver, 1) == 1
    assert saver.adelete_thread.call_args.args == (ids[0],)
    assert await retention.delete_expired_conversations(database, saver, 20) == 1
    assert {str(r[0]) for r in await rows(database, "SELECT id FROM conversations")} == set(ids[2:])


async def test_conversation_delete_rechecks_lock(database):
    cid = await Conversations(database).resolve(None)
    async with database.connection() as conn:
        await conn.execute(
            "UPDATE conversations SET created_at = now() - interval '25 hours' WHERE id = %s",
            (cid,),
        )
        await conn.commit()

    async def take_lock(thread):
        assert thread == cid
        assert await Conversations(database).acquire(cid, uuid.uuid4())

    saver = AsyncMock()
    saver.adelete_thread.side_effect = take_lock
    assert await retention.delete_expired_conversations(database, saver, 20) == 0
    assert len(await rows(database, "SELECT id FROM conversations")) == 1


async def test_daily_keys_and_counters_use_sydney_today(database):
    async with database.connection() as conn:
        for offset in (-2, -1, 0, 1):
            await conn.execute(
                "INSERT INTO daily_keys (day, key) "
                "VALUES ((now() AT TIME ZONE 'Australia/Sydney')::date + %s, %s)",
                (offset, b"key"),
            )
            await conn.execute(
                "INSERT INTO usage_counters (day, kind, scope, key) "
                "VALUES ((now() AT TIME ZONE 'Australia/Sydney')::date + %s, 'ask', 'site', %s)",
                (offset, b""),
            )
        await conn.commit()
    assert await retention.delete_old_daily_keys(database) == 1  # only day -2
    assert await retention.delete_old_counters(database) == 1
    keys = await rows(
        database,
        "SELECT day - (now() AT TIME ZONE 'Australia/Sydney')::date FROM daily_keys ORDER BY day",
    )
    counters = await rows(
        database,
        "SELECT day - (now() AT TIME ZONE 'Australia/Sydney')::date "
        "FROM usage_counters ORDER BY day",
    )
    assert keys == counters == [(-1,), (0,), (1,)]
    assert await retention.delete_old_daily_keys(database) == 0
    assert await retention.delete_old_counters(database) == 0


async def test_record_write_sweeps_real_checkpoints_and_keeps_running_thread(database, db_url):
    checkpoints = Checkpoints(make_checkpoint_pool(db_url))
    try:
        saver = await checkpoints.saver()
        conversations = Conversations(database)
        expired = await conversations.resolve(None)
        running = await conversations.resolve(None)
        expired_config = await seed_checkpoint(saver, expired)
        running_config = await seed_checkpoint(saver, running)
        records = Records(database, checkpoints)
        await records.write(record())
        async with database.connection() as conn:
            await conn.execute("UPDATE question_log SET created_at = now() - interval '31 days'")
            await conn.execute("UPDATE conversations SET created_at = now() - interval '25 hours'")
            await conn.commit()
        assert await conversations.acquire(running, uuid.uuid4())
        new = record()
        await records.write(new)
        await records.write(new)  # the turn id makes retries idempotent
        assert await rows(database, "SELECT id FROM question_log") == [(new.id,)]
        assert await rows(database, "SELECT id FROM conversations") == [(uuid.UUID(running),)]
        assert await saver.aget_tuple(expired_config) is None
        assert await saver.aget_tuple(running_config) is not None
    finally:
        await checkpoints.close()


async def test_records_isolates_insert_and_each_sweep_failure(monkeypatch, caplog):
    class BrokenDatabase:
        def connection(self):
            raise RuntimeError("secret-question-answer-token-ip")

    db = BrokenDatabase()
    checkpoints = AsyncMock()
    checkpoints.saver.side_effect = RuntimeError("secret-question-answer-token-ip")
    sweeps = []
    for name in ("delete_old_questions", "delete_old_daily_keys", "delete_old_counters"):
        sweep = AsyncMock(side_effect=RuntimeError("secret-question-answer-token-ip"))
        monkeypatch.setattr(retention, name, sweep)
        sweeps.append(sweep)
    await Records(db, checkpoints).write(record())
    sweeps[0].assert_awaited_once_with(db, limit=20)
    for sweep in sweeps[1:]:
        sweep.assert_awaited_once_with(db)
    checkpoints.saver.assert_awaited_once()
    assert len(caplog.records) == 5
    assert "RuntimeError" in caplog.text
    assert "secret-question-answer-token-ip" not in caplog.text


# --- step 6b2: the job-match log ------------------------------------------------------------


async def seed_matches(database, days):
    ids = [uuid.uuid4() for _ in days]
    async with database.connection() as conn:
        for mid, age in zip(ids, days, strict=True):
            await conn.execute(
                "INSERT INTO match_log (id, source, prompt_version, ip_hash, created_at) "
                "VALUES (%s, 'lever', 'match-v1', %s, now() - %s * interval '1 day')",
                (mid, b"hash", age),
            )
        await conn.commit()
    return ids


async def test_matches_delete_after_30_days_up_to_limit(database):
    ids = await seed_matches(database, (32, 31, 29))
    assert await retention.delete_old_matches(database, 1) == 1
    assert {r[0] for r in await rows(database, "SELECT id FROM match_log")} == set(ids[1:])
    assert await retention.delete_old_matches(database, 20) == 1
    assert await rows(database, "SELECT id FROM match_log") == [(ids[2],)]


async def test_question_sweep_and_cleanup_also_delete_old_matches(database):
    ids = await seed_matches(database, (40, 1))
    assert await retention.delete_old_questions(database, 20) == 1
    assert await rows(database, "SELECT id FROM match_log") == [(ids[1],)]


async def test_match_record_write_sweeps_old_matches(database):
    from app.records import MatchRecord, MatchRecords

    old = await seed_matches(database, (35,))
    new = MatchRecord(uuid.uuid4(), "text", "en", "match-v1", ["anthropic"], Decimal("0.01"), b"h")
    await MatchRecords(database).write(new)
    await MatchRecords(database).write(new)  # same id twice: one row
    stored = await rows(database, "SELECT id, source, job_language, cost_usd FROM match_log")
    assert stored == [(new.id, "text", "en", Decimal("0.010000"))]
    assert old[0] not in {r[0] for r in stored}
