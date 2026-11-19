"""All patrol tokens share one atomic Sydney-day counter across request kinds."""

import asyncio
from datetime import UTC, datetime

import pytest
from guard_helpers import Clock, guard_settings
from test_api import ExplodingPool
from test_guard_limits import database as database
from test_guard_limits import rows

from app.db import Database
from app.guard.limits import Admission, Limits
from app.guard.session import Rejected, Session, sign_token, verify_token_ex

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def patrol_session(ip: str = "192.0.2.1") -> Session:
    token, _ = sign_token("secret", 1000, patrol=True)
    verified = verify_token_ex("secret", token, 1000)
    assert verified is not None
    return Session(verified[0], ip, verified[1])


async def test_ten_mixed_requests_shared_by_tokens_and_servers(database, db_url):
    clock = Clock()
    settings = guard_settings(db_url)
    first = Limits(database, settings, clock=clock)
    second = Limits(database, settings, clock=clock)
    a, b = patrol_session(), patrol_session()
    assert a.token_key != b.token_key and a.ip == b.ip
    for number in range(10):
        limits, session = (first, a) if number < 5 else (second, b)
        kind = "ask" if number % 2 == 0 else "match"
        limits.sieve(kind, session)
        await limits.peek(kind, session)
        assert (await limits.admit(kind, session, sieved=True)).kind == "patrol"
    counts = await rows(database)
    assert [(kind, scope, bytes(key), count) for _, kind, scope, key, count in counts] == [
        ("patrol", "site", b"", 10)
    ]
    with pytest.raises(Rejected, match="limit_day") as caught:
        await first.admit("ask", b)
    assert caught.value.status_code == 429
    for kind in ("ask", "match"):
        with pytest.raises(Rejected, match="limit_day"):
            first.sieve(kind, a)
        with pytest.raises(Rejected, match="limit_day"):
            await second.peek(kind, a)
    normal = Session(a.token_key, a.ip)
    for kind in ("ask", "match"):
        assert (await first.admit(kind, normal)).kind == kind
    normal_counts = [row for row in await rows(database) if row[1] != "patrol"]
    assert len(normal_counts) == 6 and all(row[4] == 1 for row in normal_counts)


async def test_patrol_last_slot_is_atomic_between_servers(database, db_url):
    settings = guard_settings(db_url, patrol_limit_day=1)
    instances = [Limits(database, settings, clock=Clock()) for _ in range(2)]
    results = await asyncio.gather(
        instances[0].admit("ask", patrol_session()),
        instances[1].admit("match", patrol_session("192.0.2.2")),
        return_exceptions=True,
    )
    assert sum(isinstance(result, Admission) for result in results) == 1
    failures = [result for result in results if isinstance(result, Rejected)]
    assert len(failures) == 1 and failures[0].code == "limit_day"
    assert [row[4] for row in await rows(database)] == [1]


async def test_patrol_refund_clears_shared_marks_and_restores_slot(database, db_url):
    limits = Limits(database, guard_settings(db_url, patrol_limit_day=1), clock=Clock())
    session = patrol_session()
    admission = await limits.admit("match", session)
    with pytest.raises(Rejected, match="limit_day"):
        await limits.peek("ask", patrol_session())
    await limits.refund(admission, session)
    limits.sieve("ask", session)
    await limits.peek("ask", session)
    await limits.admit("ask", session, sieved=True)
    assert [(row[1], row[2], row[4]) for row in await rows(database)] == [("patrol", "site", 1)]


async def test_patrol_hold_and_count_roll_back_together(database, db_url):
    limits = Limits(database, guard_settings(db_url), clock=Clock())
    seen = []

    async def hold(conn):
        cursor = await conn.execute("SELECT kind, scope, count FROM usage_counters")
        seen.extend(await cursor.fetchall())
        raise RuntimeError("reservation failed")

    with pytest.raises(RuntimeError, match="reservation failed"):
        await limits.admit("match", patrol_session(), hold=hold)
    assert seen == [("patrol", "site", 1)]
    assert await rows(database) == []


async def test_patrol_fetch_keeps_normal_counters(database, db_url):
    limits = Limits(database, guard_settings(db_url), clock=Clock())
    assert (await limits.admit("fetch", patrol_session())).kind == "fetch"
    assert {(row[1], row[2], row[4]) for row in await rows(database)} == {
        ("fetch", "token", 1),
        ("fetch", "ip", 1),
        ("fetch", "site", 1),
    }


async def test_patrol_resets_at_sydney_midnight(database, db_url):
    clock = Clock(datetime(2026, 1, 1, 12, 59, 59, tzinfo=UTC))
    limits = Limits(database, guard_settings(db_url, patrol_limit_day=1), clock=clock)
    session = patrol_session()
    before = await limits.admit("ask", session)
    with pytest.raises(Rejected, match="limit_day"):
        await limits.admit("match", session)
    clock.advance(seconds=1)
    after = await limits.admit("match", session)
    assert after.day > before.day
    assert [row[4] for row in await rows(database)] == [1, 1]


async def test_patrol_coarse_window_and_db_budget_are_kept(refused_url):
    limits = Limits(Database(ExplodingPool()), guard_settings(refused_url), clock=Clock())
    for _ in range(30):
        limits.sieve("ask", patrol_session())
    with pytest.raises(Rejected, match="slow_down"):
        limits.sieve("match", patrol_session())
    for _ in range(60):
        assert limits.allow_db_query()
    with pytest.raises(Rejected, match="slow_down"):
        await limits.peek("match", patrol_session())


async def test_patrol_pause_source_is_kept(refused_url):
    class Paused:
        async def paused(self, now):
            return "month"

    pool = ExplodingPool()
    limits = Limits(Database(pool), guard_settings(refused_url), Paused(), clock=Clock())
    with pytest.raises(Rejected, match="paused_month"):
        await limits.admit("ask", patrol_session())
    assert pool.touched == []
