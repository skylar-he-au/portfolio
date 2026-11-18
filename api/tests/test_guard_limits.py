"""Real Postgres counters and deterministic process-window boundaries."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from guard_helpers import Clock, guard_settings
from test_api import ExplodingPool

from app.db import Database, make_pool
from app.guard.limits import SYDNEY, Admission, Limits, SlidingWindow
from app.guard.session import Rejected, Session
from app.migrate import migrate

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


class CountingDatabase:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.calls = 0

    def connection(self):
        self.calls += 1
        return self.db.connection()


def visitor(number: int = 1, ip: str = "192.0.2.1") -> Session:
    return Session(hashlib.sha256(str(number).encode()).digest(), ip)


async def rows(db: Database) -> list:
    async with db.connection() as conn, conn.transaction():
        cursor = await conn.execute(
            "SELECT day, kind, scope, key, count FROM usage_counters ORDER BY day, kind, scope, key"
        )
        return await cursor.fetchall()


async def seed(db: Database, clock: Clock, scope: str, key: bytes, count: int) -> None:
    async with db.connection() as conn, conn.transaction():
        await conn.execute(
            "INSERT INTO usage_counters (day, kind, scope, key, count) "
            "VALUES (%s, 'ask', %s, %s, %s) "
            "ON CONFLICT (day, kind, scope, key) DO UPDATE SET count = EXCLUDED.count",
            (clock().astimezone(SYDNEY).date(), scope, key, count),
        )


async def test_twenty_asks_then_token_and_ip_marks(database, db_url):
    clock = Clock()
    counted = CountingDatabase(database)
    limits = Limits(counted, guard_settings(db_url), clock=clock)
    session = visitor()
    for _ in range(20):
        assert (await limits.admit("ask", session)).kind == "ask"
    assert {row[4] for row in await rows(database)} == {20}
    with pytest.raises(Rejected, match="limit_day"):
        await limits.admit("ask", session)
    calls = counted.calls
    # Token full is remembered before either sieve or the database.
    for _ in range(40):
        with pytest.raises(Rejected, match="limit_day"):
            await limits.admit("ask", session)
    assert counted.calls == calls
    # A new token reaches the IP layer, which is full; its token insert rolls back.
    with pytest.raises(Rejected, match="limit_day"):
        await limits.admit("ask", visitor(2))
    assert {row[4] for row in await rows(database)} == {20}
    calls = counted.calls
    with pytest.raises(Rejected, match="limit_day"):
        await limits.admit("ask", visitor(3))
    assert counted.calls == calls
    # Kinds have independent marks and database counters.
    admission = await limits.admit("match", session)
    assert admission.kind == "match"
    assert {row[4] for row in await rows(database) if row[1] == "match"} == {1}


async def test_site_mark_skips_all_further_database_queries(database, db_url):
    counted = CountingDatabase(database)
    limits = Limits(counted, guard_settings(db_url, ask_limit_site=3), clock=Clock())
    for number in range(3):
        await limits.admit("ask", visitor(number, f"192.0.2.{number + 1}"))
    with pytest.raises(Rejected, match="limit_day"):
        await limits.admit("ask", visitor(9, "192.0.2.99"))
    calls = counted.calls
    for number in range(10, 20):
        with pytest.raises(Rejected, match="limit_day"):
            await limits.admit("ask", visitor(number, f"192.0.2.{number}"))
    assert counted.calls == calls
    counts = await rows(database)
    assert len(counts) == 7  # failed request left no token/IP rows
    assert [row[4] for row in counts if row[2] == "site"] == [3]


async def test_concurrent_last_token_slot_has_one_winner(database, db_url):
    clock = Clock()
    session = visitor()
    await seed(database, clock, "token", session.token_key, 19)
    limits = Limits(database, guard_settings(db_url), clock=clock)
    results = await asyncio.gather(
        limits.admit("ask", session), limits.admit("ask", session), return_exceptions=True
    )
    assert sum(isinstance(result, Admission) for result in results) == 1
    failures = [result for result in results if isinstance(result, Rejected)]
    assert len(failures) == 1 and failures[0].code == "limit_day"
    counts = await rows(database)
    assert [row[4] for row in counts if row[2] == "token"] == [20]
    assert [row[4] for row in counts if row[2] in {"ip", "site"}] == [1, 1]


async def test_ip_failure_rolls_back_existing_token_increment(database, db_url):
    clock = Clock()
    limits = Limits(database, guard_settings(db_url), clock=clock)
    session = visitor()
    admission = await limits.admit("ask", session)
    await seed(database, clock, "ip", admission.ip_hash, 20)
    before = await rows(database)
    with pytest.raises(Rejected, match="limit_day"):
        await limits.admit("ask", session)
    assert await rows(database) == before


async def test_sydney_midnight_clears_marks(database, db_url):
    # January is UTC+11: 13:00 UTC is the next Sydney day.
    clock = Clock(datetime(2026, 1, 1, 12, 59, 59, tzinfo=UTC))
    limits = Limits(database, guard_settings(db_url, ask_limit_token=1), clock=clock)
    first = await limits.admit("ask", visitor())
    with pytest.raises(Rejected, match="limit_day"):
        await limits.admit("ask", visitor())
    clock.advance(seconds=1)
    second = await limits.admit("ask", visitor())
    assert second.day > first.day
    assert second.ip_hash != first.ip_hash


async def test_ip_hashes_and_daily_key_cache(database, db_url):
    clock = Clock()
    limits = Limits(database, guard_settings(db_url), clock=clock)
    # HTTP canonicalization is tested separately; Limits accepts canonical Session.ip.
    ipv6 = "2001:db8::/64"
    a = await limits.admit("ask", visitor(1, ipv6))
    b = await limits.admit("ask", visitor(2, ipv6))
    c = await limits.admit("ask", visitor(3, "192.0.2.1"))
    d = await limits.admit("ask", visitor(4, "192.0.2.2"))
    assert a.ip_hash == b.ip_hash
    assert c.ip_hash != d.ip_hash
    other_process = Limits(database, guard_settings(db_url), clock=clock)
    assert (await other_process.admit("ask", visitor(5, ipv6))).ip_hash == a.ip_hash
    clock.advance(days=1)
    next_day = await limits.admit("ask", visitor(1, ipv6))
    assert next_day.ip_hash != a.ip_hash
    clock.advance(days=1)
    third_day = await limits.admit("ask", visitor(1, ipv6))
    assert len(limits._daily_keys) == 2
    assert set(limits._daily_keys) == {next_day.day, third_day.day}


async def test_rolled_back_daily_key_is_not_cached(database, db_url):
    clock = Clock()
    await seed(database, clock, "token", visitor().token_key, 20)
    limits = Limits(database, guard_settings(db_url), clock=clock)
    with pytest.raises(Rejected, match="limit_day"):
        await limits.admit("ask", visitor())
    assert limits._daily_keys == {}
    async with database.connection() as conn, conn.transaction():
        cursor = await conn.execute("SELECT count(*) FROM daily_keys")
        assert await cursor.fetchone() == (0,)
    assert (await limits.admit("ask", visitor(2))).ip_hash


@pytest.mark.parametrize(("paused", "code"), [("day", "limit_day"), ("month", "paused_month")])
async def test_pause_precedes_database(refused_url, paused, code):
    clock = Clock()
    pool = ExplodingPool()

    class Pause:
        async def paused(self, now):
            assert now == clock()
            return paused

    limits = Limits(Database(pool), guard_settings(refused_url), Pause(), clock=clock)
    with pytest.raises(Rejected, match=code):
        await limits.admit("ask", visitor())
    assert pool.touched == []


async def test_pause_database_error_becomes_unavailable(refused_url):
    class Pause:
        async def paused(self, now):
            raise OSError("offline")

    limits = Limits(Database(ExplodingPool()), guard_settings(refused_url), Pause())
    with pytest.raises(Rejected, match="unavailable"):
        await limits.admit("ask", visitor())


async def test_coarse_ip_window_and_shared_db_budget(refused_url):
    clock = Clock()

    class Pause:
        async def paused(self, now):
            return "day"

    pool = ExplodingPool()
    limits = Limits(Database(pool), guard_settings(refused_url), Pause(), clock=clock)
    for _ in range(30):
        with pytest.raises(Rejected, match="limit_day"):
            await limits.admit("ask", visitor())
    with pytest.raises(Rejected, match="slow_down"):
        await limits.admit("ask", visitor())
    # The coarse failure consumed no DB slot; feedback uses the same remaining 30.
    assert all(limits.allow_db_query() for _ in range(30))
    assert not limits.allow_db_query()
    with pytest.raises(Rejected, match="slow_down"):
        await limits.admit("ask", visitor(2, "192.0.2.2"))
    clock.advance(seconds=60)
    with pytest.raises(Rejected, match="limit_day"):
        await limits.admit("ask", visitor())
    assert limits.allow_db_query()
    assert pool.touched == []


async def test_sixty_admissions_per_rolling_minute(refused_url):
    class Pause:
        async def paused(self, now):
            return "day"

    clock = Clock()
    limits = Limits(Database(ExplodingPool()), guard_settings(refused_url), Pause(), clock=clock)
    for number in range(60):
        with pytest.raises(Rejected, match="limit_day"):
            await limits.admit("ask", visitor(number, f"192.0.2.{number + 1}"))
    with pytest.raises(Rejected, match="slow_down"):
        await limits.admit("ask", visitor(100, "192.0.2.100"))
    clock.advance(seconds=59)
    assert not limits.allow_db_query()
    clock.advance(seconds=1)
    assert limits.allow_db_query()


def test_window_has_bounded_keys_and_expires():
    sieve = SlidingWindow(2, 60, max_keys=2)
    assert sieve.allow(b"a", 0)
    assert sieve.allow(b"a", 1)
    assert not sieve.allow(b"a", 2)
    assert sieve.allow(b"b", 2)
    assert not sieve.allow(b"c", 3)
    assert len(sieve._entries) == 2
    assert not sieve.allow(b"c", 60)  # a still has its second slot in the window
    assert sieve.allow(b"c", 62)
    assert len(sieve._entries) == 1


def test_rejected_attempts_also_occupy_window_slots():
    sieve = SlidingWindow(2, 60)
    assert sieve.allow(b"a", 0)
    assert sieve.allow(b"a", 1)
    assert not sieve.allow(b"a", 59)
    assert sieve.allow(b"a", 61)
    assert not sieve.allow(b"a", 62)


def test_sieves_store_hashes_and_default_bound(refused_url):
    clock = Clock()
    limits = Limits(Database(ExplodingPool()), guard_settings(refused_url), clock=clock)
    ip = "192.0.2.1"
    assert limits.allow_new_session(ip)
    assert ip not in repr(limits._issuing._entries)
    assert all(len(key) == 32 for key in limits._issuing._entries)
    assert limits._issuing.max_keys == limits._coarse.max_keys == 10_000
    assert limits._issuing.limit == 10 and limits._coarse.limit == 30
