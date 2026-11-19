"""Bounded process sieves followed by atomic daily database counters.

Kinds: `ask`, `match`, and `fetch` (fetching a job ad by link, which also has an
in-process per-minute window per token and per IP). The job match uses three extra steps:
`sieve` (the in-process checks alone, once per request), `peek` (read-only: would one more
fit today?) and `refund` (take back one count, for a job ad with no requirements); `admit`
can run a `hold` callback (the spending reservation) in the transaction that adds to the
counts, so both happen or neither does.
"""

import hmac
import os
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

import psycopg
from psycopg import AsyncConnection
from psycopg_pool import PoolTimeout

from app.config import Settings
from app.db import Database
from app.guard.session import Rejected, Session

SYDNEY = ZoneInfo("Australia/Sydney")
COUNTER_SQL = """
    INSERT INTO usage_counters (day, kind, scope, key, count) VALUES (%s, %s, %s, %s, 1)
    ON CONFLICT (day, kind, scope, key) DO UPDATE SET count = usage_counters.count + 1
    WHERE usage_counters.count < %s RETURNING count
"""
KINDS = frozenset({"ask", "match", "fetch"})
SCOPES = ("token", "ip", "site")
DEV_MARK_SECONDS = 30


@dataclass(frozen=True)
class Admission:
    kind: str
    token_key: bytes
    ip_hash: bytes
    day: date


class PauseSource(Protocol):
    async def paused(self, now: datetime) -> Literal["day", "month"] | None: ...


class _NeverPaused:
    async def paused(self, now: datetime) -> Literal["day", "month"] | None:
        return None


class SlidingWindow:
    """Store at most `limit` timestamps per key, expiring at the window boundary."""

    def __init__(
        self, limit: int, seconds: float, *, max_keys: int = 10_000, count_rejected: bool = True
    ) -> None:
        self.limit = limit
        self.seconds = seconds
        self.max_keys = max_keys
        self.count_rejected = count_rejected
        self._entries: dict[bytes, deque[float]] = {}

    def allow(self, key: bytes, now: float) -> bool:
        cutoff = now - self.seconds
        entries = self._entries.get(key)
        if entries is None:
            if len(self._entries) >= self.max_keys:
                expired = [k for k, times in self._entries.items() if times[-1] <= cutoff]
                for old in expired:
                    del self._entries[old]
                if len(self._entries) >= self.max_keys:
                    return False
            entries = self._entries[key] = deque()
        while entries and entries[0] <= cutoff:
            entries.popleft()
        allowed = len(entries) < self.limit
        if allowed or self.count_rejected:
            entries.append(now)
            if len(entries) > self.limit:
                entries.popleft()
        return allowed


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Limits:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        pause_source: PauseSource | None = None,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._db = db
        self._settings = settings
        self._pause_source = pause_source if pause_source is not None else _NeverPaused()
        self._clock = clock
        self._process_key = os.urandom(32)
        self._marked_day: date | None = None
        self._marks: dict[tuple[str, str, bytes], float] = {}
        self._daily_keys: dict[date, bytes] = {}
        self._issuing = SlidingWindow(10, 3600)
        self._coarse = SlidingWindow(30, 60)
        self._db_budget = SlidingWindow(60, 60, count_rejected=False)
        self._fetch_minute = SlidingWindow(settings.fetch_limit_minute, 60)

    def _private_key(self, value: bytes) -> bytes:
        return hmac.digest(self._process_key, value, "sha256")

    def allow_new_session(self, ip: str) -> bool:
        return self._issuing.allow(self._private_key(ip.encode()), self._clock().timestamp())

    def allow_db_query(self) -> bool:
        return self._db_budget.allow(b"", self._clock().timestamp())

    def _marks_for(
        self, kind: str, session: Session, day: date
    ) -> tuple[tuple[str, str, bytes], ...]:
        if day != self._marked_day:
            self._marks.clear()
            self._marked_day = day
        return (
            (kind, "token", self._private_key(session.token_key)),
            (kind, "ip", self._private_key(session.ip.encode())),
            (kind, "site", self._private_key(b"")),
        )

    def _mark_live(self, mark: tuple[str, str, bytes], now: datetime) -> bool:
        added_at = self._marks.get(mark)
        if added_at is None:
            return False
        if self._settings.app_env == "dev" and now.timestamp() - added_at >= DEV_MARK_SECONDS:
            self._marks.pop(mark, None)
            return False
        return True

    def sieve(self, kind: str, session: Session) -> None:
        """The in-process checks of `admit`, once per request: a known full count for this
        kind today (`limit_day`), then the per-IP coarse window (`slow_down`)."""
        if kind not in KINDS:
            raise ValueError("unknown admission kind")
        now = self._clock()
        marks = self._marks_for(kind, session, now.astimezone(SYDNEY).date())
        if any(self._mark_live(mark, now) for mark in marks):
            raise Rejected("limit_day")
        if not self._coarse.allow(marks[1][2], now.timestamp()):
            raise Rejected("slow_down")

    async def _day_key(self, conn: AsyncConnection, day: date, *, create: bool) -> bytes | None:
        day_key = self._daily_keys.get(day)
        if day_key is not None:
            return day_key
        if create:
            await conn.execute(
                "INSERT INTO daily_keys (day, key) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (day, os.urandom(32)),
            )
        cursor = await conn.execute("SELECT key FROM daily_keys WHERE day = %s", (day,))
        row = await cursor.fetchone()
        return None if row is None else bytes(row[0])

    def _cache_day_key(self, day: date, day_key: bytes) -> None:
        self._daily_keys[day] = day_key
        today = self._clock().astimezone(SYDNEY).date()
        keep = {today, today - timedelta(days=1)}
        self._daily_keys = {d: key for d, key in self._daily_keys.items() if d in keep}

    async def peek(self, kind: str, session: Session) -> None:
        """Read-only: raise `limit_day` if this token, IP or the site has no count of `kind`
        left today. Never adds to a count (the job match calls it before fetching a job ad)."""
        if kind not in KINDS:
            raise ValueError("unknown admission kind")
        now = self._clock()
        day = now.astimezone(SYDNEY).date()
        marks = self._marks_for(kind, session, day)
        if any(self._mark_live(mark, now) for mark in marks):
            raise Rejected("limit_day")
        if not self._db_budget.allow(b"", now.timestamp()):
            raise Rejected("slow_down")
        try:
            async with self._db.connection() as conn:
                day_key = await self._day_key(conn, day, create=False)
                keys = {"token": session.token_key, "site": b""}
                if day_key is not None:
                    keys["ip"] = hmac.digest(day_key, session.ip.encode(), "sha256")
                cursor = await conn.execute(
                    "SELECT scope, key, count FROM usage_counters WHERE day = %s AND kind = %s "
                    "AND scope = ANY(%s) AND key = ANY(%s)",
                    (day, kind, list(keys), list(keys.values())),
                )
                rows = await cursor.fetchall()
        except (psycopg.Error, PoolTimeout, OSError, TimeoutError):
            raise Rejected("unavailable") from None
        for scope, key, count in rows:
            if keys.get(scope) != bytes(key):
                continue
            if count >= getattr(self._settings, f"{kind}_limit_{scope}"):
                self._marks[marks[SCOPES.index(scope)]] = self._clock().timestamp()
                raise Rejected("limit_day")

    async def refund(self, admission: Admission, session: Session) -> None:
        """Take back the count `admission` added (all three scopes), e.g. for a pasted text
        that turned out not to be a job ad. Clears this request's marks for that kind."""
        try:
            async with self._db.connection() as conn, conn.transaction():
                await conn.execute(
                    "UPDATE usage_counters SET count = count - 1 "
                    "WHERE day = %s AND kind = %s AND count > 0 AND ("
                    "(scope = 'token' AND key = %s) OR (scope = 'ip' AND key = %s) "
                    "OR (scope = 'site' AND key = %s))",
                    (admission.day, admission.kind, admission.token_key, admission.ip_hash, b""),
                )
        except (psycopg.Error, PoolTimeout, OSError, TimeoutError):
            raise Rejected("unavailable") from None
        if self._marked_day == admission.day:
            for mark in self._marks_for(admission.kind, session, admission.day):
                self._marks.pop(mark, None)

    async def admit(
        self,
        kind: str,
        session: Session,
        *,
        sieved: bool = False,
        hold: Callable[[AsyncConnection], Awaitable[Any]] | None = None,
    ) -> Admission:
        """Add one to today's `kind` count for this token, this IP hash and the site.

        `sieved`: the caller already ran `sieve` for this request, so the per-IP coarse
        window is not counted again (the known-full marks of `kind` are still checked).
        `hold`: run in the same transaction after the counts are added; if it raises, the
        counts are rolled back and its exception propagates (no pause check here then:
        `hold` does its own). `fetch` has no pause check either: the job match checked
        before (`Ledger.paused(now, "match")`).
        """
        if kind not in KINDS:
            raise ValueError("unknown admission kind")
        now = self._clock()
        day = now.astimezone(SYDNEY).date()
        marks = self._marks_for(kind, session, day)
        if any(self._mark_live(mark, now) for mark in marks):
            raise Rejected("limit_day")
        if not sieved and not self._coarse.allow(marks[1][2], now.timestamp()):
            raise Rejected("slow_down")
        if kind == "fetch":
            stamp = now.timestamp()
            token_ok = self._fetch_minute.allow(b"t" + marks[0][2], stamp)
            ip_ok = self._fetch_minute.allow(b"i" + marks[1][2], stamp)
            if not (token_ok and ip_ok):
                raise Rejected("slow_down")
        if not self._db_budget.allow(b"", now.timestamp()):
            raise Rejected("slow_down")
        failed_mark: tuple[str, str, bytes] | None = None
        try:
            if hold is None and kind != "fetch":
                paused = await self._pause_source.paused(now)
                if paused is not None:
                    raise Rejected("limit_day" if paused == "day" else "paused_month")
            async with self._db.connection() as conn, conn.transaction():
                day_key = await self._day_key(conn, day, create=True)
                if day_key is None:
                    raise Rejected("unavailable")
                ip_hash = hmac.digest(day_key, session.ip.encode(), "sha256")
                for index, (scope, key) in enumerate(
                    (("token", session.token_key), ("ip", ip_hash), ("site", b""))
                ):
                    limit = getattr(self._settings, f"{kind}_limit_{scope}")
                    cursor = await conn.execute(COUNTER_SQL, (day, kind, scope, key, limit))
                    if await cursor.fetchone() is None:
                        failed_mark = marks[index]
                        # Raising inside transaction() rolls back earlier increments too.
                        raise Rejected("limit_day")
                if hold is not None:
                    await hold(conn)
            # A key from a rolled-back insertion must never escape into the cache.
            self._cache_day_key(day, day_key)
            return Admission(kind, session.token_key, ip_hash, day)
        except Rejected:
            if failed_mark is not None and self._marked_day == day:
                self._marks[failed_mark] = self._clock().timestamp()
            raise
        except (psycopg.Error, PoolTimeout, OSError, TimeoutError):
            raise Rejected("unavailable") from None
