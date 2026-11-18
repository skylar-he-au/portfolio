"""Bounded process sieves followed by atomic daily database counters."""

import hmac
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

import psycopg
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
        self._marks: set[tuple[str, str, bytes]] = set()
        self._daily_keys: dict[date, bytes] = {}
        self._issuing = SlidingWindow(10, 3600)
        self._coarse = SlidingWindow(30, 60)
        self._db_budget = SlidingWindow(60, 60, count_rejected=False)

    def _private_key(self, value: bytes) -> bytes:
        return hmac.digest(self._process_key, value, "sha256")

    def allow_new_session(self, ip: str) -> bool:
        return self._issuing.allow(self._private_key(ip.encode()), self._clock().timestamp())

    def allow_db_query(self) -> bool:
        return self._db_budget.allow(b"", self._clock().timestamp())

    async def admit(self, kind: str, session: Session) -> Admission:
        if kind not in {"ask", "match", "tts"}:
            raise ValueError("unknown admission kind")
        now = self._clock()
        day = now.astimezone(SYDNEY).date()
        if day != self._marked_day:
            self._marks.clear()
            self._marked_day = day
        marks = (
            (kind, "token", self._private_key(session.token_key)),
            (kind, "ip", self._private_key(session.ip.encode())),
            (kind, "site", self._private_key(b"")),
        )
        if any(mark in self._marks for mark in marks):
            raise Rejected("limit_day")
        if not self._coarse.allow(marks[1][2], now.timestamp()):
            raise Rejected("slow_down")
        if not self._db_budget.allow(b"", now.timestamp()):
            raise Rejected("slow_down")
        failed_mark: tuple[str, str, bytes] | None = None
        try:
            paused = await self._pause_source.paused(now)
            if paused is not None:
                raise Rejected("limit_day" if paused == "day" else "paused_month")
            async with self._db.connection() as conn, conn.transaction():
                day_key = self._daily_keys.get(day)
                if day_key is None:
                    await conn.execute(
                        "INSERT INTO daily_keys (day, key) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (day, os.urandom(32)),
                    )
                    cursor = await conn.execute("SELECT key FROM daily_keys WHERE day = %s", (day,))
                    row = await cursor.fetchone()
                    if row is None:
                        raise Rejected("unavailable")
                    day_key = bytes(row[0])
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
            # A key from a rolled-back insertion must never escape into the cache.
            self._daily_keys[day] = day_key
            today = self._clock().astimezone(SYDNEY).date()
            keep = {today, today - timedelta(days=1)}
            self._daily_keys = {d: key for d, key in self._daily_keys.items() if d in keep}
            return Admission(kind, session.token_key, ip_hash, day)
        except Rejected:
            if failed_mark is not None and self._marked_day == day:
                self._marks.add(failed_mark)
            raise
        except (psycopg.Error, PoolTimeout, OSError, TimeoutError):
            raise Rejected("unavailable") from None
