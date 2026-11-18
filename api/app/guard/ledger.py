"""Postgres reservations and settlement for one question's spending."""

import asyncio
import logging
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, Protocol
from uuid import UUID
from zoneinfo import ZoneInfo

from psycopg import AsyncConnection

from app.db import Database
from app.guard.prices import cost_usd

logger = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "google", "openai")
RESERVATION_TTL_S = 150
MONTHLY_FRACTION = Decimal("0.8")
SYDNEY = ZoneInfo("Australia/Sydney")
_warned_unconfigured = False

type PauseKind = Literal["day", "month"]


class AIPaused(Exception):
    def __init__(self, kind: PauseKind) -> None:
        self.kind = kind
        super().__init__(kind)


class ProviderBudgetExhausted(Exception):
    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(provider)


class PauseSource(Protocol):
    async def paused(self, now: datetime) -> PauseKind | None: ...


def sydney_day(now: datetime) -> date:
    return now.astimezone(SYDNEY).date()


def month_bounds(day: date) -> tuple[date, date]:
    first = day.replace(day=1)
    following = (first + timedelta(days=32)).replace(day=1)
    return first, following


def next_month_start(now: datetime) -> datetime:
    _, following = month_bounds(sydney_day(now))
    return datetime(following.year, following.month, 1, tzinfo=SYDNEY)


@dataclass(frozen=True)
class SpendLimits:
    daily: Mapping[str, Decimal | None]
    monthly: Mapping[str, Decimal | None]
    models: Mapping[str, str]
    input_tokens_estimate: int = 12000

    @classmethod
    def from_settings(cls, settings: Any) -> "SpendLimits":
        def amounts(period: str) -> dict[str, Decimal | None]:
            result = {}
            for provider in PROVIDERS:
                value = getattr(settings, f"spend_{period}_usd_{provider}", None)
                result[provider] = None if value is None else Decimal(str(value))
            return result

        return cls(
            daily=amounts("daily"),
            monthly=amounts("monthly"),
            models={
                "anthropic": settings.answer_model,
                "google": settings.check_model,
                "openai": settings.embed_model,
            },
            input_tokens_estimate=getattr(settings, "spend_input_tokens_estimate", 12000),
        )


async def _lock_provider(conn: AsyncConnection, provider: str) -> None:
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (f"spend-ledger:{provider}",),
    )


async def _add_spent(conn: AsyncConnection, provider: str, day: date, usd: Decimal) -> None:
    if usd == 0:
        return
    await conn.execute(
        "INSERT INTO spend_daily (day, provider, spent_usd) VALUES (%s, %s, %s) "
        "ON CONFLICT (day, provider) DO UPDATE "
        "SET spent_usd = spend_daily.spent_usd + EXCLUDED.spent_usd",
        (day, provider, usd),
    )


async def _book_expired(conn: AsyncConnection, provider: str, now: datetime) -> None:
    cursor = await conn.execute(
        "DELETE FROM spend_reservations WHERE provider = %s AND expires_at < %s RETURNING day, usd",
        (provider, now),
    )
    for day, usd in await cursor.fetchall():
        await _add_spent(conn, provider, day, usd)


async def _totals(
    conn: AsyncConnection, provider: str, day: date, now: datetime
) -> tuple[Decimal, Decimal]:
    first, following = month_bounds(day)
    cursor = await conn.execute(
        "SELECT "
        "COALESCE((SELECT sum(spent_usd) FROM spend_daily "
        "WHERE provider = %s AND day = %s), 0) + "
        "COALESCE((SELECT sum(usd) FROM spend_reservations "
        "WHERE provider = %s AND day = %s AND expires_at >= %s), 0), "
        "COALESCE((SELECT sum(spent_usd) FROM spend_daily "
        "WHERE provider = %s AND day >= %s AND day < %s), 0) + "
        "COALESCE((SELECT sum(usd) FROM spend_reservations "
        "WHERE provider = %s AND day >= %s AND day < %s AND expires_at >= %s), 0)",
        (
            provider,
            day,
            provider,
            day,
            now,
            provider,
            first,
            following,
            provider,
            first,
            following,
            now,
        ),
    )
    row = await cursor.fetchone()
    # The SELECT always returns its two scalar expressions.
    return row[0], row[1]


class Ledger:
    def __init__(self, db: Database, limits: SpendLimits, max_output_tokens: int) -> None:
        self.db = db
        self.limits = limits
        self.max_output_tokens = max_output_tokens

    def _configured(self) -> bool:
        global _warned_unconfigured
        configured = all(
            self.limits.daily.get(provider) is not None
            and self.limits.monthly.get(provider) is not None
            for provider in PROVIDERS
        )
        if not configured and not _warned_unconfigured:
            logger.warning("AI spending limits are not configured")
            _warned_unconfigured = True
        return configured

    def question_worst(self) -> dict[str, Decimal]:
        return {
            provider: count
            * cost_usd(
                self.limits.models[provider],
                {
                    "input_tokens": self.limits.input_tokens_estimate,
                    "output_tokens": 0 if provider == "openai" else self.max_output_tokens,
                },
            )
            for provider, count in zip(PROVIDERS, (3, 2, 3), strict=True)
        }

    async def _has_pause(self, conn: AsyncConnection, now: datetime) -> bool:
        cursor = await conn.execute("SELECT 1 FROM ai_pause WHERE until > %s LIMIT 1", (now,))
        return await cursor.fetchone() is not None

    async def paused(self, now: datetime) -> PauseKind | None:
        if not self._configured():
            return "day"
        async with self.db.connection() as conn:
            if await self._has_pause(conn, now):
                return "month"
            worst = self.question_worst()
            totals = {
                provider: await _totals(conn, provider, sydney_day(now), now)
                for provider in PROVIDERS
            }
        for provider, (_, month) in totals.items():
            if month + worst[provider] > self.limits.monthly[provider] * MONTHLY_FRACTION:
                return "month"
        for provider, (day, _) in totals.items():
            if day + worst[provider] > self.limits.daily[provider]:
                return "day"
        return None

    async def _check(
        self, conn: AsyncConnection, provider: str, day: date, now: datetime, extra: Decimal
    ) -> None:
        day_total, month_total = await _totals(conn, provider, day, now)
        if month_total + extra > self.limits.monthly[provider] * MONTHLY_FRACTION:
            raise AIPaused("month")
        if day_total + extra > self.limits.daily[provider]:
            raise AIPaused("day")

    async def reserve_question(
        self, question_id: UUID, now: datetime | None = None
    ) -> "SpendScope":
        if not self._configured():
            raise AIPaused("day")
        now = now if now is not None else datetime.now(UTC)
        day = sydney_day(now)
        amounts = self.question_worst()
        async with self.db.connection() as conn, conn.transaction():
            for provider in PROVIDERS:
                await _lock_provider(conn, provider)
            if await self._has_pause(conn, now):
                raise AIPaused("month")
            for provider in PROVIDERS:
                await _book_expired(conn, provider, now)
                await self._check(conn, provider, day, now, amounts[provider])
            for provider in PROVIDERS:
                await conn.execute(
                    "INSERT INTO spend_reservations (id, provider, day, usd, expires_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        question_id,
                        provider,
                        day,
                        amounts[provider],
                        now + timedelta(seconds=RESERVATION_TTL_S),
                    ),
                )
        return SpendScope(self, question_id, day, amounts)

    async def pause_provider(self, provider: str, now: datetime | None = None) -> None:
        now = now if now is not None else datetime.now(UTC)
        async with self.db.connection() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO ai_pause (provider, until, reason) VALUES (%s, %s, %s) "
                "ON CONFLICT (provider) DO UPDATE "
                "SET until = EXCLUDED.until, reason = EXCLUDED.reason",
                (provider, next_month_start(now), "provider budget error"),
            )


@dataclass
class SpendScope:
    ledger: Ledger
    question_id: UUID
    day: date
    reserved: dict[str, Decimal]
    used: dict[str, Decimal] = field(default_factory=lambda: dict.fromkeys(PROVIDERS, Decimal(0)))
    pending: dict[str, Decimal] = field(
        default_factory=lambda: dict.fromkeys(PROVIDERS, Decimal(0)),
    )
    called: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    settled: bool = False
    _settle_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    async def ensure(self, provider: str, worst: Decimal) -> None:
        async with self.lock:
            need = self.used[provider] + self.pending[provider] + worst
            if need > self.reserved[provider]:
                diff = need - self.reserved[provider]
                now = datetime.now(UTC)
                async with self.ledger.db.connection() as conn, conn.transaction():
                    await _lock_provider(conn, provider)
                    await _book_expired(conn, provider, now)
                    await self.ledger._check(conn, provider, self.day, now, diff)
                    cursor = await conn.execute(
                        "UPDATE spend_reservations SET usd = usd + %s "
                        "WHERE id = %s AND provider = %s",
                        (diff, self.question_id, provider),
                    )
                    if cursor.rowcount == 0:
                        raise AIPaused("day")
                self.reserved[provider] += diff
            self.pending[provider] += worst
            self.called.add(provider)

    def finish(self, provider: str, worst: Decimal, actual: Decimal | None) -> None:
        self.pending[provider] -= worst
        self.used[provider] += actual if actual is not None else worst

    async def _settle(self, charges: dict[str, Decimal]) -> None:
        async with self.ledger.db.connection() as conn, conn.transaction():
            for provider in PROVIDERS:
                await _lock_provider(conn, provider)
            for provider in PROVIDERS:
                cursor = await conn.execute(
                    "DELETE FROM spend_reservations WHERE id = %s AND provider = %s RETURNING day",
                    (self.question_id, provider),
                )
                row = await cursor.fetchone()
                if row is not None:
                    await _add_spent(conn, provider, row[0], charges[provider])
        self.settled = True

    async def settle(self) -> None:
        async with self.lock:
            if self.settled:
                return
            if self._settle_task is None:
                charges = {
                    provider: self.used[provider] + self.pending[provider] for provider in PROVIDERS
                }
                self._settle_task = asyncio.create_task(self._settle(charges))
            task = self._settle_task
        await asyncio.shield(task)

    def total_usd(self) -> Decimal:
        return sum(self.used.values(), Decimal(0))

    def providers(self) -> list[str]:
        return [provider for provider in PROVIDERS if provider in self.called]


current_scope: ContextVar[SpendScope | None] = ContextVar("spend_scope", default=None)
