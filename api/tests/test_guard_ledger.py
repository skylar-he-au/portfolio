import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.db import Database, make_pool
from app.guard.ledger import (
    MONTHLY_FRACTION,
    PROVIDERS,
    SYDNEY,
    AIPaused,
    Ledger,
    SpendLimits,
    month_bounds,
    next_month_start,
    sydney_day,
)
from app.guard.prices import cost_usd
from app.migrate import migrate

pytestmark = pytest.mark.anyio

MODELS = {
    "anthropic": "claude-haiku-5-5",
    "google": "gemini-3.5-flash-lite",
    "openai": "text-embedding-3-small",
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def limits(daily: str = "10", monthly: str = "100", estimate: int = 12000) -> SpendLimits:
    return SpendLimits(
        dict.fromkeys(PROVIDERS, Decimal(daily)),
        dict.fromkeys(PROVIDERS, Decimal(monthly)),
        MODELS,
        estimate,
    )


@pytest.fixture
def migrated_url(db_url: str) -> str:
    migrate(db_url)
    return db_url


@pytest.fixture
async def ledger(migrated_url: str) -> AsyncIterator[Ledger]:
    db = Database(make_pool(migrated_url))
    try:
        yield Ledger(db, limits(), 1024)
    finally:
        await db.close()


async def execute(ledger: Ledger, query: str, params: tuple = ()) -> None:
    async with ledger.db.connection() as conn:
        await conn.execute(query, params)


async def rows(ledger: Ledger, query: str, params: tuple = ()) -> list:
    async with ledger.db.connection() as conn:
        cursor = await conn.execute(query, params)
        return await cursor.fetchall()


async def add_spent(ledger: Ledger, day: date, provider: str, amount: str) -> None:
    await execute(
        ledger,
        "INSERT INTO spend_daily (day, provider, spent_usd) VALUES (%s, %s, %s)",
        (day, provider, Decimal(amount)),
    )


def test_settings_missing_limits_and_default_estimate() -> None:
    settings = SimpleNamespace(
        answer_model=MODELS["anthropic"],
        check_model=MODELS["google"],
        embed_model=MODELS["openai"],
    )
    result = SpendLimits.from_settings(settings)
    assert result.daily == result.monthly == dict.fromkeys(PROVIDERS)
    assert result.models == MODELS
    assert result.input_tokens_estimate == 12000
    settings.spend_daily_usd_anthropic = 0.15
    settings.spend_monthly_usd_google = "3.25"
    settings.spend_input_tokens_estimate = 9000
    result = SpendLimits.from_settings(settings)
    assert result.daily["anthropic"] == Decimal("0.15")
    assert result.monthly["google"] == Decimal("3.25")
    assert result.input_tokens_estimate == 9000


def test_calendar_helpers_and_worst_case() -> None:
    assert month_bounds(date(2026, 12, 31)) == (date(2026, 12, 1), date(2027, 1, 1))
    now = datetime(2026, 11, 30, 13, 30, tzinfo=UTC)
    assert sydney_day(now) == date(2026, 12, 1)
    assert next_month_start(now) == datetime(2027, 1, 1, tzinfo=SYDNEY)
    ledger = Ledger(None, limits(), 1024)
    expected = {}
    for provider, count in zip(PROVIDERS, (3, 2, 3), strict=True):
        expected[provider] = count * cost_usd(
            MODELS[provider],
            {"input_tokens": 12000, "output_tokens": 0 if provider == "openai" else 1024},
        )
    assert ledger.question_worst() == expected


async def test_daily_rejection_rolls_back_all_providers(ledger: Ledger) -> None:
    now = datetime.now(UTC)
    ledger.limits = limits(daily="0.15")
    await add_spent(ledger, sydney_day(now), "anthropic", "0.149")
    with pytest.raises(AIPaused) as caught:
        await ledger.reserve_question(uuid4(), now)
    assert caught.value.kind == "day"
    assert await rows(ledger, "SELECT * FROM spend_reservations") == []


async def test_monthly_eighty_percent_rejects_before_daily(ledger: Ledger) -> None:
    now = datetime.now(UTC)
    ledger.limits = limits(daily="0.01", monthly="1")
    await add_spent(ledger, sydney_day(now), "anthropic", "0.8")
    with pytest.raises(AIPaused) as caught:
        await ledger.reserve_question(uuid4(), now)
    assert caught.value.kind == "month"
    assert await rows(ledger, "SELECT * FROM spend_reservations") == []


async def test_concurrent_reservations_admit_exactly_one(ledger: Ledger) -> None:
    amounts = ledger.question_worst()
    ledger.limits = SpendLimits(amounts, ledger.limits.monthly, MODELS)
    now = datetime.now(UTC)
    results = await asyncio.gather(
        ledger.reserve_question(uuid4(), now),
        ledger.reserve_question(uuid4(), now),
        return_exceptions=True,
    )
    failures = [result for result in results if isinstance(result, AIPaused)]
    assert len(failures) == 1
    assert failures[0].kind == "day"
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert len(await rows(ledger, "SELECT * FROM spend_reservations")) == 3


async def test_expiry_books_on_the_original_day(ledger: Ledger) -> None:
    now = datetime.now(UTC)
    old_day = sydney_day(now) - timedelta(days=40)
    question_id = uuid4()
    await add_spent(ledger, old_day, "anthropic", "0.002")
    await execute(
        ledger,
        "INSERT INTO spend_reservations VALUES (%s, %s, %s, %s, %s)",
        (question_id, "anthropic", old_day, Decimal("0.01"), now - timedelta(seconds=1)),
    )
    await ledger.reserve_question(uuid4(), now)
    remaining = await rows(
        ledger,
        "SELECT * FROM spend_reservations WHERE id = %s",
        (question_id,),
    )
    assert remaining == []
    spent = await rows(
        ledger,
        "SELECT day, spent_usd FROM spend_daily WHERE provider = 'anthropic'",
    )
    assert spent == [(old_day, Decimal("0.012"))]


@pytest.mark.parametrize("day", [date(2026, 11, 17), date(2026, 11, 30), date(2027, 4, 4)])
async def test_settle_uses_start_day_across_midnight_month_and_dst(
    ledger: Ledger, day: date
) -> None:
    now = datetime(day.year, day.month, day.day, 23, 59, 50, tzinfo=SYDNEY)
    scope = await ledger.reserve_question(uuid4(), now)
    assert scope.day == day
    scope.used["anthropic"] = Decimal("0.001")
    # Settlement reads the row's day, regardless of the wall clock when it runs.
    await scope.settle()
    spent = await rows(ledger, "SELECT day, spent_usd FROM spend_daily")
    assert spent == [(day, Decimal("0.001"))]


async def test_topup_grows_the_reservation(ledger: Ledger) -> None:
    scope = await ledger.reserve_question(uuid4())
    worst = cost_usd(MODELS["anthropic"], {"input_tokens": 120000, "output_tokens": 1024})
    assert worst > scope.reserved["anthropic"]
    await scope.ensure("anthropic", worst)
    assert scope.reserved["anthropic"] == worst
    assert scope.pending["anthropic"] == worst
    stored = await rows(
        ledger,
        "SELECT usd FROM spend_reservations WHERE id = %s AND provider = 'anthropic'",
        (scope.question_id,),
    )
    assert stored == [(worst,)]
    scope.finish("anthropic", worst, Decimal("0.001"))
    assert scope.pending["anthropic"] == 0
    assert scope.total_usd() == Decimal("0.001")
    assert scope.providers() == ["anthropic"]


@pytest.mark.parametrize("period", ["day", "month"])
async def test_failed_topup_leaves_memory_and_database_unchanged(
    ledger: Ledger, period: str
) -> None:
    scope = await ledger.reserve_question(uuid4())
    original = scope.reserved.copy()
    if period == "day":
        ledger.limits = SpendLimits(original, ledger.limits.monthly, MODELS)
    else:
        monthly = {p: amount / MONTHLY_FRACTION for p, amount in original.items()}
        ledger.limits = SpendLimits(ledger.limits.daily, monthly, MODELS)
    with pytest.raises(AIPaused) as caught:
        await scope.ensure("anthropic", Decimal("0.1"))
    assert caught.value.kind == period
    assert scope.reserved == original
    assert scope.pending["anthropic"] == 0
    assert scope.called == set()
    stored = await rows(
        ledger,
        "SELECT usd FROM spend_reservations WHERE id = %s AND provider = 'anthropic'",
        (scope.question_id,),
    )
    assert stored == [(original["anthropic"],)]


async def test_topup_of_expired_row_fails_and_rolls_back_booking(ledger: Ledger) -> None:
    now = datetime.now(UTC)
    scope = await ledger.reserve_question(uuid4(), now - timedelta(seconds=200))
    with pytest.raises(AIPaused) as caught:
        await scope.ensure("anthropic", Decimal("0.1"))
    assert caught.value.kind == "day"
    assert len(await rows(ledger, "SELECT * FROM spend_reservations")) == 3
    assert await rows(ledger, "SELECT * FROM spend_daily") == []


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_finally_settles_real_spend_once(ledger: Ledger, outcome: str) -> None:
    ledger.limits = limits(estimate=0)
    ledger.max_output_tokens = 0
    scope = await ledger.reserve_question(uuid4())
    await scope.ensure("anthropic", Decimal("0.01"))

    async def question() -> None:
        try:
            scope.finish("anthropic", Decimal("0.01"), Decimal("0.001"))
            if outcome == "error":
                raise RuntimeError("question failed")
            if outcome == "cancel":
                asyncio.current_task().cancel()
                await asyncio.sleep(0)
        finally:
            await scope.settle()

    task = asyncio.create_task(question())
    if outcome == "success":
        await task
    else:
        with pytest.raises(RuntimeError if outcome == "error" else asyncio.CancelledError):
            await task
    await scope.settle()
    assert scope.settled
    assert await rows(ledger, "SELECT spent_usd FROM spend_daily") == [(Decimal("0.001"),)]
    assert await rows(ledger, "SELECT * FROM spend_reservations") == []


async def test_settlement_is_shielded_from_caller_cancellation(ledger: Ledger) -> None:
    scope = await ledger.reserve_question(uuid4())
    scope.used["anthropic"] = Decimal("0.001")
    entered, release = asyncio.Event(), asyncio.Event()
    original = scope._settle

    async def delayed(charges: dict[str, Decimal]) -> None:
        entered.set()
        await release.wait()
        await original(charges)

    scope._settle = delayed
    caller = asyncio.create_task(scope.settle())
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release.set()
    await scope.settle()
    assert await rows(ledger, "SELECT spent_usd FROM spend_daily") == [(Decimal("0.001"),)]


async def test_settle_charges_pending_calls_at_worst(ledger: Ledger) -> None:
    scope = await ledger.reserve_question(uuid4())
    await scope.ensure("anthropic", Decimal("0.001"))
    await scope.settle()
    assert await rows(ledger, "SELECT spent_usd FROM spend_daily") == [(Decimal("0.001"),)]


async def test_already_booked_reservation_is_not_charged_again(ledger: Ledger) -> None:
    now = datetime.now(UTC)
    scope = await ledger.reserve_question(uuid4(), now - timedelta(seconds=200))
    await ledger.reserve_question(uuid4(), now)
    before = await rows(ledger, "SELECT provider, spent_usd FROM spend_daily ORDER BY provider")
    scope.used["anthropic"] = Decimal("0.001")
    await scope.settle()
    after = await rows(
        ledger,
        "SELECT provider, spent_usd FROM spend_daily ORDER BY provider",
    )
    assert after == before


async def test_live_pause_overrides_other_limits(ledger: Ledger) -> None:
    now = datetime.now(UTC)
    await ledger.pause_provider("anthropic", now)
    assert await ledger.paused(now) == "month"
    pauses = await rows(ledger, "SELECT until, reason FROM ai_pause")
    assert pauses == [(next_month_start(now), "provider budget error")]
    with pytest.raises(AIPaused) as caught:
        await ledger.reserve_question(uuid4(), now)
    assert caught.value.kind == "month"
    assert await rows(ledger, "SELECT * FROM spend_reservations") == []


async def test_paused_checks_month_before_day_and_does_not_write(ledger: Ledger) -> None:
    now = datetime.now(UTC)
    ledger.limits = limits(daily="0.15")
    await add_spent(ledger, sydney_day(now), "anthropic", "0.149")
    assert await ledger.paused(now) == "day"
    await add_spent(ledger, sydney_day(now), "google", "80")
    assert await ledger.paused(now) == "month"
    assert await rows(ledger, "SELECT * FROM spend_reservations") == []


async def test_december_in_sydney_ignores_november_spending(ledger: Ledger) -> None:
    now = datetime(2026, 11, 30, 13, 30, tzinfo=UTC)
    await add_spent(ledger, date(2026, 11, 30), "anthropic", "100")
    assert await ledger.paused(now) is None
    scope = await ledger.reserve_question(uuid4(), now)
    assert scope.day == date(2026, 12, 1)


async def test_paused_counts_active_reservations_but_not_expired(ledger: Ledger) -> None:
    now = datetime.now(UTC)
    amounts = ledger.question_worst()
    ledger.limits = SpendLimits(amounts, ledger.limits.monthly, MODELS)
    scope = await ledger.reserve_question(uuid4(), now)
    assert await ledger.paused(now) == "day"
    assert await ledger.paused(now + timedelta(seconds=151)) is None
    # paused is read-only, including when it sees expired rows.
    reservations = await rows(
        ledger,
        "SELECT * FROM spend_reservations WHERE id = %s",
        (scope.question_id,),
    )
    assert len(reservations) == 3


async def test_unconfigured_limits_fail_closed_once_per_process(monkeypatch, caplog) -> None:
    monkeypatch.setattr("app.guard.ledger._warned_unconfigured", False)
    incomplete = SpendLimits(dict.fromkeys(PROVIDERS), dict.fromkeys(PROVIDERS), MODELS)
    ledger = Ledger(None, incomplete, 1024)
    with caplog.at_level(logging.WARNING):
        assert await ledger.paused(datetime.now(UTC)) == "day"
        assert await Ledger(None, incomplete, 1024).paused(datetime.now(UTC)) == "day"
        with pytest.raises(AIPaused) as caught:
            await ledger.reserve_question(uuid4())
    assert caught.value.kind == "day"
    assert len(caplog.records) == 1
