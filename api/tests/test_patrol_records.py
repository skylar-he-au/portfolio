"""Patrol APIs settle real spending without writing visitor records."""

import time
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from guard_helpers import guard_headers
from test_ask import (
    EXAMPLE_Q,
    MATCH,
    MemoryCheckpoints,
    MemoryConversations,
    Models,
    general_turn,
    make_deps,
)
from test_graph import FakeRetriever
from test_match_api import JD, Script, events_of, parse_sse, post
from test_match_api import make_world as make_world

from app.api.ask import AskService
from app.guard.ledger import AIPaused
from app.guard.session import sign_token
from app.main import create_app
from app.qa.models import ChatProvider
from app.records import Records

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def token_for(world, patrol):
    secret = world.settings.session_secret.get_secret_value()
    return sign_token(secret, time.time(), patrol=patrol)[0]


@pytest.mark.parametrize("patrol", [False, True])
async def test_ask_record_policy_keeps_reservation_and_spend(make_world, patrol):
    world = make_world()
    models = Models(*general_turn())
    deps = make_deps(models, FakeRetriever([MATCH]))
    deps = replace(deps, gemini=ChatProvider("google", "gemini-3.5-flash-lite", models.gemini))
    checkpoints = MemoryCheckpoints()
    records = Records(world.db, checkpoints)
    records.write = AsyncMock(wraps=records.write)
    service = AskService(
        MemoryConversations(), checkpoints, lambda: deps, records=records, ledger=world.ledger
    )
    app = create_app(
        world.settings,
        ask_service=service,
        match_service=world.service,
        limits=world.limits,
        ledger=world.ledger,
    )
    original = world.ledger.reserve_question
    reservations = []

    async def reserve(turn_id):
        scope = await original(turn_id)
        reservations.append(world.rows("SELECT provider FROM spend_reservations"))
        return scope

    world.ledger.reserve_question = reserve
    headers = guard_headers(world.settings, token=token_for(world, patrol))
    # Only /api/session pays attention to this header.
    headers["X-Patrol"] = "ignored"
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
        ) as client:
            response = await client.post("/api/ask", json={"question": EXAMPLE_Q})
        await service.wait_idle()
        assert response.status_code == 200
        assert parse_sse(response.text)[-1] == ("done", {})
        assert reservations and len(reservations[0]) == 3
        assert records.write.await_count == (0 if patrol else 1)
        assert world.rows("SELECT count(*) FROM question_log") == [(0 if patrol else 1,)]
        assert world.rows("SELECT * FROM match_log") == []
        assert world.rows("SELECT * FROM spend_reservations") == []
        spent = world.rows("SELECT provider, spent_usd FROM spend_daily")
        assert sum(cost for _, cost in spent) > 0
        assert world.counts("patrol" if patrol else "ask") == (
            {"site": 1} if patrol else {"token": 1, "ip": 1, "site": 1}
        )
    finally:
        await service.wait_idle()
        await service.close()


@pytest.mark.parametrize("patrol", [False, True])
async def test_match_record_policy_keeps_reservation_and_spend(make_world, patrol):
    world = make_world()
    records = world.service._records
    records.write = AsyncMock(wraps=records.write)
    original = world.ledger.reserve
    reservations = []

    async def reserve(kind, turn_id, *, conn=None):
        scope = await original(kind, turn_id, conn=conn)
        cursor = await conn.execute("SELECT provider FROM spend_reservations")
        reservations.append(await cursor.fetchall())
        return scope

    world.ledger.reserve = reserve
    events = await events_of(world, JD, token=token_for(world, patrol))
    assert events[-1] == ("done", {})
    assert reservations and len(reservations[0]) == 3
    assert records.write.await_count == (0 if patrol else 1)
    assert world.rows("SELECT count(*) FROM match_log") == [(0 if patrol else 1,)]
    assert world.rows("SELECT * FROM question_log") == []
    assert world.rows("SELECT * FROM spend_reservations") == []
    assert sum(cost for _, cost in world.rows("SELECT provider, spent_usd FROM spend_daily")) > 0
    assert world.counts("patrol" if patrol else "match") == (
        {"site": 1} if patrol else {"token": 1, "ip": 1, "site": 1}
    )


async def test_patrol_not_a_job_refunds_count_but_records_spend(make_world):
    world = make_world(script=Script(requirements=[]), patrol_limit_day=1)
    token = token_for(world, True)
    events = await events_of(world, "What's the weather like?", token=token)
    assert events[-1][1]["code"] == "not_a_job"
    assert world.counts("patrol") == {"site": 0}
    assert world.rows("SELECT * FROM match_log") == []
    assert world.rows("SELECT * FROM spend_reservations") == []
    assert sum(cost for _, cost in world.rows("SELECT provider, spent_usd FROM spend_daily")) > 0
    world.script.requirements = ["Python"]
    assert (await events_of(world, JD, token=token))[-1] == ("done", {})
    assert world.counts("patrol") == {"site": 1}


async def test_patrol_match_limit_returns_http_429(make_world):
    world = make_world(patrol_limit_day=1)
    assert (await events_of(world, JD, token=token_for(world, True)))[-1] == ("done", {})
    response = await post(world, JD, token=token_for(world, True))
    assert response.status_code == 429 and response.json()["code"] == "limit_day"


async def test_patrol_still_obeys_daily_spend_cap(make_world, monkeypatch):
    world = make_world()

    async def refuse(*args, **kwargs):
        raise AIPaused("day")

    monkeypatch.setattr(world.ledger, "reserve", refuse)
    events = await events_of(world, JD, token=token_for(world, True))
    assert events[-1][1]["code"] == "limit_day"
    assert world.counts("patrol") == {}
    assert world.rows("SELECT * FROM spend_reservations") == []
