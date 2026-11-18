"""Step 5c1: the database conversation lock, spend scope and 30-day question log of `/api/ask`."""

import asyncio
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import psycopg
import pytest
from fastapi.testclient import TestClient
from guard_helpers import AdmittingLimits, NoSpendLedger, guard_headers, guard_settings
from psycopg_pool import PoolTimeout
from test_ask import (
    EXAMPLE_Q,
    MATCH,
    MemoryCheckpoints,
    MemoryConversations,
    Models,
    collect,
    general_turn,
    make_deps,
    one,
)
from test_graph import GOOD_ANSWER, FakeRetriever, Slow, rewrite_reply
from test_guard_ledger import limits
from test_retention import rows

from app.api.ask import AskService
from app.db import Database, make_pool
from app.guard.ledger import AIPaused, Ledger, ProviderBudgetExhausted, current_scope
from app.guard.limits import SYDNEY, Admission
from app.guard.session import REJECTIONS, Rejected
from app.main import create_app
from app.migrate import migrate
from app.qa.memory import Checkpoints, Conversations, make_checkpoint_pool
from app.qa.models import MAX_OUTPUT_TOKENS, ChatProvider
from app.qa.prompts import PROMPT_VERSION
from app.records import Records

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def postgres_service(db: Database, url: str, models: Models, retriever: FakeRetriever):
    deps = make_deps(models, retriever)
    return AskService(Conversations(db), Checkpoints(make_checkpoint_pool(url)), lambda: deps)


async def test_two_services_on_one_database_share_the_conversation_lock(db_url: str):
    """The lock lives in the database, so a second instance sees the first one's question."""
    migrate(db_url)
    c, g = general_turn()
    db = Database(make_pool(db_url))
    first_service = postgres_service(
        db, db_url, Models([Slow(0.5, rewrite_reply()), c[1]], g), FakeRetriever([MATCH])
    )
    second_service = postgres_service(db, db_url, Models(*general_turn()), FakeRetriever([MATCH]))
    try:
        first = first_service.stream(EXAMPLE_Q, None)
        name, meta = await anext(first)
        assert name == "meta"
        busy = await collect(second_service, "Another question", meta["conversation_id"])
        assert busy == [
            ("error", {"code": "busy", "message": "Please wait for the current answer to finish."})
        ]
        rest = [event async for event in first]
        assert rest[-1] == ("done", {})
        assert one(rest, "check") is not None
        again = await collect(second_service, EXAMPLE_Q, meta["conversation_id"])
        assert again[-1] == ("done", {})
    finally:
        await first_service.wait_idle()
        await second_service.wait_idle()
        await first_service.close()
        await second_service.close()
        await db.close()


@pytest.mark.parametrize(
    ("kind", "code", "status"),
    [("day", "limit_day", 429), ("month", "paused_month", 503)],
)
async def test_reservation_rejection_is_json_before_stream(db_url, kind, code, status):
    migrate(db_url)
    ledger = SimpleNamespace(reserve_question=AsyncMock(side_effect=AIPaused(kind)))
    service = AskService(MemoryConversations(), MemoryCheckpoints(), lambda: None, ledger=ledger)
    service.stream = AsyncMock()
    admission = AdmittingLimits()
    settings = guard_settings(db_url)
    app = create_app(settings, ask_service=service, ledger=NoSpendLedger(), limits=admission)
    with TestClient(app, headers=guard_headers(settings)) as client:
        response = client.post("/api/ask", json={"question": EXAMPLE_Q})
    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"code": code, "message": REJECTIONS[code][1]}
    assert admission.calls == 1
    ledger.reserve_question.assert_awaited_once()
    assert isinstance(ledger.reserve_question.call_args.args[0], uuid.UUID)
    service.stream.assert_not_called()
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT count(*) FROM question_log").fetchone() == (0,)


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("secret"),
        PoolTimeout("secret"),
        OSError("secret"),
        TimeoutError("secret"),
    ],
)
async def test_reservation_database_failure_is_unavailable(error):
    ledger = SimpleNamespace(reserve_question=AsyncMock(side_effect=error))
    service = AskService(MemoryConversations(), MemoryCheckpoints(), lambda: None, ledger=ledger)
    with pytest.raises(Rejected, match="unavailable"):
        await service.reserve(uuid.uuid4())


@pytest.mark.parametrize("outcome", ["done", "error", "timeout"])
async def test_question_records_and_settlement_after_every_outcome(db_url, outcome):
    migrate(db_url)
    db = Database(make_pool(db_url))
    ledger = Ledger(db, limits(), MAX_OUTPUT_TOKENS)
    checkpoints = Checkpoints(make_checkpoint_pool(db_url))
    if outcome == "done":
        models = Models(*general_turn())
    elif outcome == "error":
        models = Models([RuntimeError("secret-question")], [RuntimeError("secret-answer")])
    else:
        models = Models([Slow(5, rewrite_reply())], [])
    deps = make_deps(models, FakeRetriever([MATCH]))
    # The metered path needs a model with a configured price.
    deps = replace(deps, gemini=ChatProvider("google", "gemini-3.5-flash-lite", models.gemini))
    service = AskService(
        Conversations(db),
        checkpoints,
        lambda: deps,
        records=Records(db, checkpoints),
        ledger=ledger,
    )
    turn_id = uuid.uuid4()
    admission = Admission("ask", b"secret-token", b"daily-ip-hash", datetime.now(UTC).date())
    scope = await service.reserve(turn_id)
    assert scope is not None
    finish_settle = asyncio.Event()
    entered_settle = asyncio.Event()
    original_settle = scope.settle

    async def delayed_settle():
        entered_settle.set()
        await finish_settle.wait()
        await original_settle()

    try:
        assert len(await rows(db, "SELECT * FROM spend_reservations")) == 3
        if outcome == "timeout":
            # Set the deadline after graph setup, and hold settlement to prove that
            # the response returns while its task still owns finalisation.
            await service._get_graph()
            service.timeout_s = 0.3
            scope.settle = delayed_settle
        events = [
            event
            async for event in service.stream(
                EXAMPLE_Q, None, admission, scope=scope, turn_id=turn_id
            )
        ]
        meta = one(events, "meta")
        assert uuid.UUID(meta["turn_id"]) == turn_id
        if outcome == "done":
            assert events[-1] == ("done", {})
        else:
            assert one(events, "error")["code"] == (
                "timeout" if outcome == "timeout" else "unavailable"
            )
        if outcome == "timeout":
            await asyncio.wait_for(entered_settle.wait(), 2)
            assert await rows(db, "SELECT id FROM question_log") == []
            assert not scope.settled
            assert service._tasks
            finish_settle.set()
        await service.wait_idle()
        assert scope.settled
        assert current_scope.get() is None  # the scope belongs to the graph task
        records = await rows(db, "SELECT * FROM question_log")
        assert len(records) == 1
        columns = await rows(
            db,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'question_log' ORDER BY ordinal_position",
        )
        assert [r[0] for r in columns] == [
            "id",
            "created_at",
            "question",
            "answer_language",
            "prompt_version",
            "providers",
            "cost_usd",
            "ip_hash",
            "thumbs_down",
        ]
        row = dict(zip((r[0] for r in columns), records[0], strict=True))
        assert row["id"] == turn_id
        assert row["question"] == EXAMPLE_Q
        assert row["answer_language"] == ("en" if outcome == "done" else None)
        assert row["prompt_version"] == PROMPT_VERSION
        assert row["providers"] == scope.providers()
        expected_providers = ["anthropic"] if outcome == "timeout" else ["anthropic", "google"]
        assert row["providers"] == expected_providers
        assert row["cost_usd"] == scope.total_usd().quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )
        assert row["cost_usd"] > 0
        assert bytes(row["ip_hash"]) == admission.ip_hash
        assert row["thumbs_down"] is False
        for secret in ("secret-token", "192.0.2.1", GOOD_ANSWER, "secret-answer"):
            assert secret not in repr(records)
        assert await rows(db, "SELECT * FROM spend_reservations") == []
        spent = await rows(db, "SELECT provider, spent_usd FROM spend_daily ORDER BY provider")
        assert spent == [
            (provider, amount.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
            for provider, amount in scope.used.items()
            if amount
        ]
        next_owner = uuid.uuid4()
        assert await Conversations(db).acquire(meta["conversation_id"], next_owner)
        await Conversations(db).release(meta["conversation_id"], next_owner)
    finally:
        finish_settle.set()
        await service.wait_idle()
        await service.close()
        await db.close()


class ErrorGraph:
    def __init__(self, error):
        self.error = error

    async def astream(self, *args, **kwargs):
        # A model failure forwarded by the graph, without a real provider call.
        yield {"type": "custom", "data": {"event": "status", "data": {}}}
        raise self.error

    async def aget_state(self, config):
        return SimpleNamespace(values={})


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ProviderBudgetExhausted("anthropic"), "paused_month"),
        (AIPaused("day"), "limit_day"),
        (AIPaused("month"), "paused_month"),
    ],
)
async def test_budget_errors_have_the_guard_message(error, code):
    service = AskService(MemoryConversations(), MemoryCheckpoints(), lambda: None)
    service._graph = ErrorGraph(error)
    events = await collect(service, EXAMPLE_Q)
    await service.wait_idle()
    assert events[-1] == ("error", {"code": code, "message": REJECTIONS[code][1]})


async def test_busy_and_setup_failure_settle_without_a_record():
    conversations = MemoryConversations()
    records = SimpleNamespace(write=AsyncMock())
    service = AskService(conversations, MemoryCheckpoints(), lambda: None, records=records)
    service._graph = ErrorGraph(RuntimeError("secret"))
    cid = await conversations.resolve(None)
    assert await conversations.acquire(cid, uuid.uuid4())
    admission = Admission("ask", b"token", b"hash", datetime.now(SYDNEY).date())
    for conversation in (cid, None):
        scope = SimpleNamespace(settle=AsyncMock())
        if conversation is None:
            conversations.resolve = AsyncMock(side_effect=psycopg.OperationalError("secret"))
        events = [
            event
            async for event in service.stream(EXAMPLE_Q, conversation, admission, scope=scope)
        ]
        assert events[0][0] == "error"
        assert events[0][1]["code"] == ("busy" if conversation else "unavailable")
        scope.settle.assert_awaited_once()
    records.write.assert_not_called()


async def test_closing_at_meta_releases_lock_and_reservation_without_record():
    conversations = MemoryConversations()
    records = SimpleNamespace(write=AsyncMock())
    service = AskService(conversations, MemoryCheckpoints(), lambda: None, records=records)
    service._graph = ErrorGraph(RuntimeError("unused"))
    scope = SimpleNamespace(settle=AsyncMock())
    stream = service.stream(EXAMPLE_Q, None, scope=scope)
    _, meta = await anext(stream)
    await stream.aclose()
    assert conversations.owners == {}
    assert meta["conversation_id"] in conversations.known
    assert not service._tasks
    scope.settle.assert_awaited_once()
    records.write.assert_not_called()


async def test_finalisation_failures_do_not_escape_or_log_contents(caplog):
    secret = "question-answer-conversation-token-ip-secret"
    conversations = MemoryConversations()
    conversations.release = AsyncMock(side_effect=RuntimeError(secret))
    records = SimpleNamespace(write=AsyncMock(side_effect=RuntimeError(secret)))
    scope = SimpleNamespace(
        settle=AsyncMock(side_effect=RuntimeError(secret)),
        providers=lambda: [],
        total_usd=lambda: Decimal(0),
    )
    graph = ErrorGraph(RuntimeError(secret))
    graph.aget_state = AsyncMock(side_effect=RuntimeError(secret))
    service = AskService(conversations, MemoryCheckpoints(), lambda: None, records=records)
    service._graph = graph
    admission = Admission("ask", b"token", b"hash", datetime.now(SYDNEY).date())
    events = [event async for event in service.stream(EXAMPLE_Q, None, admission, scope=scope)]
    tasks = list(service._tasks)
    assert await asyncio.gather(*tasks, return_exceptions=True) == [None] * len(tasks)
    await service.wait_idle()
    assert events[-1][1]["code"] == "internal"
    assert conversations.release.await_count == 2
    scope.settle.assert_awaited_once()
    records.write.assert_awaited_once()
    assert records.write.call_args.args[0].answer_language is None
    assert secret not in caplog.text
    assert "RuntimeError" in caplog.text


async def test_provider_budget_error_from_model_pauses_and_records(db_url):
    migrate(db_url)
    db = Database(make_pool(db_url))
    ledger = Ledger(db, limits(), MAX_OUTPUT_TOKENS)
    checkpoints = Checkpoints(make_checkpoint_pool(db_url))

    class OutOfCredit(Exception):
        status_code = 400

    models = Models([OutOfCredit("credit balance is too low")], [])
    deps = make_deps(models, FakeRetriever())
    service = AskService(
        Conversations(db),
        checkpoints,
        lambda: deps,
        records=Records(db, checkpoints),
        ledger=ledger,
    )
    try:
        turn_id = uuid.uuid4()
        scope = await service.reserve(turn_id)
        admission = Admission("ask", b"token", b"hash", datetime.now(SYDNEY).date())
        events = [
            event
            async for event in service.stream(
                EXAMPLE_Q, None, admission, scope=scope, turn_id=turn_id
            )
        ]
        await service.wait_idle()
        assert events[-1] == (
            "error",
            {"code": "paused_month", "message": REJECTIONS["paused_month"][1]},
        )
        pauses = await rows(db, "SELECT provider FROM ai_pause WHERE until > now()")
        assert pauses == [("anthropic",)]
        assert await rows(db, "SELECT id FROM question_log") == [(turn_id,)]
        assert await rows(db, "SELECT * FROM spend_reservations") == []
        assert models.gemini.calls == []
    finally:
        await service.wait_idle()
        await service.close()
        await db.close()


@pytest.mark.parametrize("ending", ["disconnect", "shutdown"])
async def test_cancelled_task_still_settles_and_records(ending):
    class HangingGraph(ErrorGraph):
        async def astream(self, *args, **kwargs):
            yield {"type": "custom", "data": {"event": "status", "data": {}}}
            await asyncio.Event().wait()

    conversations = MemoryConversations()
    records = SimpleNamespace(write=AsyncMock())
    scope = SimpleNamespace(settle=AsyncMock(), providers=lambda: [], total_usd=lambda: Decimal(0))
    service = AskService(conversations, MemoryCheckpoints(), lambda: None, records=records)
    service._graph = HangingGraph(None)
    admission = Admission("ask", b"token", b"hash", datetime.now(SYDNEY).date())
    stream = service.stream(EXAMPLE_Q, None, admission, scope=scope)
    _, meta = await anext(stream)
    assert (await anext(stream))[0] == "status"
    if ending == "disconnect":
        await stream.aclose()
    else:
        await service.close()
    await service.wait_idle()
    await stream.aclose()
    assert conversations.owners == {}
    scope.settle.assert_awaited_once()
    records.write.assert_awaited_once()
    record = records.write.call_args.args[0]
    assert record.id == uuid.UUID(meta["turn_id"])
    assert record.answer_language is None
    assert record.ip_hash == admission.ip_hash


async def test_record_without_ledger_has_no_spend():
    records = SimpleNamespace(write=AsyncMock())
    service = AskService(MemoryConversations(), MemoryCheckpoints(), lambda: None, records=records)
    service._graph = ErrorGraph(RuntimeError("failed"))
    assert await service.reserve(uuid.uuid4()) is None
    admission = Admission("ask", b"token", b"hash", datetime.now(SYDNEY).date())
    events = [event async for event in service.stream(EXAMPLE_Q, None, admission)]
    await service.wait_idle()
    record = records.write.call_args.args[0]
    assert record.id == uuid.UUID(one(events, "meta")["turn_id"])
    assert record.providers == []
    assert record.cost_usd == Decimal(0)
