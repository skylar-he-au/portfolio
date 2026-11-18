"""`POST /api/ask` with the real compiled graph on fake models, retriever and embedder.

Most tests drive `AskService.stream` directly with in-memory conversations and checkpoints;
the SSE wire format, validation and keep-alive go through the HTTP endpoint; conversation
ids, the Postgres checkpointer and the lazy pools use the test database.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

import anyio
import psycopg
import pytest
from fakes import FakeEmbedder
from fastapi.testclient import TestClient
from guard_helpers import AdmittingLimits, NoSpendLedger, guard_headers
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde import jsonplus
from pydantic import SecretStr
from test_graph import (
    EXAMPLE_Q,
    GOOD_ANSWER,
    MATCH,
    TESTS,
    WORK_RIGHTS,
    FakeChat,
    FakeRetriever,
    Slow,
    check_reply,
    private_check_reply,
    rewrite_reply,
)

from app.api.ask import AskService
from app.config import Settings
from app.db import Database, make_pool
from app.main import create_app
from app.migrate import migrate
from app.qa.graph import GraphDeps
from app.qa.memory import Checkpoints, Conversations, checkpoint_serde, make_checkpoint_pool
from app.qa.models import ChatProvider

pytestmark = pytest.mark.anyio

Event = tuple[str, dict[str, Any]]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- fakes and helpers -------------------------------------------------------------------


class MemoryConversations:
    def __init__(self) -> None:
        self.known: set[str] = set()
        self.owners: dict[str, uuid.UUID] = {}

    async def resolve(self, requested: str | None) -> str:
        if requested in self.known:
            return requested  # type: ignore[return-value]
        new = str(uuid.uuid4())
        self.known.add(new)
        return new

    async def acquire(self, conversation_id: str, owner: uuid.UUID) -> bool:
        if conversation_id in self.owners:
            return False
        self.owners[conversation_id] = owner
        return True

    async def release(self, conversation_id: str, owner: uuid.UUID) -> None:
        if self.owners.get(conversation_id) == owner:
            del self.owners[conversation_id]


class MemoryCheckpoints:
    def __init__(self) -> None:
        self.inner = InMemorySaver(serde=checkpoint_serde())

    async def saver(self) -> InMemorySaver:
        return self.inner

    async def close(self) -> None:
        pass


class Models:
    """Scripted Claude and Gemini shared by every turn of one test."""

    def __init__(self, claude: list[Any], gemini: list[Any]) -> None:
        self.claude = FakeChat(script=list(claude))
        self.gemini = FakeChat(script=list(gemini))


def make_deps(
    models: Models,
    retriever: FakeRetriever,
    *,
    embedder: FakeEmbedder | None = None,
    call_timeout: float = 5.0,
) -> GraphDeps:
    return GraphDeps(
        ChatProvider("anthropic", "claude-haiku-5-5", models.claude),
        ChatProvider("google", "gemini-2.5-flash-lite", models.gemini),
        retriever,
        embed_model="text-embedding-3-small",
        call_timeout=call_timeout,
        embedder=embedder or FakeEmbedder(),
    )


def make_service(
    models: Models,
    retriever: FakeRetriever,
    *,
    timeout_s: float = 10.0,
    embedder: FakeEmbedder | None = None,
) -> AskService:
    deps = make_deps(models, retriever, embedder=embedder)
    return AskService(MemoryConversations(), MemoryCheckpoints(), lambda: deps, timeout_s=timeout_s)


async def collect(service: AskService, question: str, cid: str | None = None) -> list[Event]:
    return [event async for event in service.stream(question, cid)]


def names(events: list[Event]) -> list[str]:
    """Event names without `status`, consecutive tokens collapsed into one."""
    out: list[str] = []
    for name, _ in events:
        if name == "status" or (name == "token" and out and out[-1] == "token"):
            continue
        out.append(name)
    return out


def one(events: list[Event], name: str) -> dict[str, Any]:
    found = [d for n, d in events if n == name]
    assert len(found) == 1, (name, events)
    return found[0]


async def stored_turns(service: AskService, cid: str) -> list[dict[str, Any]]:
    graph = await service._get_graph()
    state = await graph.aget_state({"configurable": {"thread_id": cid}})
    return state.values.get("turns") or []


def general_turn(answer: str = GOOD_ANSWER) -> tuple[list[Any], list[Any]]:
    return [rewrite_reply(), answer], [check_reply()]


def private_turn() -> tuple[list[Any], list[Any]]:
    claude = [rewrite_reply("work rights", private=True), "I can work in Australia [1]."]
    return claude, [private_check_reply(enough_info=True)]


def parse_sse(body: str) -> tuple[list[Event], int]:
    """(events, keep-alive comments) from a raw `text/event-stream` body."""
    import json

    events: list[Event] = []
    pings = 0
    for block in body.split("\n\n"):
        lines = [line for line in block.split("\n") if line]
        if not lines:
            continue
        if all(line.startswith(":") for line in lines):
            pings += 1
            continue
        fields = dict(line.split(": ", 1) for line in lines if not line.startswith(":"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events, pings


def settings(url: str) -> Settings:
    return Settings(_env_file=None, database_url=SecretStr(url), app_env="dev")  # type: ignore[arg-type]


# --- event order -------------------------------------------------------------------------


async def test_general_passed_check_reason_is_none():
    events = await collect(make_service(Models(*general_turn()), FakeRetriever([MATCH])), EXAMPLE_Q)
    check = one(events, "check")
    assert check["passed"] is True
    assert check["reason"] is None


async def test_general_question_event_order():
    models = Models(*general_turn())
    events = await collect(make_service(models, FakeRetriever([MATCH, TESTS])), EXAMPLE_Q)

    assert names(events) == ["meta", "route", "token", "answer", "check", "trace", "done"]
    assert events[0][0] == "meta"
    # recall runs first, then rewrite; `route` comes when rewrite ends.
    assert [(n, d.get("node"), d.get("state")) for n, d in events[1:6]] == [
        ("status", "recall", "start"),
        ("status", "recall", "end"),
        ("status", "rewrite", "start"),
        ("status", "rewrite", "end"),
        ("route", None, None),
    ]
    tokens = "".join(d["text"] for n, d in events if n == "token")
    assert tokens == GOOD_ANSWER
    meta = one(events, "meta")
    uuid.UUID(meta["conversation_id"])
    assert len(meta["turn_id"]) == 32
    assert one(events, "check") == {
        "passed": True,
        "unsupported": [],
        "retried": False,
        "reason": None,
    }
    trace = one(events, "trace")
    assert [n["name"] for n in trace["nodes"]] == [
        "recall",
        "rewrite",
        "retrieve",
        "answer",
        "check",
    ]
    assert trace["nodes"][0]["provider"] == "openai"
    assert trace["memory"] == []
    assert events[-1] == ("done", {})


async def test_private_question_has_no_tokens():
    models = Models(*private_turn())
    events = await collect(make_service(models, FakeRetriever([WORK_RIGHTS])), "Can you work?")
    assert names(events) == ["meta", "route", "answer", "check", "trace", "done"]
    assert one(events, "answer")["citations"][0]["url"] == "/about#work-rights"


async def test_sse_wire_format_through_the_endpoint():
    models = Models(*general_turn())
    service = make_service(models, FakeRetriever([MATCH, TESTS]))
    app = create_app(
        settings("postgresql://nobody@127.0.0.1:1/none"),
        ask_service=service,
        ledger=NoSpendLedger(),
        limits=AdmittingLimits(),
    )
    with TestClient(app, headers=guard_headers(app.state.settings)) as client:
        resp = client.post("/api/ask", json={"question": EXAMPLE_Q})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events, _ = parse_sse(resp.text)
    assert names(events) == ["meta", "route", "token", "answer", "check", "trace", "done"]
    assert one(events, "answer")["text"] == GOOD_ANSWER


# --- one conversation, several turns -----------------------------------------------------


async def test_turns_in_one_conversation_keep_their_own_results_and_memory():
    c1, g1 = private_turn()
    c2, g2 = general_turn()
    # Turn 3: both providers fail on rewrite.
    c3, g3 = [RuntimeError("claude down")], [RuntimeError("gemini down")]
    c4, g4 = general_turn("Every prompt has a version [1].")
    models = Models(c1 + c2 + c3 + c4, g1 + g2 + g3 + g4)
    retriever = FakeRetriever([WORK_RIGHTS], [MATCH, TESTS], [TESTS])
    service = make_service(models, retriever)

    first = await collect(service, "Can you work in Australia?")
    cid = one(first, "meta")["conversation_id"]
    second = await collect(service, EXAMPLE_Q, cid)
    third = await collect(service, "How is it tested?", cid)
    fourth = await collect(service, "How is it tested?", cid)

    assert {one(e, "meta")["conversation_id"] for e in (second, third, fourth)} == {cid}
    ids = [one(e, "meta")["turn_id"] for e in (first, second, third, fourth)]
    assert len(set(ids)) == 4

    # Turn 1: private.
    assert one(first, "check") == {
        "passed": True,
        "unsupported": [],
        "retried": False,
        "reason": None,
    }
    assert [c["url"] for c in one(first, "answer")["citations"]] == ["/about#work-rights"]
    assert one(first, "trace")["memory"] == []
    # Turn 2: general, remembers turn 1.
    assert [c["url"] for c in one(second, "answer")["citations"]] == [MATCH.url, TESTS.url]
    assert [s["url"] for s in one(second, "trace")["sections"]] == [MATCH.url, TESTS.url]
    assert [m["turn_id"] for m in one(second, "trace")["memory"]] == [ids[0]]
    # Turn 3: error, nothing after it, not remembered.
    assert third[-1] == (
        "error",
        {
            "code": "unavailable",
            "message": "The assistant is unavailable right now — please try again later "
            "or email me.",
        },
    )
    assert "done" not in names(third) and "trace" not in names(third)
    # Turn 4: its own check, citations and sections; remembers turns 1 and 2 only.
    assert one(fourth, "check") == {
        "passed": True,
        "unsupported": [],
        "retried": False,
        "reason": None,
    }
    assert one(fourth, "answer") == {
        "text": "Every prompt has a version [1].",
        "citations": [{"n": 1, "url": TESTS.url, "heading": TESTS.heading}],
    }
    assert [s["url"] for s in one(fourth, "trace")["sections"]] == [TESTS.url]
    assert [m["turn_id"] for m in one(fourth, "trace")["memory"]] == ids[:2]
    # The rewrite prompt of turn 4 shows both earlier turns.
    rewrite_prompt = str(models.claude.calls[-2][1].content)
    assert "Can you work in Australia?" in rewrite_prompt and EXAMPLE_Q in rewrite_prompt

    turns = await stored_turns(service, cid)
    assert [t["turn_id"] for t in turns] == [ids[0], ids[1], ids[3]]
    assert turns[1]["question"] == EXAMPLE_Q and turns[1]["answer"] == GOOD_ANSWER
    assert all(len(t["embedding"]) == 1536 * 4 for t in turns)


async def test_third_turn_recalls_the_first_two():
    c, g = general_turn()
    models = Models(c * 3, g * 3)
    service = make_service(models, FakeRetriever([MATCH], [MATCH], [MATCH]))
    first = await collect(service, "What is RoleRadar?")
    cid = one(first, "meta")["conversation_id"]
    second = await collect(service, "How does it match jobs?", cid)
    third = await collect(service, "How is it tested?", cid)
    memory = one(third, "trace")["memory"]
    assert [m["turn_id"] for m in memory] == [
        one(first, "meta")["turn_id"],
        one(second, "meta")["turn_id"],
    ]
    assert [m["question"] for m in memory] == ["What is RoleRadar?", "How does it match jobs?"]
    assert all(isinstance(m["score"], float) for m in memory)


# --- conversation ids, busy, deadline, errors ----------------------------------------------


async def test_same_conversation_twice_at_once_is_busy():
    slow_rewrite = Slow(0.5, rewrite_reply())
    c, g = general_turn()
    models = Models([slow_rewrite, c[1], *c], g * 2)
    service = make_service(models, FakeRetriever([MATCH], [MATCH]))

    first = service.stream(EXAMPLE_Q, None)
    name, meta = await anext(first)
    assert name == "meta"
    busy = await collect(service, "Another question", meta["conversation_id"])
    assert busy == [
        ("error", {"code": "busy", "message": "Please wait for the current answer to finish."})
    ]
    rest = [event async for event in first]
    assert rest[-1] == ("done", {})
    # Free again once the first answer is done.
    again = await collect(service, "How is it tested?", meta["conversation_id"])
    assert again[-1] == ("done", {})


async def test_late_release_of_a_finished_turn_does_not_free_the_next_one():
    c, g = general_turn()
    models = Models([*c, Slow(0.5, c[0]), c[1]], g * 2)
    service = make_service(models, FakeRetriever([MATCH], [MATCH]))
    first = await collect(service, EXAMPLE_Q)
    cid = one(first, "meta")["conversation_id"]
    # Ask again straight after `done`, before the first graph task's own cleanup has run.
    second = service.stream("How is it tested?", cid)
    assert (await anext(second))[0] == "meta"
    for _ in range(5):
        await asyncio.sleep(0)  # let the first task's done-callback run
    assert (await collect(service, "Third", cid))[0][1]["code"] == "busy"
    assert [e async for e in second][-1] == ("done", {})


async def test_graph_error_sends_internal_and_stops():
    class Broken(FakeRetriever):
        async def __call__(self, query: str, *, pages: Any = None) -> list:
            raise RuntimeError("retriever bug")

    models = Models([rewrite_reply()], [])
    events = await collect(make_service(models, Broken()), EXAMPLE_Q)
    assert events[0][0] == "meta"
    assert events[-1] == (
        "error",
        {"code": "internal", "message": "Something went wrong — please try again."},
    )
    assert "done" not in names(events)


async def test_whole_request_deadline_sends_timeout_and_frees_the_conversation():
    models = Models([Slow(5, rewrite_reply())], [Slow(5, rewrite_reply())])
    service = make_service(models, FakeRetriever(), timeout_s=0.3)
    start = time.monotonic()
    events = await collect(service, EXAMPLE_Q)
    assert time.monotonic() - start < 2
    assert events[-1] == (
        "error",
        {"code": "timeout", "message": "That took too long — please try again."},
    )
    assert names(events) == ["meta", "error"]
    with anyio.fail_after(2):  # the cancelled graph task stops; the conversation is free
        await service.wait_idle()
    assert service._conversations.owners == {}
    turns = await stored_turns(service, one(events, "meta")["conversation_id"])
    assert turns == []


async def test_keep_alive_comments_are_sent(monkeypatch):
    import fastapi.routing

    monkeypatch.setattr(fastapi.routing, "_PING_INTERVAL", 0.05)
    c, g = general_turn()
    models = Models([Slow(0.4, c[0]), c[1]], g)
    service = make_service(models, FakeRetriever([MATCH]))
    app = create_app(
        settings("postgresql://nobody@127.0.0.1:1/none"),
        ask_service=service,
        ledger=NoSpendLedger(),
        limits=AdmittingLimits(),
    )
    with TestClient(app, headers=guard_headers(app.state.settings)) as client:
        resp = client.post("/api/ask", json={"question": EXAMPLE_Q})
    events, pings = parse_sse(resp.text)
    assert pings >= 3
    assert ": ping" in resp.text
    assert events[-1] == ("done", {})


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 501},
        {},
        {"question": "ok", "extra": 1},
        {"question": "ok", "conversation_id": "x" * 101},
        {"question": 5},
    ],
)
def test_bad_request_body_is_422(body):
    app = create_app(
        settings("postgresql://nobody@127.0.0.1:1/none"),
        ask_service=make_service(Models([], []), FakeRetriever()),
        ledger=NoSpendLedger(),
        limits=AdmittingLimits(),
    )
    with TestClient(app, headers=guard_headers(app.state.settings)) as client:
        assert client.post("/api/ask", json=body).status_code == 422


def test_question_of_500_characters_is_accepted():
    models = Models(*general_turn())
    service = make_service(models, FakeRetriever([MATCH]))
    app = create_app(
        settings("postgresql://nobody@127.0.0.1:1/none"),
        ask_service=service,
        ledger=NoSpendLedger(),
        limits=AdmittingLimits(),
    )
    with TestClient(app, headers=guard_headers(app.state.settings)) as client:
        resp = client.post("/api/ask", json={"question": "x" * 500})
    assert resp.status_code == 200
    assert parse_sse(resp.text)[0][-1] == ("done", {})


async def test_logs_have_usage_but_no_question_answer_or_ids(caplog):
    question = "How does RoleRadar decide which jobs match a user? zebra-marker-q"
    answer = "RoleRadar ranks jobs by similarity [1]. zebra-marker-a"
    models = Models([rewrite_reply(), answer], [check_reply()])
    with caplog.at_level(logging.DEBUG):
        events = await collect(make_service(models, FakeRetriever([MATCH])), question)
        failed = await collect(
            make_service(Models([RuntimeError(question)], [RuntimeError(answer)]), FakeRetriever()),
            question,
        )
    assert events[-1] == ("done", {}) and failed[-1][0] == "error"
    assert "ask done:" in caplog.text and "ask failed: code=unavailable" in caplog.text
    for secret in ("zebra-marker", one(events, "meta")["conversation_id"], "similarity"):
        assert secret not in caplog.text


# --- database: conversation ids, checkpointer, lazy pools ----------------------------------


def test_startup_and_ping_never_touch_either_pool(refused_url: str):
    from test_api import ExplodingPool

    pool, checkpoint_pool = ExplodingPool(), ExplodingPool()
    with TestClient(
        create_app(settings(refused_url), pool=pool, checkpoint_pool=checkpoint_pool)
    ) as c:
        assert c.get("/api/ping").status_code == 200
    assert pool.touched == [] and checkpoint_pool.touched == []


async def test_unreachable_database_is_unavailable(refused_url: str):
    db = Database(make_pool(refused_url))
    models = Models([], [])
    deps = make_deps(models, FakeRetriever())
    service = AskService(
        Conversations(db), Checkpoints(make_checkpoint_pool(refused_url)), lambda: deps
    )
    try:
        events = await collect(service, EXAMPLE_Q)
    finally:
        await service.close()
        await db.close()
    assert events == [
        (
            "error",
            {
                "code": "unavailable",
                "message": "The assistant is unavailable right now — please try again later "
                "or email me.",
            },
        )
    ]
    assert models.claude.calls == []


def _tables(url: str) -> set[str]:
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
    return {r[0] for r in rows}


def test_postgres_conversations_and_checkpoints(db_url: str, monkeypatch, caplog):
    migrate(db_url)
    c, g = general_turn()
    models = Models(c * 5, g * 5)
    deps = make_deps(models, FakeRetriever([MATCH, TESTS], [TESTS], [MATCH], [MATCH], [MATCH]))
    db = Database(make_pool(db_url))
    service = AskService(Conversations(db), Checkpoints(make_checkpoint_pool(db_url)), lambda: deps)
    app = create_app(settings(db_url), ask_service=service)
    monkeypatch.setattr(jsonplus, "_warned_unregistered_types", set())

    def ask(client: TestClient, cid: str | None) -> list[Event]:
        body: dict[str, Any] = {"question": EXAMPLE_Q}
        if cid is not None:
            body["conversation_id"] = cid
        return parse_sse(client.post("/api/ask", json=body).text)[0]

    with (
        TestClient(app, headers=guard_headers(app.state.settings)) as client,
        caplog.at_level(logging.WARNING),
    ):
        assert client.get("/api/ping").status_code == 200
        assert "checkpoint_migrations" not in _tables(db_url)  # not until the first question

        first = ask(client, None)
        cid = one(first, "meta")["conversation_id"]
        assert "checkpoint_migrations" in _tables(db_url)
        # Known id: same conversation; turn 2 recalls turn 1 from the Postgres checkpoint.
        second = ask(client, cid)
        assert one(second, "meta")["conversation_id"] == cid
        assert [m["turn_id"] for m in one(second, "trace")["memory"]] == [
            one(first, "meta")["turn_id"]
        ]
        # Unknown, malformed and expired ids: a new conversation each.
        unknown = one(ask(client, str(uuid.uuid4())), "meta")["conversation_id"]
        malformed = one(ask(client, "not-a-uuid"), "meta")["conversation_id"]
        with psycopg.connect(db_url) as conn:
            conn.execute(
                "UPDATE conversations SET created_at = now() - interval '25 hours' WHERE id = %s",
                (cid,),
            )
        expired = one(ask(client, cid), "meta")["conversation_id"]
        assert len({cid, unknown, malformed, expired}) == 4
        client.portal.call(db.close)  # type: ignore[union-attr]  # on the app's loop

    # Sections and recalled turns went into the checkpoint and came back without the
    # "unregistered type" warning.
    assert "unregistered" not in caplog.text
    with psycopg.connect(db_url) as conn:
        count = conn.execute("SELECT count(*) FROM conversations").fetchone()
    assert count is not None and count[0] == 4
