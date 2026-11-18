"""Authenticated scheduled cleanup and its bounded, independent retention steps."""

import json
import threading
import time
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock

import psycopg
import pytest
from fastapi.testclient import TestClient
from guard_helpers import guard_settings
from langgraph.checkpoint.base import empty_checkpoint
from test_api import ExplodingPool

from app import cleanup
from app.cleanup import CleanupResult, run_cleanup
from app.db import Database, make_pool
from app.main import ORIGIN_EXEMPT_PATHS, create_app
from app.migrate import migrate
from app.qa.memory import Checkpoints, make_checkpoint_pool

AUDIENCE = "https://api.example.com/api/cleanup"
EMAIL = "scheduler@example.iam.gserviceaccount.com"
HEADERS = {"Authorization": "Bearer google-id-token"}
COUNTS = dict.fromkeys(("questions", "conversations", "daily_keys", "counters", "reservations"), 0)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def settings(url, **overrides):
    return guard_settings(
        url, **({"cleanup_audience": AUDIENCE, "cleanup_invoker_email": EMAIL} | overrides)
    )


def claims(**overrides):
    return {
        "aud": AUDIENCE,
        "email": EMAIL,
        "iss": "https://accounts.google.com",
        "email_verified": True,
    } | overrides


@pytest.mark.parametrize("field", ["cleanup_audience", "cleanup_invoker_email"])
def test_empty_cleanup_settings_are_none(refused_url, field):
    assert getattr(settings(refused_url, **{field: ""}), field) is None


@pytest.mark.parametrize(
    "authorization, returned_claims, error, overrides",
    [
        (None, claims(), None, {}),
        ("Basic token", claims(), None, {}),
        ("Bearer", claims(), None, {}),
        ("Bearer token extra", claims(), None, {}),
        ("Bearer token", claims(), ValueError("private-token"), {}),
        ("Bearer token", claims(aud="wrong"), None, {}),
        ("Bearer token", claims(email="wrong"), None, {}),
        ("Bearer token", claims(email=None), None, {}),
        ("Bearer token", claims(email_verified=False), None, {}),
        ("Bearer token", claims(email_verified=1), None, {}),
        ("Bearer token", claims(iss="wrong"), None, {}),
        ("Bearer token", claims(), None, {"cleanup_audience": None}),
        ("Bearer token", claims(), None, {"cleanup_invoker_email": None}),
        (
            "Bearer token",
            claims(),
            None,
            {"cleanup_audience": None, "cleanup_invoker_email": None},
        ),
    ],
)
def test_cleanup_auth_rejects_before_database(
    refused_url, monkeypatch, authorization, returned_claims, error, overrides
):
    verifier = Mock(return_value=returned_claims, side_effect=error)
    monkeypatch.setattr("app.api.cleanup.id_token.verify_oauth2_token", verifier)
    pool = ExplodingPool()
    checkpoints = AsyncMock()
    app = create_app(settings(refused_url, **overrides), pool=pool, cleanup_checkpoints=checkpoints)
    headers = {"Authorization": authorization} if authorization is not None else {}
    with TestClient(app) as client:
        start = time.monotonic()
        response = client.post("/api/cleanup", headers=headers)
        elapsed = time.monotonic() - start
    assert response.status_code == 401
    assert response.json() == {"code": "auth", "message": "Please try again."}
    assert elapsed < 1
    assert pool.touched == []
    checkpoints.saver.assert_not_awaited()
    checkpoints.close.assert_awaited_once()
    if authorization in (None, "Basic token", "Bearer", "Bearer token extra") or overrides:
        verifier.assert_not_called()


@pytest.mark.parametrize(
    "issuer, verified", [("https://accounts.google.com", True), ("accounts.google.com", "true")]
)
def test_cleanup_accepts_verified_claims_without_origin(refused_url, monkeypatch, issuer, verified):
    verifier_threads = []

    def verify(token, request, *, audience):
        assert token == "google-id-token" and audience == AUDIENCE
        verifier_threads.append(threading.get_ident())
        return claims(iss=issuer, email=EMAIL.upper(), email_verified=verified)

    monkeypatch.setattr("app.api.cleanup.id_token.verify_oauth2_token", verify)
    runner = AsyncMock(return_value=CleanupResult(dict(COUNTS), []))
    monkeypatch.setattr("app.api.cleanup.run_cleanup", runner)
    pool = ExplodingPool()
    checkpoints = AsyncMock()
    app = create_app(settings(refused_url), pool=pool, cleanup_checkpoints=checkpoints)
    assert "/api/cleanup" in ORIGIN_EXEMPT_PATHS
    with TestClient(app) as client:
        event_loop_thread = client.portal.call(threading.get_ident)
        response = client.post("/api/cleanup", headers=HEADERS)
    assert response.status_code == 200 and response.json() == {"deleted": COUNTS}
    assert verifier_threads and verifier_threads[0] != event_loop_thread
    runner.assert_awaited_once_with(app.state.db, checkpoints)
    assert pool.touched == []


def seed_rows(url):
    ids = {
        name: uuid.uuid4() for name in ("old_question", "question", "expired", "running", "recent")
    }
    with psycopg.connect(url) as conn:
        for name, days in (("old_question", 31), ("question", 1)):
            conn.execute(
                "INSERT INTO question_log (id, question, prompt_version, ip_hash, created_at) "
                "VALUES (%s, 'question', 'v-test', %s, now() - %s * interval '1 day')",
                (ids[name], b"hash", days),
            )
        for name, hours in (("expired", 25), ("running", 25), ("recent", 1)):
            conn.execute(
                "INSERT INTO conversations (id, created_at) "
                "VALUES (%s, now() - %s * interval '1 hour')",
                (ids[name], hours),
            )
        conn.execute(
            "UPDATE conversations SET running_until = now() + interval '1 hour' WHERE id = %s",
            (ids["running"],),
        )
        for offset in (-2, -1):
            conn.execute(
                "INSERT INTO daily_keys (day, key) "
                "VALUES ((now() AT TIME ZONE 'Australia/Sydney')::date + %s, %s)",
                (offset, b"key"),
            )
        for offset in (-3, 0):
            conn.execute(
                "INSERT INTO usage_counters (day, kind, scope, key) "
                "VALUES ((now() AT TIME ZONE 'Australia/Sydney')::date + %s, 'ask', 'site', %s)",
                (offset, b""),
            )
        for provider in ("anthropic", "google", "openai"):
            for age in ("2 days", "1 hour"):
                conn.execute(
                    "INSERT INTO spend_reservations (id, provider, day, usd, expires_at) "
                    "VALUES (%s, %s, (now() AT TIME ZONE 'Australia/Sydney')::date - 2, "
                    "%s, now() - %s::interval)",
                    (uuid.uuid4(), provider, Decimal("0.125"), age),
                )
            conn.execute(
                "INSERT INTO spend_daily (day, provider, spent_usd) "
                "VALUES ((now() AT TIME ZONE 'Australia/Sydney')::date - 2, %s, 0.25)",
                (provider,),
            )
        conn.execute(
            "INSERT INTO ai_pause (provider, until, reason) "
            "VALUES ('google', now() - interval '2 days', 'provider budget error')"
        )
    return ids


def assert_rows_kept(url, ids, *, conversations_deleted=True):
    with psycopg.connect(url) as conn:
        assert conn.execute("SELECT id FROM question_log").fetchall() == [(ids["question"],)]
        conversations = {ids["running"], ids["recent"]}
        if not conversations_deleted:
            conversations.add(ids["expired"])
        assert {row[0] for row in conn.execute("SELECT id FROM conversations")} == conversations
        keys = conn.execute(
            "SELECT day - (now() AT TIME ZONE 'Australia/Sydney')::date FROM daily_keys"
        ).fetchall()
        assert keys == [(-1,)]
        counters = conn.execute(
            "SELECT day - (now() AT TIME ZONE 'Australia/Sydney')::date FROM usage_counters"
        ).fetchall()
        assert counters == [(0,)]
        reservations = conn.execute(
            "SELECT provider, expires_at > now() - interval '1 day' "
            "FROM spend_reservations ORDER BY provider"
        ).fetchall()
        assert reservations == [(p, True) for p in ("anthropic", "google", "openai")]
        spending = conn.execute(
            "SELECT provider, spent_usd, day = (now() AT TIME ZONE 'Australia/Sydney')::date - 2 "
            "FROM spend_daily ORDER BY provider"
        ).fetchall()
        assert spending == [(p, Decimal("0.375"), True) for p in ("anthropic", "google", "openai")]
        assert conn.execute("SELECT count(*) FROM ai_pause").fetchone() == (1,)


async def seed_checkpoints(checkpoints, ids):
    saver = await checkpoints.saver()
    configs = []
    for name in ("expired", "running"):
        config = {"configurable": {"thread_id": str(ids[name]), "checkpoint_ns": ""}}
        await saver.aput(config, empty_checkpoint(), {}, {})
        assert await saver.aget_tuple(config) is not None
        configs.append(config)
    return saver, configs


def test_cleanup_real_retention_and_spend_booking(db_url, monkeypatch):
    migrate(db_url)
    ids = seed_rows(db_url)
    monkeypatch.setattr("app.api.cleanup.id_token.verify_oauth2_token", Mock(return_value=claims()))
    checkpoints = Checkpoints(make_checkpoint_pool(db_url))
    with TestClient(create_app(settings(db_url), cleanup_checkpoints=checkpoints)) as client:
        saver, configs = client.portal.call(seed_checkpoints, checkpoints, ids)
        response = client.post("/api/cleanup", headers=HEADERS)
        assert response.status_code == 200
        assert response.json() == {
            "deleted": {
                "questions": 1,
                "conversations": 1,
                "daily_keys": 1,
                "counters": 1,
                "reservations": 3,
            }
        }
        assert client.portal.call(saver.aget_tuple, configs[0]) is None
        assert client.portal.call(saver.aget_tuple, configs[1]) is not None
        assert_rows_kept(db_url, ids)
        assert client.post("/api/cleanup", headers=HEADERS).json() == {"deleted": COUNTS}


def test_cleanup_initializes_missing_checkpoint_tables(db_url, monkeypatch):
    migrate(db_url)
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT to_regclass('checkpoints')").fetchone() == (None,)
        conn.execute(
            "INSERT INTO conversations (id, created_at) VALUES (%s, now() - interval '25 hours')",
            (uuid.uuid4(),),
        )
    monkeypatch.setattr("app.api.cleanup.id_token.verify_oauth2_token", Mock(return_value=claims()))
    with TestClient(create_app(settings(db_url))) as client:
        response = client.post("/api/cleanup", headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"deleted": COUNTS | {"conversations": 1}}
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT to_regclass('checkpoints')").fetchone() == ("checkpoints",)


def test_cleanup_deletes_more_than_one_conversation_batch(db_url, monkeypatch):
    migrate(db_url)
    with psycopg.connect(db_url) as conn:
        conn.execute(
            "INSERT INTO conversations (id, created_at) "
            "SELECT gen_random_uuid(), now() - interval '25 hours' FROM generate_series(1, 1200)"
        )
    monkeypatch.setattr("app.api.cleanup.id_token.verify_oauth2_token", Mock(return_value=claims()))
    with TestClient(create_app(settings(db_url))) as client:
        response = client.post("/api/cleanup", headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"deleted": COUNTS | {"conversations": 1200}}
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT count(*) FROM conversations").fetchone() == (0,)


def test_cleanup_step_failure_still_runs_other_steps(db_url, monkeypatch, caplog):
    migrate(db_url)
    ids = seed_rows(db_url)
    checkpoints = AsyncMock()
    checkpoints.saver.side_effect = RuntimeError("private-row-contents")
    monkeypatch.setattr("app.api.cleanup.id_token.verify_oauth2_token", Mock(return_value=claims()))
    with TestClient(create_app(settings(db_url), cleanup_checkpoints=checkpoints)) as client:
        response = client.post("/api/cleanup", headers=HEADERS)
    assert response.status_code == 500
    assert response.json() == {
        "deleted": {
            "questions": 1,
            "conversations": 0,
            "daily_keys": 1,
            "counters": 1,
            "reservations": 3,
        },
        "failed": ["conversations"],
    }
    assert "cleanup step conversations failed: RuntimeError" in caplog.text
    assert "private-row-contents" not in caplog.text
    assert_rows_kept(db_url, ids, conversations_deleted=False)


@pytest.mark.anyio
async def test_cleanup_batch_cap_and_single_saver(monkeypatch):
    questions = AsyncMock(return_value=500)
    conversations = AsyncMock(return_value=500)
    keys = AsyncMock(return_value=2)
    counters = AsyncMock(return_value=3)
    monkeypatch.setattr(cleanup.retention, "delete_old_questions", questions)
    monkeypatch.setattr(cleanup.retention, "delete_expired_conversations", conversations)
    monkeypatch.setattr(cleanup.retention, "delete_old_daily_keys", keys)
    monkeypatch.setattr(cleanup.retention, "delete_old_counters", counters)
    db = MagicMock()
    conn = db.connection.return_value.__aenter__.return_value = AsyncMock()
    conn.transaction = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    checkpoints = AsyncMock()
    result = await run_cleanup(db, checkpoints)
    assert result.failed == []
    assert result.deleted == {
        "questions": 10_000,
        "conversations": 10_000,
        "daily_keys": 2,
        "counters": 3,
        "reservations": 0,
    }
    assert questions.await_count == conversations.await_count == 20
    assert all(call.kwargs["limit"] == 500 for call in questions.await_args_list)
    assert all(call.kwargs["limit"] == 500 for call in conversations.await_args_list)
    checkpoints.saver.assert_awaited_once()
    keys.assert_awaited_once_with(db)
    counters.assert_awaited_once_with(db)


@pytest.mark.anyio
async def test_cleanup_reservation_booking_is_atomic(db_url, monkeypatch):
    migrate(db_url)
    seed_rows(db_url)
    db = Database(make_pool(db_url))
    checkpoints = Checkpoints(make_checkpoint_pool(db_url))
    original = cleanup._add_spent
    providers = []

    async def fail_on_second_provider(conn, provider, day, usd):
        providers.append(provider)
        if provider == "google":
            raise RuntimeError("private-reservation")
        await original(conn, provider, day, usd)

    monkeypatch.setattr(cleanup, "_add_spent", fail_on_second_provider)
    try:
        result = await run_cleanup(db, checkpoints)
    finally:
        await checkpoints.close()
        await db.close()
    assert result.failed == ["reservations"] and result.deleted["reservations"] == 0
    assert providers == ["anthropic", "google"]
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT count(*) FROM spend_reservations").fetchone() == (6,)
        spending = conn.execute("SELECT spent_usd FROM spend_daily").fetchall()
        assert spending == [(Decimal("0.25"),)] * 3


@pytest.mark.anyio
async def test_cleanup_main_uses_settings_and_prints_json(db_url, monkeypatch, capsys):
    migrate(db_url)
    ids = seed_rows(db_url)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setattr(cleanup, "get_settings", lambda: settings(db_url))
    assert await cleanup.main() == 0
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert json.loads(output) == {
        "deleted": {
            "questions": 1,
            "conversations": 1,
            "daily_keys": 1,
            "counters": 1,
            "reservations": 3,
        },
        "failed": [],
    }
    assert_rows_kept(db_url, ids)


@pytest.mark.anyio
async def test_cleanup_main_failure_exits_one_and_closes_both(refused_url, monkeypatch, capsys):
    monkeypatch.setattr(cleanup, "get_settings", lambda: settings(refused_url))
    db = AsyncMock()
    checkpoints = AsyncMock()
    monkeypatch.setattr(cleanup, "Database", Mock(return_value=db))
    monkeypatch.setattr(cleanup, "Checkpoints", Mock(return_value=checkpoints))
    monkeypatch.setattr(cleanup, "make_pool", Mock())
    monkeypatch.setattr(cleanup, "make_checkpoint_pool", Mock())
    monkeypatch.setattr(
        cleanup, "run_cleanup", AsyncMock(return_value=CleanupResult(dict(COUNTS), ["questions"]))
    )
    assert await cleanup.main() == 1
    assert json.loads(capsys.readouterr().out) == {"deleted": COUNTS, "failed": ["questions"]}
    db.close.assert_awaited_once()
    checkpoints.close.assert_awaited_once()
