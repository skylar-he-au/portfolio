"""Feedback uses session guards and bounded process limits, without question admission."""

import uuid
from datetime import UTC, datetime
from unittest.mock import Mock

import psycopg
import pytest
from fastapi.testclient import TestClient
from guard_helpers import Clock, guard_headers, guard_settings
from test_api import ExplodingPool

from app.api.feedback import feedback_router
from app.guard.session import Session, require_session
from app.main import create_app
from app.migrate import migrate


@pytest.fixture
def clock():
    # Sydney midnight falls on the previous UTC date in summer.
    return Clock(datetime(2026, 1, 1, 12, 59, tzinfo=UTC))


def seed_question(db_url, *, days=1):
    turn_id = uuid.uuid4()
    with psycopg.connect(db_url) as conn:
        conn.execute(
            "INSERT INTO question_log (id, question, prompt_version, ip_hash, created_at) "
            "VALUES (%s, 'question', 'v-test', %s, now() - %s * interval '1 day')",
            (turn_id, b"hash", days),
        )
    return turn_id


@pytest.mark.parametrize("days, changed", [(1, True), (31, False)])
def test_feedback_updates_only_recent_questions(db_url, days, changed):
    migrate(db_url)
    turn_id = seed_question(db_url, days=days)
    settings = guard_settings(db_url)
    with TestClient(create_app(settings), headers=guard_headers(settings)) as client:
        result = client.post("/api/feedback", json={"turn_id": str(turn_id)})
    assert result.status_code == 204 and result.content == b""
    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            "SELECT thumbs_down FROM question_log WHERE id = %s", (turn_id,)
        ).fetchone()
        assert row == (changed,)
        assert conn.execute("SELECT count(*) FROM usage_counters").fetchone() == (0,)


def test_unknown_turn_is_still_204(db_url):
    migrate(db_url)
    settings = guard_settings(db_url)
    with TestClient(create_app(settings), headers=guard_headers(settings)) as client:
        result = client.post("/api/feedback", json={"turn_id": str(uuid.uuid4())})
        assert result.status_code == 204


@pytest.mark.parametrize(
    "missing, value, status",
    [
        ("Authorization", None, 401),
        ("Authorization", "Bearer invalid", 401),
        ("X-Origin-Secret", None, 403),
        ("X-Origin-Secret", "wrong", 403),
    ],
)
def test_feedback_guards_never_touch_database(refused_url, missing, value, status):
    settings = guard_settings(refused_url)
    headers = guard_headers(settings)
    headers.pop(missing)
    if value is not None:
        headers[missing] = value
    pool = ExplodingPool()
    with TestClient(create_app(settings, pool=pool), headers=headers) as client:
        response = client.post("/api/feedback", json={"turn_id": str(uuid.uuid4())})
    assert response.status_code == status
    assert response.json()["code"] == ("auth" if status == 401 else "forbidden")
    assert pool.touched == []


@pytest.mark.parametrize(
    "body", [{}, {"turn_id": "bad"}, {"turn_id": str(uuid.uuid4()), "extra": True}]
)
def test_feedback_rejects_malformed_body_without_sql(refused_url, body):
    settings = guard_settings(refused_url)
    pool = ExplodingPool()
    with TestClient(create_app(settings, pool=pool), headers=guard_headers(settings)) as client:
        assert client.post("/api/feedback", json=body).status_code == 422
    assert pool.touched == []


def test_feedback_daily_cap_counts_rejections_and_resets_in_sydney(refused_url, clock, monkeypatch):
    monkeypatch.setattr("app.main.feedback_router", lambda: feedback_router(clock=clock))
    settings = guard_settings(refused_url)
    pool = ExplodingPool()
    limits = Mock()
    limits.allow_db_query.return_value = False
    headers = guard_headers(settings)
    app = create_app(settings, pool=pool, limits=limits)
    with TestClient(app, headers=headers) as client:
        for _ in range(20):
            response = client.post("/api/feedback", json={"turn_id": str(uuid.uuid4())})
            assert response.status_code == 429 and response.json()["code"] == "slow_down"
        assert limits.allow_db_query.call_count == 20
        response = client.post("/api/feedback", json={"turn_id": str(uuid.uuid4())})
        assert response.status_code == 429 and response.json()["code"] == "slow_down"
        assert limits.allow_db_query.call_count == 20
        client.post(
            "/api/feedback", headers=guard_headers(settings), json={"turn_id": str(uuid.uuid4())}
        )
        assert limits.allow_db_query.call_count == 21
        clock.advance(minutes=1)
        client.post("/api/feedback", json={"turn_id": str(uuid.uuid4())})
        assert limits.allow_db_query.call_count == 22
    assert pool.touched == []
    limits.admit.assert_not_called()


def test_feedback_accepts_twenty_requests_then_limits_only_that_token(db_url, clock, monkeypatch):
    migrate(db_url)
    monkeypatch.setattr("app.main.feedback_router", lambda: feedback_router(clock=clock))
    settings = guard_settings(db_url)
    headers = guard_headers(settings)
    with TestClient(create_app(settings), headers=headers) as client:
        body = {"turn_id": str(uuid.uuid4())}
        for _ in range(20):
            assert client.post("/api/feedback", json=body).status_code == 204
        response = client.post("/api/feedback", json=body)
        assert response.status_code == 429 and response.json()["code"] == "slow_down"
        result = client.post("/api/feedback", headers=guard_headers(settings), json=body)
        assert result.status_code == 204
        clock.advance(minutes=1)
        assert client.post("/api/feedback", json=body).status_code == 204


def test_feedback_key_capacity_rejects_new_keys_but_preserves_existing(refused_url):
    settings = guard_settings(refused_url)
    pool = ExplodingPool()
    limits = Mock()
    limits.allow_db_query.return_value = False
    app = create_app(settings, pool=pool, limits=limits)
    key = 0
    app.dependency_overrides[require_session] = lambda: Session(key.to_bytes(32), "127.0.0.1")
    with TestClient(app, headers=guard_headers(settings)) as client:
        body = {"turn_id": str(uuid.uuid4())}
        for key in range(10_000):  # noqa: B007 - read by the dependency override
            assert client.post("/api/feedback", json=body).status_code == 429
        assert limits.allow_db_query.call_count == 10_000
        key = 10_000
        assert client.post("/api/feedback", json=body).status_code == 429
        assert limits.allow_db_query.call_count == 10_000
        key = 0
        assert client.post("/api/feedback", json=body).status_code == 429
        assert limits.allow_db_query.call_count == 10_001
    assert pool.touched == []


def test_feedback_database_down_is_unavailable(refused_url):
    settings = guard_settings(refused_url)
    with TestClient(create_app(settings), headers=guard_headers(settings)) as client:
        response = client.post("/api/feedback", json={"turn_id": str(uuid.uuid4())})
    assert response.status_code == 503 and response.json()["code"] == "unavailable"


# --- step 6b2: thumbs-down on a job match ------------------------------------------------


def seed_match(db_url, *, days=1):
    match_id = uuid.uuid4()
    with psycopg.connect(db_url) as conn:
        conn.execute(
            "INSERT INTO match_log (id, source, prompt_version, ip_hash, created_at) "
            "VALUES (%s, 'text', 'match-v1', %s, now() - %s * interval '1 day')",
            (match_id, b"hash", days),
        )
    return match_id


@pytest.mark.parametrize("days, changed", [(1, True), (31, False)])
def test_feedback_marks_only_recent_matches(db_url, days, changed):
    migrate(db_url)
    match_id = seed_match(db_url, days=days)
    turn_id = seed_question(db_url)
    settings = guard_settings(db_url)
    with TestClient(create_app(settings), headers=guard_headers(settings)) as client:
        assert client.post("/api/feedback", json={"match_id": str(match_id)}).status_code == 204
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT thumbs_down FROM match_log").fetchone() == (changed,)
        # The question with no feedback is untouched.
        row = conn.execute("SELECT thumbs_down FROM question_log WHERE id = %s", (turn_id,))
        assert row.fetchone() == (False,)


@pytest.mark.parametrize(
    "body",
    [
        {"turn_id": str(uuid.uuid4()), "match_id": str(uuid.uuid4())},
        {"match_id": None},
        {"turn_id": None, "match_id": None},
        {"match_id": "bad"},
    ],
)
def test_feedback_needs_exactly_one_id(refused_url, body):
    settings = guard_settings(refused_url)
    pool = ExplodingPool()
    with TestClient(create_app(settings, pool=pool), headers=guard_headers(settings)) as client:
        assert client.post("/api/feedback", json=body).status_code == 422
    assert pool.touched == []
