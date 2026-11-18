"""Guard failures must finish before the SSE response starts."""

import psycopg
import pytest
from fastapi.testclient import TestClient
from guard_helpers import AdmittingLimits, NoSpendLedger, guard_headers, guard_settings
from test_api import ExplodingPool

from app.guard.limits import Admission
from app.guard.session import sign_token
from app.main import create_app
from app.migrate import migrate


class QuietService:
    def __init__(self):
        self.admissions: list[Admission] = []

    async def reserve(self, turn_id):
        return None

    async def stream(self, question, conversation_id, admission=None, *, scope=None, turn_id=None):
        self.admissions.append(admission)
        yield "done", {}

    async def close(self):
        pass


def test_ask_without_token_is_auth(refused_url: str):
    settings = guard_settings(refused_url)
    service = QuietService()
    pool = ExplodingPool()
    app = create_app(settings, pool=pool, ask_service=service, ledger=NoSpendLedger())
    with TestClient(app) as client:
        response = client.post(
            "/api/ask",
            headers={"X-Origin-Secret": "test-origin", "X-Client-IP": "127.0.0.1"},
            json={"question": "Hello"},
        )
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"code": "auth", "message": "Please try again."}
    assert pool.touched == [] and service.admissions == []


@pytest.mark.parametrize(
    "token",
    [
        "invalid",
        sign_token("test-session", 0)[0],
        sign_token("another-secret", 4_000_000_000)[0],
    ],
)
def test_bad_token_never_opens_database(refused_url, token):
    settings = guard_settings(refused_url)
    pool = ExplodingPool()
    service = QuietService()
    app = create_app(settings, pool=pool, ask_service=service, ledger=NoSpendLedger())
    with TestClient(app) as client:
        result = client.post(
            "/api/ask", headers=guard_headers(settings, token=token), json={"question": "Hello"}
        )
    assert result.status_code == 401 and result.json()["code"] == "auth"
    assert pool.touched == [] and service.admissions == []


def test_valid_request_with_database_down_is_json_unavailable(refused_url):
    settings = guard_settings(refused_url)
    service = QuietService()
    app = create_app(settings, ask_service=service, ledger=NoSpendLedger())
    with TestClient(app) as client:
        result = client.post(
            "/api/ask", headers=guard_headers(settings), json={"question": "Hello"}
        )
    assert result.status_code == 503
    assert result.headers["content-type"] == "application/json"
    assert result.json()["code"] == "unavailable"
    assert service.admissions == []


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_invalid_question_never_reaches_admit(refused_url, question):
    settings = guard_settings(refused_url)
    limits = AdmittingLimits()
    service = QuietService()
    app = create_app(settings, ask_service=service, ledger=NoSpendLedger(), limits=limits)
    with TestClient(app) as client:
        result = client.post(
            "/api/ask", headers=guard_headers(settings), json={"question": question}
        )
    assert result.status_code == 422
    assert limits.calls == 0 and service.admissions == []


def test_invalid_questions_leave_real_counters_unchanged(db_url):
    migrate(db_url)
    settings = guard_settings(db_url)
    service = QuietService()
    app = create_app(settings, ask_service=service, ledger=NoSpendLedger())
    with TestClient(app, headers=guard_headers(settings)) as client:
        assert client.post("/api/ask", json={"question": "valid"}).status_code == 200
        with psycopg.connect(db_url) as conn:
            before = conn.execute("SELECT * FROM usage_counters ORDER BY scope").fetchall()
        assert len(before) == 3
        for question in ("", "  ", "x" * 501):
            assert client.post("/api/ask", json={"question": question}).status_code == 422
        with psycopg.connect(db_url) as conn:
            after = conn.execute("SELECT * FROM usage_counters ORDER BY scope").fetchall()
    assert after == before
    assert len(service.admissions) == 1


def test_http_daily_rejection_precedes_sse(db_url):
    migrate(db_url)
    settings = guard_settings(db_url, ask_limit_token=1)
    service = QuietService()
    app = create_app(settings, ask_service=service, ledger=NoSpendLedger())
    with TestClient(app, headers=guard_headers(settings)) as client:
        assert client.post("/api/ask", json={"question": "valid"}).status_code == 200
        result = client.post("/api/ask", json={"question": "over limit"})
    assert result.status_code == 429
    assert result.headers["content-type"] == "application/json"
    assert result.json() == {
        "code": "limit_day",
        "message": "Today's AI limit is reached — please email me",
    }
    assert len(service.admissions) == 1
    assert service.admissions[0].kind == "ask"


@pytest.mark.parametrize(
    ("first_ip", "second_ip", "same"),
    [
        ("2001:db8::1", "2001:db8::2", True),
        ("::ffff:192.0.2.1", "192.0.2.1", True),
        ("192.0.2.1", "192.0.2.2", False),
    ],
)
def test_http_canonical_ip_and_xff_cannot_change_counter(db_url, first_ip, second_ip, same):
    migrate(db_url)
    settings = guard_settings(db_url)
    service = QuietService()
    app = create_app(settings, ask_service=service, ledger=NoSpendLedger())
    with TestClient(app, headers=guard_headers(settings)) as client:
        for ip, forwarded in ((first_ip, "203.0.113.1"), (second_ip, "203.0.113.2")):
            result = client.post(
                "/api/ask",
                json={"question": "valid"},
                headers={"X-Client-IP": ip, "X-Forwarded-For": forwarded},
            )
            assert result.status_code == 200
    first, second = service.admissions
    assert (first.ip_hash == second.ip_hash) is same
    with psycopg.connect(db_url) as conn:
        counts = conn.execute(
            "SELECT count FROM usage_counters WHERE scope = 'ip' ORDER BY count"
        ).fetchall()
    assert counts == ([(2,)] if same else [(1,), (1,)])
