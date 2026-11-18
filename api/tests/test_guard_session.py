"""Stateless tokens, trusted headers and mocked Turnstile responses."""

import hashlib
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient
from guard_helpers import AdmittingLimits, Clock, guard_headers, guard_settings
from test_api import ExplodingPool
from test_guard_ask import QuietService

from app.api.ask import MESSAGES
from app.config import Settings
from app.db import Database
from app.guard.limits import Limits
from app.guard.session import REJECTIONS, sign_token, verify_token
from app.main import create_app


def test_sign_and_verify():
    token, exp = sign_token("secret", 1000)
    assert exp == 87400
    assert (
        verify_token("secret", token, 1000)
        == hashlib.sha256(token.split(".")[1].encode("ascii")).digest()
    )
    assert verify_token("secret", token, exp - 1) is not None
    assert verify_token("secret", token, exp) is None
    assert verify_token("secret", token, exp + 1) is None
    assert verify_token("another", token, 1000) is None
    replacement = "A" if token[-2] != "A" else "B"
    assert verify_token("secret", token[:-2] + replacement + token[-1], 1000) is None
    changed_id = ("0" if token[3] != "0" else "1") + token[4:]
    assert verify_token("secret", "v1." + changed_id, 1000) is None


@pytest.mark.parametrize(
    "token",
    [
        "",
        "v1.a.10000.sig",
        "v2." + "a" * 32 + ".10000." + "a" * 43,
        "v1." + "A" * 32 + ".10000." + "a" * 43,
        "v1." + "a" * 32 + ".+10000." + "a" * 43,
        "v1." + "a" * 32 + ".١٢٣." + "a" * 43,
        "v1." + "a" * 32 + "." + "9" * 5000 + "." + "a" * 43,
        "v1." + "a" * 32 + ".10000." + "a" * 43 + "=",
        "v1." + "a" * 32 + ".10000." + "a" * 43 + ".extra",
    ],
)
def test_token_parser_is_strict(token):
    assert verify_token("secret", token, 1000) is None


@contextmanager
def issuing_client(
    settings: Settings,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    clock: Clock | None = None,
) -> Iterator[TestClient]:
    pool = ExplodingPool()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    limits = Limits(Database(pool), settings, clock=clock or Clock())
    app = create_app(settings, pool=pool, limits=limits, turnstile_client=http)
    with TestClient(app, headers=guard_headers(settings)) as client:
        try:
            yield client
        finally:
            client.portal.call(http.aclose)
    assert pool.touched == []


def success(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"success": True, "hostname": "example.com"})


@pytest.mark.parametrize(
    ("response", "status", "code"),
    [
        (httpx.Response(200, json={"success": False}), 401, "auth"),
        (httpx.Response(200, json={"success": 1}), 401, "auth"),
        (httpx.Response(200, json=[]), 401, "auth"),
        (httpx.Response(500, json={}), 503, "unavailable"),
        (httpx.Response(200, text="not json"), 503, "unavailable"),
    ],
)
def test_siteverify_failures(refused_url, response, status, code):
    with issuing_client(guard_settings(refused_url), lambda _: response) as client:
        result = client.post("/api/session", json={"turnstile": "challenge"})
    assert result.status_code == status
    assert result.json() == {"code": code, "message": REJECTIONS[code][1]}


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_siteverify_transport_error(refused_url, error):
    def broken(request):
        raise error("unreachable", request=request)

    with issuing_client(guard_settings(refused_url), broken) as client:
        result = client.post("/api/session", json={"turnstile": "challenge"})
    assert result.status_code == 503
    assert result.json()["code"] == "unavailable"


@pytest.mark.parametrize(
    ("ip", "canonical"),
    [("::ffff:192.0.2.1", "192.0.2.1"), ("2001:db8::2", "2001:db8::/64")],
)
def test_session_test_secret_and_canonical_remoteip(refused_url, ip, canonical):
    settings = guard_settings(refused_url)

    def check(request):
        assert request.method == "POST"
        assert str(request.url) == "https://challenges.cloudflare.com/turnstile/v0/siteverify"
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        assert parse_qs(request.content.decode()) == {
            "secret": [settings.turnstile_secret.get_secret_value()],
            "response": ["challenge"],
            "remoteip": [canonical],
        }
        assert all(value == 5.0 for value in request.extensions["timeout"].values())
        return success(request)

    with issuing_client(settings, check) as client:
        result = client.post(
            "/api/session", json={"turnstile": "challenge"}, headers={"X-Client-IP": ip}
        )
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    body = result.json()
    now = datetime.now(UTC).timestamp()
    assert verify_token("test-session", body["token"], now) is not None
    assert body["expires_at"].endswith("Z")
    expires = datetime.fromisoformat(body["expires_at"]).timestamp()
    assert expires == int(body["token"].split(".")[2])


@pytest.mark.parametrize(
    "secret",
    [
        "1x0000000000000000000000000000000AA",
        "2x0000000000000000000000000000000AA",
        "3x0000000000000000000000000000000AA",
    ],
)
def test_each_official_secret_skips_hostname_and_action(refused_url, secret):
    settings = guard_settings(refused_url, turnstile_secret=secret)
    with issuing_client(settings, success) as client:
        assert client.post("/api/session", json={"turnstile": "challenge"}).status_code == 200


@pytest.mark.parametrize(
    ("hostname", "action", "expected"),
    [("wrong.com", "ask", 401), ("a.com", "other", 401), ("a.com", "ask", 200)],
)
def test_production_turnstile_bindings(refused_url, hostname, action, expected):
    settings = guard_settings(
        refused_url, turnstile_secret="live-secret", turnstile_hostnames=["a.com"]
    )

    def check(_request):
        return httpx.Response(200, json={"success": True, "hostname": hostname, "action": action})

    with issuing_client(settings, check) as client:
        result = client.post("/api/session", json={"turnstile": "challenge"})
    assert result.status_code == expected


def test_issuing_sieve_counts_failed_challenges_and_resets(refused_url):
    calls = 0
    clock = Clock()

    def failed(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"success": False})

    with issuing_client(guard_settings(refused_url), failed, clock=clock) as client:
        for _ in range(10):
            assert client.post("/api/session", json={"turnstile": "challenge"}).status_code == 401
        result = client.post("/api/session", json={"turnstile": "challenge"})
        assert result.status_code == 429 and result.json()["code"] == "slow_down"
        assert calls == 10
        clock.advance(hours=1)
        assert client.post("/api/session", json={"turnstile": "challenge"}).status_code == 401
    assert calls == 11


@pytest.mark.parametrize("token", ["", "x" * 2049])
def test_empty_or_long_challenge_never_calls_cloudflare(refused_url, token):
    def exploding(_request):
        pytest.fail("Cloudflare must not be called")

    with issuing_client(guard_settings(refused_url), exploding) as client:
        for _ in range(10):
            result = client.post("/api/session", json={"turnstile": token})
            assert result.status_code == 401
        assert client.post("/api/session", json={"turnstile": token}).json()["code"] == "slow_down"


@pytest.mark.parametrize("body", [{}, {"turnstile": 3}, {"turnstile": "x", "extra": 1}, []])
def test_session_body_shape(refused_url, body):
    with issuing_client(guard_settings(refused_url), success) as client:
        assert client.post("/api/session", json=body).status_code == 422


@pytest.mark.parametrize("route", ["/api/ask", "/api/session"])
@pytest.mark.parametrize("origin", [None, "wrong"])
def test_origin_before_bad_token(refused_url, route, origin):
    app = create_app(guard_settings(refused_url), pool=ExplodingPool())
    headers = {"X-Client-IP": "127.0.0.1", "Authorization": "Bearer invalid"}
    if origin is not None:
        headers["X-Origin-Secret"] = origin
    with TestClient(app) as client:
        result = client.post(route, headers=headers, json={"question": "Hello"})
        assert client.get("/api/ping").status_code == 200
    assert result.status_code == 403 and result.json()["code"] == "forbidden"


@pytest.mark.parametrize("ip", [None, "bad", "127.1", "fe80::1%en0"])
def test_missing_or_invalid_client_ip(refused_url, ip):
    settings = guard_settings(refused_url)
    app = create_app(settings, pool=ExplodingPool())
    headers = guard_headers(settings)
    del headers["X-Client-IP"]
    headers["X-Forwarded-For"] = "192.0.2.1"
    if ip is not None:
        headers["X-Client-IP"] = ip
    with TestClient(app) as client:
        result = client.post("/api/ask", headers=headers, json={"question": "Hello"})
    assert result.status_code == 403 and result.json()["code"] == "forbidden"


@pytest.mark.parametrize(
    "name",
    [
        "origin_secret",
        "session_secret",
        "turnstile_secret",
        "spend_daily_usd_anthropic",
        "spend_daily_usd_google",
        "spend_daily_usd_openai",
        "spend_monthly_usd_anthropic",
        "spend_monthly_usd_google",
        "spend_monthly_usd_openai",
    ],
)
def test_prod_requires_each_variable(refused_url, name):
    settings = guard_settings(refused_url, app_env="prod", **{name: None})
    with pytest.raises(RuntimeError, match=name.upper()) as error:
        create_app(settings)
    assert "test-session" not in str(error.value)


@pytest.mark.parametrize("name", ["origin_secret", "session_secret", "turnstile_secret"])
def test_dev_missing_secrets_warn_once(refused_url, caplog, name):
    settings = guard_settings(refused_url, **{name: None})
    pool = ExplodingPool()
    app = create_app(settings, pool=pool)
    headers = {"X-Origin-Secret": "test-origin", "X-Client-IP": "127.0.0.1"}
    with caplog.at_level(logging.WARNING), TestClient(app) as client:
        for _ in range(2):
            result = client.post("/api/ask", headers=headers, json={"question": "Hello"})
            assert result.status_code == (403 if name == "origin_secret" else 401)
            result = client.post("/api/session", headers=headers, json={"turnstile": "challenge"})
            assert result.status_code == 403
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == f"{name.upper()} is not configured"
    assert pool.touched == []


def test_bearer_scheme_is_case_insensitive(refused_url):
    settings = guard_settings(refused_url)
    headers = guard_headers(settings)
    headers["Authorization"] = headers["Authorization"].replace("Bearer", "bEaReR")
    app = create_app(settings, ask_service=QuietService(), limits=AdmittingLimits())
    with TestClient(app) as client:
        assert client.post("/api/ask", headers=headers, json={"question": "Hi"}).status_code == 200


def test_unavailable_message_matches_ask():
    assert REJECTIONS["unavailable"][1] == MESSAGES["unavailable"]
