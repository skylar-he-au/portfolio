"""Patrol authentication and the separate raw and counting client addresses."""

import hashlib
import hmac
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import pytest
from fastapi.testclient import TestClient
from guard_helpers import guard_headers, guard_settings
from starlette.requests import Request
from test_api import ExplodingPool
from test_guard_session import issuing_client, success

from app.guard.session import (
    Rejected,
    raw_client_ip,
    require_session,
    sign_token,
    verify_token,
    verify_token_ex,
)
from app.main import create_app


def patrol_header(stamp: int, secret: str = "test-patrol") -> str:  # noqa: S107
    timestamp = str(stamp)
    signature = hmac.digest(secret.encode(), timestamp.encode("ascii"), "sha256").hex()
    return f"{timestamp}.{signature}"


@pytest.mark.parametrize("patrol", [False, True])
def test_patrol_session_skips_verifier_only_with_header(refused_url, patrol):
    settings = guard_settings(refused_url, patrol_secret="test-patrol")
    pool = ExplodingPool()
    ips = []
    limits = SimpleNamespace(allow_new_session=lambda ip: ips.append(ip) or True)
    verifier = SimpleNamespace(verify=AsyncMock())
    app = create_app(settings, pool=pool, limits=limits)
    app.state.turnstile_verifier = verifier
    headers = guard_headers(settings)
    if patrol:
        headers["X-Patrol"] = patrol_header(int(time.time()))
    with TestClient(app, headers=headers) as client:
        response = client.post("/api/session", json={"turnstile": "" if patrol else "challenge"})
    assert response.status_code == 200
    assert set(response.json()) == {"token", "expires_at"}
    assert response.headers["cache-control"] == "no-store"
    verified = verify_token_ex("test-session", response.json()["token"], time.time())
    assert verified is not None and verified[1] is patrol
    assert ips == ["127.0.0.1"]
    if patrol:
        verifier.verify.assert_not_called()
    else:
        verifier.verify.assert_awaited_once_with("challenge", "127.0.0.1")
    assert pool.touched == []


@pytest.mark.parametrize("invalid", ["old", "signature", "malformed", "unset"])
def test_invalid_patrol_is_always_auth(refused_url, invalid, monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr("app.api.session.time.time", lambda: now)
    settings = guard_settings(
        refused_url, patrol_secret=None if invalid == "unset" else "test-patrol"
    )
    header = patrol_header(now - 301 if invalid == "old" else now)
    if invalid == "signature":
        header = patrol_header(now, "wrong-patrol")
    elif invalid == "malformed":
        header = "not-a-timestamp.signature"
    with issuing_client(settings, success) as client:
        response = client.post(
            "/api/session", json={"turnstile": "challenge"}, headers={"X-Patrol": header}
        )
    assert response.status_code == 401
    assert response.json()["code"] == "auth"


@pytest.mark.parametrize(
    "header",
    [
        b"",
        b"1800000000." + b"a" * 63,
        b"1800000000." + b"A" * 64,
        b"1800000000." + b"g" * 64,
        b"+1800000000." + b"a" * 64,
        b"1" * 5000 + b"." + b"a" * 64,
        b"1800000000." + b"a" * 64 + b".extra",
        b"\xff." + b"a" * 64,
    ],
)
def test_patrol_header_parser_is_strict(refused_url, header):
    settings = guard_settings(refused_url, patrol_secret="test-patrol")
    with issuing_client(settings, success) as client:
        response = client.post(
            "/api/session", json={"turnstile": "challenge"}, headers={"X-Patrol": header}
        )
    assert response.status_code == 401 and response.json()["code"] == "auth"


@pytest.mark.parametrize(("offset", "status"), [(-300, 200), (300, 200), (301, 401)])
def test_patrol_timestamp_boundary(refused_url, monkeypatch, offset, status):
    now = 1_800_000_000
    monkeypatch.setattr("app.api.session.time.time", lambda: now)
    settings = guard_settings(refused_url, patrol_secret="test-patrol")
    with issuing_client(settings, success) as client:
        response = client.post(
            "/api/session",
            json={"turnstile": ""},
            headers={"X-Patrol": patrol_header(now + offset)},
        )
    assert response.status_code == status


@pytest.mark.parametrize("missing", ["session_secret", "turnstile_secret"])
def test_patrol_keeps_early_forbidden_check(refused_url, missing):
    settings = guard_settings(refused_url, patrol_secret="test-patrol", **{missing: None})
    pool = ExplodingPool()
    app = create_app(settings, pool=pool)
    verifier = SimpleNamespace(verify=AsyncMock())
    app.state.turnstile_verifier = verifier
    assert settings.origin_secret is not None
    headers = {
        "X-Origin-Secret": settings.origin_secret.get_secret_value(),
        "X-Client-IP": "192.0.2.1",
        "X-Patrol": patrol_header(int(time.time())),
    }
    with TestClient(app) as client:
        response = client.post("/api/session", headers=headers, json={"turnstile": ""})
    assert response.status_code == 403 and response.json()["code"] == "forbidden"
    verifier.verify.assert_not_called()
    assert pool.touched == []


@pytest.mark.parametrize("origin", [None, "wrong"])
def test_patrol_still_requires_worker_origin(refused_url, origin):
    settings = guard_settings(refused_url, patrol_secret="test-patrol")
    app = create_app(settings, pool=ExplodingPool())
    verifier = SimpleNamespace(verify=AsyncMock())
    app.state.turnstile_verifier = verifier
    headers = {
        "X-Client-IP": "192.0.2.1",
        "X-Patrol": patrol_header(int(time.time())),
    }
    if origin is not None:
        headers["X-Origin-Secret"] = origin
    with TestClient(app) as client:
        response = client.post("/api/session", headers=headers, json={"turnstile": ""})
    assert response.status_code == 403 and response.json()["code"] == "forbidden"
    verifier.verify.assert_not_called()


def test_patrol_keeps_new_session_rate_limit(refused_url):
    settings = guard_settings(refused_url, patrol_secret="test-patrol")
    with issuing_client(settings, success) as client:
        headers = {"X-Patrol": patrol_header(int(time.time()))}
        for _ in range(10):
            response = client.post("/api/session", headers=headers, json={"turnstile": ""})
            assert response.status_code == 200
        response = client.post("/api/session", headers=headers, json={"turnstile": ""})
    assert response.status_code == 429 and response.json()["code"] == "slow_down"


@pytest.mark.parametrize("patrol", [False, True])
def test_signed_marker_reaches_session(refused_url, patrol):
    settings = guard_settings(refused_url)
    token, _ = sign_token("test-session", 1000, patrol=patrol)
    verified = verify_token_ex("test-session", token, 1000)
    assert verified == (hashlib.sha256(token.split(".")[1].encode("ascii")).digest(), patrol)
    assert verify_token("test-session", token, 1000) == verified[0]
    live, _ = sign_token("test-session", time.time(), patrol=patrol)
    headers = guard_headers(settings, token=live)
    headers["X-Patrol"] = "ignored-on-other-endpoints"
    request = Request(
        {
            "type": "http",
            "app": SimpleNamespace(state=SimpleNamespace(settings=settings)),
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )
    session = require_session(request)
    assert session.patrol is patrol


@pytest.mark.parametrize("part", [0, 1, 2, 3])
def test_changing_any_signed_token_component_is_rejected(refused_url, part):
    token, _ = sign_token("test-session", time.time())
    parts = token.split(".")
    if part == 0:
        parts[0] = "p1"
    else:
        value = parts[part]
        parts[part] = ("1" if value[0] != "1" else "2") + value[1:]
    changed = ".".join(parts)
    assert verify_token("test-session", changed, time.time()) is None
    assert verify_token_ex("test-session", changed, time.time()) is None
    settings = guard_settings(refused_url)
    headers = guard_headers(settings, token=changed)
    request = Request(
        {
            "type": "http",
            "app": SimpleNamespace(state=SimpleNamespace(settings=settings)),
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )
    with pytest.raises(Rejected, match="auth"):
        require_session(request)


@pytest.mark.parametrize(
    ("ip", "remoteip", "counting_ip"),
    [
        ("2001:db8:1:2:3:4:5:6", "2001:db8:1:2:3:4:5:6", "2001:db8:1:2::/64"),
        ("192.0.2.1", "192.0.2.1", "192.0.2.1"),
        ("::ffff:192.0.2.1", "192.0.2.1", "192.0.2.1"),
    ],
)
def test_siteverify_raw_ip_and_issuing_network(refused_url, ip, remoteip, counting_ip):
    seen = []
    issued = []

    def transport(request):
        seen.append(parse_qs(request.content.decode())["remoteip"])
        return success(request)

    with issuing_client(guard_settings(refused_url), transport) as client:
        original = client.app.state.limits.allow_new_session

        def allow(address):
            issued.append(address)
            return original(address)

        client.app.state.limits.allow_new_session = allow
        response = client.post(
            "/api/session", json={"turnstile": "challenge"}, headers={"X-Client-IP": ip}
        )
    assert response.status_code == 200
    assert seen == [[remoteip]]
    assert issued == [counting_ip]


@pytest.mark.parametrize("ip", ["", "bad", "127.1", "fe80::1%en0"])
def test_raw_client_address_rejects_invalid_or_scoped_ip(ip):
    request = Request({"type": "http", "headers": [(b"x-client-ip", ip.encode())]})
    with pytest.raises(Rejected, match="forbidden"):
        raw_client_ip(request)
