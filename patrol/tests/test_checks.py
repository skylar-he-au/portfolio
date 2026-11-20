"""Unit tests: no model, no browser."""

import hmac
import re

import pytest

from patrol.checks import (
    Observation,
    Snapshot,
    base_origin,
    domain_pattern,
    patrol_header,
    same_origin,
    verdict,
)
from patrol.run import Report, _init_script, format_report, redact

BASE = "http://localhost:4321"


# X-Patrol, convention 2: "<unix seconds>.<hex(HMAC-SHA256(PATROL_SECRET, "<unix seconds>"))>".
# The expected value was computed apart from this code:
#   printf '1700000000' | openssl dgst -sha256 -hmac test-patrol
def test_patrol_header_matches_openssl_vector():
    assert patrol_header("test-patrol", 1_700_000_000.9) == (
        "1700000000.e4769a4576f4551ad25294cdd5093cf4fd30d06fdb1ab049fead6335c8880c3e"
    )


def test_patrol_header_matches_the_api_tests_helper():
    # api/tests/test_patrol_session.py `patrol_header`, which the API's own tests accept
    stamp = 1_763_600_000
    expected = hmac.digest(b"test-patrol", str(stamp).encode("ascii"), "sha256").hex()
    assert patrol_header("test-patrol", stamp) == f"{stamp}.{expected}"
    # and the shape the API's regex wants: [0-9]{1,20}\.[0-9a-f]{64}
    assert re.fullmatch(r"[0-9]{1,20}\.[0-9a-f]{64}", patrol_header("x", stamp))


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("http://localhost:4321/", True),
        ("http://localhost:4321/api/ask", True),
        ("http://LOCALHOST:4321/projects/x#y", True),
        ("http://localhost:8000/api/ping", False),  # another port
        ("https://localhost:4321/", False),  # another scheme
        ("http://127.0.0.1:4321/", False),
        ("https://challenges.cloudflare.com/turnstile/v0/api.js", False),
        ("http://localhost.evil.com:4321/", False),
        ("http://evil.com/?http://localhost:4321/", False),
        ("http://user@evil.com:4321/", False),
        ("ws://localhost:4321/", False),
        ("data:text/plain,hi", False),
        ("not a url", False),
        ("http://localhost:99999/", False),
    ],
)
def test_same_origin_lets_only_the_site_through(url, allowed):
    assert same_origin(url, BASE) is allowed


def test_same_origin_default_ports():
    assert same_origin("https://example.com/", "https://example.com")
    assert same_origin("https://example.com:443/x", "https://example.com")
    assert not same_origin("https://www.example.com/", "https://example.com")
    assert not same_origin("http://example.com/", "https://example.com")


def test_base_origin_and_domain_pattern():
    assert base_origin("https://Example.com/some/path?q=1") == "https://example.com"
    assert base_origin("http://localhost:4321/") == BASE
    assert domain_pattern("https://example.com") == "https://example.com"
    assert domain_pattern(BASE) == "localhost:4321"
    for bad in ("example.com", "ftp://example.com", "javascript:alert(1)", ""):
        with pytest.raises(ValueError):
            base_origin(bad)


def _green() -> Observation:
    obs = Observation(base_url=BASE, session_status=200, home_status=200)
    obs.add(Snapshot(url=f"{BASE}/", home=True))
    obs.add(
        Snapshot(
            url=f"{BASE}/",
            home=True,
            answer_chars=120,
            answer_done=True,
            answer_links=[f"{BASE}/projects/careermate#rag", "https://github.com/x"],
        )
    )
    obs.navigations += [f"{BASE}/", f"{BASE}/projects/careermate#rag", f"{BASE}/"]
    obs.add(Snapshot(url=f"{BASE}/", home=True, match_chars=300, match_done=True))
    return obs


def test_verdict_green_needs_every_step():
    steps = verdict(_green())
    assert [s.name for s in steps] == [
        "session",
        "home",
        "ask",
        "citation",
        "open-citation",
        "match",
    ]
    assert all(s.ok for s in steps)
    assert _green().links == {f"{BASE}/projects/careermate#rag"}  # off-site links never count


def _failed(obs: Observation) -> list[str]:
    return [s.name for s in verdict(obs) if not s.ok]


def test_verdict_session_refused():
    obs = Observation(base_url=BASE, session_status=401, session_error="HTTP 401 auth")
    assert _failed(obs)[0] == "session"
    assert "HTTP 401 auth" in verdict(obs)[0].detail


def test_verdict_answer_without_done_button_is_not_answered():
    obs = Observation(base_url=BASE, session_status=200, home_status=200)
    obs.add(
        Snapshot(
            url=f"{BASE}/",
            home=True,
            answer_chars=50,
            answer_done=False,
            answer_links=[f"{BASE}/projects/a#b"],
        )
    )
    obs.add_status("/api/ask", 503)
    obs.add(Snapshot(url=f"{BASE}/", home=True, errors=["The assistant is unavailable"]))
    assert _failed(obs) == ["ask", "citation", "open-citation", "match"]
    ask = verdict(obs)[2]
    assert "/api/ask returned 503" in ask.detail and "unavailable" in ask.detail


def test_verdict_link_seen_but_not_opened():
    obs = _green()
    obs.navigations = [f"{BASE}/", f"{BASE}/#ask"]
    assert _failed(obs) == ["open-citation"]


def test_verdict_navigation_to_another_page_is_not_a_citation():
    obs = _green()
    obs.navigations = [f"{BASE}/", f"{BASE}/privacy"]
    assert _failed(obs) == ["open-citation"]


def test_verdict_match_needs_result_and_done():
    obs = _green()
    obs.matched = False
    obs.add(Snapshot(url=f"{BASE}/", home=True, match_chars=0, match_done=True))
    obs.add(Snapshot(url=f"{BASE}/", home=True, match_chars=10, match_done=False))
    assert _failed(obs) == ["match"]
    assert "no /api/match call" in verdict(obs)[5].detail


def test_snapshots_from_other_origins_are_ignored():
    obs = Observation(base_url=BASE, session_status=200, home_status=200)
    obs.add(
        Snapshot(
            url="https://evil.com/",
            home=True,
            answer_chars=9,
            answer_done=True,
            answer_links=[f"{BASE}/projects/a"],
            match_chars=9,
            match_done=True,
        )
    )
    assert not (obs.home or obs.answered or obs.matched or obs.links)


def test_init_script_only_writes_on_the_site():
    script = _init_script(BASE, '{"token": "p1.x", "expires_at": "2026-11-21T00:00:00Z"}')
    assert 'location.origin !== "http://localhost:4321"' in script
    assert "sessionStorage.setItem('session', " in script
    assert '"{\\"token\\": \\"p1.x\\"' in script  # the JSON text, quoted as one JS string


def test_report_never_prints_secrets():
    report = Report(Observation(base_url=BASE, session_status=401))
    text = format_report(report)
    assert "RED at step 'session'" in text
    assert redact("key abc and token p1.zz", ["abc", "p1.zz", ""]) == (
        "key [redacted] and token [redacted]"
    )


def test_report_marks_steps_after_a_refused_session_not_reached():
    text = format_report(Report(Observation(base_url=BASE, session_status=401)))
    assert "FAIL  session" in text
    assert "--    home           not reached" in text and "FAIL  home" not in text
