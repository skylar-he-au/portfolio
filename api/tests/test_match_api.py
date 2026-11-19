"""`POST /api/match` end to end: real `Limits` and `Ledger` on a test database, the real
graph with fake models (`fakes.FakeJsonChat`) and a fake batch retriever, and job links
fetched through `httpx.MockTransport` (every outbound request is recorded).

Numbers in the comments refer to the 6b2 acceptance list in `docs/briefs/step6.md`.
"""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from fakes import FakeJsonChat
from guard_helpers import Clock, guard_headers, guard_settings
from langchain_core.messages import BaseMessage

from app.api import match as match_api
from app.api.match import (
    MATCH_TIMEOUT_S,
    MAX_JD_CHARS,
    MESSAGES,
    SECTION_CHARS_ESTIMATE,
    MatchService,
    match_worst,
    match_worst_for,
)
from app.db import Database, make_pool
from app.guard.ledger import PROVIDERS, AIPaused, Ledger, SpendLimits, sydney_day
from app.guard.limits import Limits
from app.guard.prices import cost_usd
from app.guard.session import sign_token
from app.guard.usage import estimate_input_tokens
from app.main import create_app
from app.match import prompts
from app.match.graph import DEFAULT_STAGE_TIMEOUTS, MatchDeps
from app.match.state import Requirement
from app.migrate import migrate
from app.qa.models import MAX_OUTPUT_TOKENS, ChatProvider
from app.rag.types import ModelsUnavailable, Section
from app.records import MatchRecords

pytestmark = pytest.mark.anyio

FIXTURES = Path(__file__).parent / "fixtures" / "match"
LINK = "https://job-boards.greenhouse.io/acme/jobs/4005542201"
JD = "We need Python and Kubernetes. Great benefits."
MODELS = {
    "anthropic": "claude-haiku-5-5",
    "google": "gemini-3.5-flash-lite",
    "openai": "text-embedding-3-small",
}
# SPEC "费用有上限": the site's daily caps per provider.
SPEC_DAILY = {"anthropic": Decimal("0.15"), "google": Decimal("0.09"), "openai": Decimal("0.06")}

Event = tuple[str, dict[str, Any]]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- fakes ------------------------------------------------------------------------------------


def kind(messages: Sequence[BaseMessage]) -> str:
    system = str(messages[0].content)
    for name, start in (
        ("read", "You read a job ad"),
        ("match", "You check one job requirement"),
        ("write", "You write a short reading guide"),
        ("check", "You check links in a reading guide"),
    ):
        if system.startswith(start):
            return name
    raise AssertionError(f"unknown prompt: {system[:40]}")


def requirement_of(messages: Sequence[BaseMessage]) -> str:
    found = re.search(r"<requirement-[0-9a-f]+>\n(.*?)\n</requirement-", str(messages[1].content))
    assert found
    return found.group(1)


REASONING = "Section 1 shows it, says the secret reasoning."


class Script:
    """Replies by prompt kind; tests override pieces. Records the job ads it was given."""

    def __init__(self, requirements: Sequence[str] = ("Python", "Kubernetes")) -> None:
        self.requirements = list(requirements)
        self.job_ads: list[str] = []
        self.calls: list[str] = []

    async def read(self, messages: list[BaseMessage]) -> Any:
        self.job_ads.append(str(messages[1].content))
        return {
            "language": "en",
            "requirements": [{"text": t, "query": f"q {t}"} for t in self.requirements],
        }

    async def match(self, messages: list[BaseMessage]) -> Any:
        await asyncio.sleep(0.05)  # finishes after the no-section requirements
        return {"reasoning": REASONING, "found": True, "sections": [1]}

    async def write(self, messages: list[BaseMessage]) -> Any:
        n = len(re.findall(r"^Finding \d+:", str(messages[1].content), flags=re.M))
        return {"items": [{"finding": i, "text": f"guide text {i}"} for i in range(1, n + 1)]}

    async def check(self, messages: list[BaseMessage]) -> Any:
        n = len(re.findall(r"^Pair \d+:", str(messages[1].content), flags=re.M))
        return {"verdicts": [{"n": i, "shows": True} for i in range(1, n + 1)]}

    async def __call__(self, messages: list[BaseMessage]) -> Any:
        name = kind(messages)
        self.calls.append(name)
        return await getattr(self, name)(messages)


def section(sid: str) -> Section:
    return Section("/projects/roleradar", sid, f"Heading {sid}", f"text of {sid}", 0.01)


class FakeSearch:
    """Sections for "q Python" only; every other requirement finds nothing."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, queries: Sequence[str], *, page_prefix: str) -> list[list[Section]]:
        self.calls.append(list(queries))
        return [[section("python")] if q == "q Python" else [] for q in queries]


@dataclass
class Outbound:
    """MockTransport handler: one fixed response; records every request."""

    status: int = 200
    body: bytes = (FIXTURES / "greenhouse.json").read_bytes()
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, content=self.body)


# --- the world: app, service, database --------------------------------------------------------


@dataclass
class World:
    url: str
    db: Database
    settings: Any
    ledger: Ledger
    limits: Limits
    service: MatchService
    script: Script
    outbound: Outbound
    clock: Clock
    app: Any
    token: str  # one visitor's session token; other visitors pass their own

    def client(self, ip: str = "192.0.2.1", token: str | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://test",
            headers=guard_headers(self.settings, ip=ip, token=token or self.token),
        )

    def rows(self, query: str, params: tuple = ()) -> list[tuple]:
        with psycopg.connect(self.url) as conn:
            return conn.execute(query, params).fetchall()

    def execute(self, query: str, params: tuple = ()) -> None:
        with psycopg.connect(self.url) as conn:
            conn.execute(query, params)

    def counts(self, kind: str) -> dict[str, int]:
        return {
            scope: count
            for scope, count in self.rows(
                "SELECT scope, count FROM usage_counters WHERE kind = %s", (kind,)
            )
        }


@pytest.fixture
async def make_world(db_url: str) -> AsyncIterator[Callable[..., World]]:
    migrate(db_url)
    dbs: list[Database] = []

    def build(
        *,
        script: Script | None = None,
        outbound: Outbound | None = None,
        timeout_s: float = MATCH_TIMEOUT_S,
        stage: dict[Any, float] | None = None,
        daily: dict[str, Decimal] | None = None,
        **overrides: Any,
    ) -> World:
        settings = guard_settings(db_url, **overrides)
        db = Database(make_pool(db_url))
        dbs.append(db)
        spend = SpendLimits.from_settings(settings)
        if daily is not None:
            spend = SpendLimits(daily, spend.monthly, spend.models, spend.input_tokens_estimate)
        ledger = Ledger(db, spend, MAX_OUTPUT_TOKENS)
        ledger.set_worst("match", match_worst_for(settings))
        clock = Clock(datetime.now(UTC))
        limits = Limits(db, settings, ledger, clock=clock)
        script = script or Script()
        outbound = outbound or Outbound()
        claude = FakeJsonChat(reply=script)
        gemini = FakeJsonChat(reply=script)
        deps = MatchDeps(
            ChatProvider("anthropic", MODELS["anthropic"], claude),
            ChatProvider("google", MODELS["google"], gemini),
            FakeSearch(),
            stage_timeouts={  # type: ignore[arg-type]
                "reading": 5.0,
                "matching": 5.0,
                "checking": 5.0,
                "writing": 5.0,
                **(stage or {}),
            },
            call_timeout=5.0,
        )
        service = MatchService(
            limits,
            ledger,
            lambda: deps,
            records=MatchRecords(db),
            client=httpx.AsyncClient(transport=httpx.MockTransport(outbound)),
            timeout_s=timeout_s,
            clock=clock,
        )
        app = create_app(settings, ledger=ledger, limits=limits, match_service=service)
        assert settings.session_secret is not None
        token, _ = sign_token(settings.session_secret.get_secret_value(), time.time())
        return World(
            db_url, db, settings, ledger, limits, service, script, outbound, clock, app, token
        )

    yield build
    for db in dbs:
        await db.close()


def parse_sse(body: str) -> list[Event]:
    events: list[Event] = []
    for block in body.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if lines:
            fields = dict(line.split(": ", 1) for line in lines)
            events.append((fields["event"], json.loads(fields["data"])))
    return events


async def post(world: World, text: str, **client: Any) -> httpx.Response:
    async with world.client(**client) as http:
        response = await http.post("/api/match", json={"input": text})
    await world.service.wait_idle()
    return response


async def events_of(world: World, text: str, **client: Any) -> list[Event]:
    response = await post(world, text, **client)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def error_of(events: list[Event]) -> dict[str, Any]:
    assert events[-1][0] == "error", events
    return events[-1][1]


# --- 1. reservation by kind ---------------------------------------------------------------


def real_worst() -> dict[str, Decimal]:
    """The match worst path written out from the real prompts at their largest."""
    marker = "f" * 16
    req = Requirement(id="r1", text="x" * 120, query="x" * 200)
    sec = Section("/projects/" + "x" * 30, "x" * 40, "x" * 80, "x" * SECTION_CHARS_ESTIMATE, 0)
    read = estimate_input_tokens(prompts.read_messages("x" * 20_000, marker))
    one_match = estimate_input_tokens(prompts.match_messages(req, [sec] * 6, marker))
    write = estimate_input_tokens(
        prompts.write_messages([(n, req, [sec] * 3) for n in range(1, 9)], "en", marker)
    )
    check = estimate_input_tokens(
        prompts.check_messages([(n, req, sec) for n in range(1, 25)], marker)
    )

    def call(model: str, tokens: int) -> Decimal:
        return cost_usd(model, {"input_tokens": tokens, "output_tokens": 1024})

    haiku, flash = MODELS["anthropic"], MODELS["google"]
    return {
        # read 1 + match 8 + write 1, all on Claude (the main path)
        "anthropic": call(haiku, read) + 8 * call(haiku, one_match) + call(haiku, write),
        # check: one Gemini call for 8 requirements x 3 links
        "google": call(flash, check),
        # one embedding call for 8 search phrases of at most 200 characters
        "openai": cost_usd(MODELS["openai"], {"input_tokens": (8 * 200 + 2) // 3}),
    }


def test_worst_path_comes_from_the_real_prompts() -> None:
    assert match_worst(MODELS["anthropic"], MODELS["google"], MODELS["openai"]) == real_worst()
    worst = real_worst()
    # The SPEC estimates (read ~7k, each match ~3.5k input tokens) are below these: the
    # reservation assumes the longest JD and the longest sections.
    assert worst["anthropic"] < Decimal("0.02")
    assert worst["google"] < Decimal("0.01")


@pytest.mark.parametrize("attempts", [3, None])
async def test_concurrent_matches_admitted_by_the_formula(
    make_world: Callable[..., World], attempts: int | None
) -> None:
    world = make_world(daily=SPEC_DAILY)
    worst = real_worst()
    fits = min(int(SPEC_DAILY[p] // worst[p]) for p in PROVIDERS)  # whole matches per day
    assert fits >= 3  # three visitors at once all get in under the SPEC caps
    tries = attempts if attempts is not None else fits + 1
    now = datetime.now(UTC)
    results = await asyncio.gather(
        *(world.ledger.reserve("match", uuid4(), now) for _ in range(tries)),
        return_exceptions=True,
    )
    admitted = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, BaseException)]
    assert len(admitted) == min(tries, fits)
    assert all(isinstance(r, AIPaused) and r.kind == "day" for r in refused)
    # The Q&A path is still (3, 2, 3) calls at the estimate: unchanged by the match.
    assert world.ledger.worst("ask") == world.ledger.question_worst()


async def test_eight_parallel_topups_never_overbook(make_world: Callable[..., World]) -> None:
    """The 8 match agents of one request share one reservation; their top-ups take turns
    (`SpendScope.lock`), so what may still be spent never exceeds what is reserved."""
    world = make_world()
    cap = Decimal("0.0075")
    world.ledger.limits = SpendLimits(
        dict.fromkeys(PROVIDERS, cap), world.ledger.limits.monthly, MODELS
    )
    scope = await world.ledger.reserve(
        "match", uuid4(), amounts=dict.fromkeys(PROVIDERS, Decimal("0.001"))
    )
    worst = Decimal("0.002")  # each call needs more than is reserved for it
    results = await asyncio.gather(
        *(scope.ensure("anthropic", worst) for _ in range(8)), return_exceptions=True
    )
    ok = sum(1 for r in results if r is None)
    assert ok == 3  # 0.001 + top-ups to 0.002, 0.004, 0.006; 0.008 would pass the cap
    assert all(isinstance(r, AIPaused) for r in results if r is not None)
    assert scope.pending["anthropic"] == ok * worst <= scope.reserved["anthropic"] <= cap
    stored = world.rows(
        "SELECT usd FROM spend_reservations WHERE id = %s AND provider = 'anthropic'",
        (scope.question_id,),
    )
    assert stored == [(scope.reserved["anthropic"],)]


# --- 5. SSE shapes ------------------------------------------------------------------------


def normalised(events: list[Event]) -> list[Event]:
    out: list[Event] = []
    for name, data in events:
        data = json.loads(json.dumps(data))
        if name == "notes":
            for finding in data["findings"]:
                assert isinstance(finding["ms"], int)
                finding["ms"] = 0
        if name == "result":
            UUID(data["match_id"])
            data["match_id"] = "<uuid>"
        out.append((name, data))
    return out


def notes(findings: list[dict[str, Any]], truncated: bool = False) -> Event:
    return (
        "notes",
        {
            "requirements": [{"id": "r1", "text": "Python"}, {"id": "r2", "text": "Kubernetes"}],
            "findings": findings,
            "truncated": truncated,
        },
    )


PY_FOUND: dict[str, Any] = {
    "requirement_id": "r1",
    "status": "found",
    "sections": [{"url": "/projects/roleradar#python", "title": "Heading python"}],
    "provider": "anthropic",
    "ms": 0,
}
K8S_NOT_FOUND: dict[str, Any] = {
    "requirement_id": "r2",
    "status": "not_found",
    "sections": [],
    "provider": None,
    "ms": 0,
}
RESULT = {
    "match_id": "<uuid>",
    "items": [
        {
            "requirement_id": "r1",
            "requirement": "Python",
            "text": "guide text 1",
            "sections": [{"url": "/projects/roleradar#python", "title": "Heading python"}],
        }
    ],
    "not_covered": ["Kubernetes"],
    "unchecked": 0,
    "top8": False,
    "language": "en",
}


def expected_events(*, fetched: bool, truncated: bool) -> list[Event]:
    head: list[Event] = [("stage", {"stage": "fetching", "count": None})] if fetched else []
    return [
        *head,
        ("stage", {"stage": "reading", "count": None}),
        notes([], truncated),
        ("stage", {"stage": "matching", "count": 2}),
        notes([K8S_NOT_FOUND], truncated),
        notes([PY_FOUND, K8S_NOT_FOUND], truncated),
        ("stage", {"stage": "checking", "count": 1}),
        notes([PY_FOUND, K8S_NOT_FOUND], truncated),  # after the check (nothing changed)
        ("stage", {"stage": "writing", "count": None}),
        ("result", RESULT),
        ("done", {}),
    ]


async def test_pasted_text_event_snapshot(make_world: Callable[..., World]) -> None:
    world = make_world()
    response = await post(world, JD)
    events = parse_sse(response.text)
    assert normalised(events) == expected_events(fetched=False, truncated=False)
    # Reasoning and system prompts never reach the wire.
    assert REASONING not in response.text and "secret reasoning" not in response.text
    for probe in prompts.SYSTEM_PROMPT_PROBES:
        assert probe not in response.text
    assert world.outbound.requests == []


async def test_link_event_snapshot_with_truncation(make_world: Callable[..., World]) -> None:
    payload = json.loads((FIXTURES / "greenhouse.json").read_text())
    payload["content"] = "&lt;p&gt;" + "Python " * 4000 + "&lt;/p&gt;"  # ~28,000 characters
    world = make_world(outbound=Outbound(body=json.dumps(payload).encode()))
    events = await events_of(world, LINK)
    assert normalised(events) == expected_events(fetched=True, truncated=True)
    [request] = world.outbound.requests
    assert str(request.url) == "https://boards-api.greenhouse.io/v1/boards/acme/jobs/4005542201"
    # The model got the fetched text, cut at 20,000 characters.
    [job_ad] = world.script.job_ads
    assert "Python Python" in job_ad and len(job_ad) < MAX_JD_CHARS + 500


def test_messages_are_the_contract_table() -> None:
    assert MESSAGES == {
        "fetch_failed": "Couldn't read that link — please paste the job ad text.",
        "fetch_limit": "Too many links in a row — please paste the job ad text.",
        "not_a_job": "This doesn't look like a job ad — please paste the job description.",
        "too_long": "That's longer than 20,000 characters — please paste the main part.",
        "unavailable": "The job match isn't available right now — please try again later.",
        "timeout": "That took too long — please try again.",
        "limit_day": "Today's AI limit is reached — please email me",
        "paused_month": "The AI is paused until next month — please email me",
        "internal": "Something went wrong — please try again.",
    }


# --- 6. match_log -------------------------------------------------------------------------


async def test_one_match_log_row_with_fixed_fields_only(make_world: Callable[..., World]) -> None:
    world = make_world()
    events = await events_of(world, JD)
    match_id = UUID(events[-2][1]["match_id"])
    columns = [
        name
        for (name,) in world.rows(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'match_log' ORDER BY ordinal_position"
        )
    ]
    assert columns == [
        "id",
        "created_at",
        "source",
        "job_language",
        "prompt_version",
        "providers",
        "cost_usd",
        "ip_hash",
        "thumbs_down",
    ]
    [row] = world.rows(
        "SELECT id, source, job_language, prompt_version, providers, cost_usd, ip_hash, "
        "thumbs_down FROM match_log"
    )
    assert row[:5] == (match_id, "text", "en", prompts.PROMPT_VERSION, ["anthropic", "google"])
    assert row[5] > 0 and len(row[6]) == 32 and row[7] is False
    # Nothing from the job ad, the requirements or the result is in the database.
    dump = repr(world.rows("SELECT * FROM match_log")) + repr(
        world.rows("SELECT * FROM usage_counters")
    )
    for text in ("Python", "Kubernetes", "guide text", "benefits"):
        assert text not in dump
    # Settled: no reservation left, spending booked, one match counted.
    assert world.rows("SELECT * FROM spend_reservations") == []
    spent = dict(world.rows("SELECT provider, spent_usd FROM spend_daily"))
    assert spent["anthropic"] + spent["google"] == row[5]
    assert world.counts("match") == {"token": 1, "ip": 1, "site": 1}


async def test_thumbs_down_on_a_match(make_world: Callable[..., World]) -> None:
    world = make_world()
    events = await events_of(world, JD)
    match_id = events[-2][1]["match_id"]
    async with world.client() as http:
        assert (await http.post("/api/feedback", json={"match_id": match_id})).status_code == 204
        both = {"match_id": match_id, "turn_id": str(uuid4())}
        assert (await http.post("/api/feedback", json=both)).status_code == 422
        assert (await http.post("/api/feedback", json={})).status_code == 422
    assert world.rows("SELECT thumbs_down FROM match_log") == [(True,)]


# --- 2 and 3. fetching, counting and refunds ------------------------------------------------


@pytest.mark.parametrize(("status", "body"), [(404, b"{}"), (429, b"{}"), (200, b"not json")])
async def test_fetch_failure_counts_fetch_not_match(
    make_world: Callable[..., World], status: int, body: bytes
) -> None:
    world = make_world(outbound=Outbound(status=status, body=body))
    events = await events_of(world, LINK)
    assert events == [
        ("stage", {"stage": "fetching", "count": None}),
        ("error", {"code": "fetch_failed", "message": MESSAGES["fetch_failed"]}),
    ]
    assert len(world.outbound.requests) == 1  # no retry, 429 included
    assert world.counts("fetch") == {"token": 1, "ip": 1, "site": 1}
    assert world.counts("match") == {}
    assert world.rows("SELECT * FROM match_log") == []
    assert world.rows("SELECT * FROM spend_reservations") == []


async def test_fetch_timeout_is_fetch_failed(make_world: Callable[..., World]) -> None:
    world = make_world()

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    world.service._client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    assert error_of(await events_of(world, LINK))["code"] == "fetch_failed"
    assert world.counts("match") == {}


async def test_fourth_link_in_a_minute_is_fetch_limit(make_world: Callable[..., World]) -> None:
    world = make_world(outbound=Outbound(status=404))
    codes = [error_of(await events_of(world, LINK))["code"] for _ in range(4)]
    assert codes == ["fetch_failed"] * 3 + ["fetch_limit"]
    assert len(world.outbound.requests) == 3  # the fourth never went out
    world.clock.advance(seconds=61)
    assert error_of(await events_of(world, LINK))["code"] == "fetch_failed"


async def test_eleventh_link_in_a_day_is_fetch_limit(make_world: Callable[..., World]) -> None:
    world = make_world(outbound=Outbound(status=404))
    codes = []
    for _ in range(11):
        codes.append(error_of(await events_of(world, LINK))["code"])
        world.clock.advance(seconds=61)
    assert codes == ["fetch_failed"] * 10 + ["fetch_limit"]
    assert len(world.outbound.requests) == 10
    assert world.counts("fetch") == {"token": 10, "ip": 10, "site": 10}


async def test_site_wide_hundred_and_first_fetch_is_fetch_limit(
    make_world: Callable[..., World],
) -> None:
    world = make_world()
    day = sydney_day(world.clock.now)
    world.execute(
        "INSERT INTO usage_counters (day, kind, scope, key, count) VALUES (%s, 'fetch', 'site', "
        "'\\x'::bytea, 100)",
        (day,),
    )
    assert error_of(await events_of(world, LINK))["code"] == "fetch_limit"
    assert world.outbound.requests == []


async def test_day_used_up_rejects_before_fetching(make_world: Callable[..., World]) -> None:
    world = make_world(match_limit_token=1)
    assert (await events_of(world, JD))[-1] == ("done", {})
    response = await post(world, LINK)
    assert response.status_code == 429
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"code": "limit_day", "message": MESSAGES["limit_day"]}
    assert world.outbound.requests == []  # no outbound request
    assert world.counts("fetch") == {}
    assert world.counts("match") == {"token": 1, "ip": 1, "site": 1}


async def test_ai_resting_rejects_before_fetching(make_world: Callable[..., World]) -> None:
    world = make_world()
    await world.ledger.pause_provider("google")
    response = await post(world, LINK)
    assert (response.status_code, response.json()["code"]) == (503, "paused_month")
    assert world.outbound.requests == [] and world.counts("fetch") == {}


async def test_match_day_cap_too_small_rejects_match_but_not_ask(
    make_world: Callable[..., World],
) -> None:
    # Room for a question's worst path but not a match's: the pause check is per kind.
    world = make_world()
    worst = real_worst()
    caps = {p: max(worst[p], world.ledger.question_worst()[p]) for p in PROVIDERS}
    caps["anthropic"] = (worst["anthropic"] + world.ledger.question_worst()["anthropic"]) / 2
    assert world.ledger.question_worst()["anthropic"] < caps["anthropic"] < worst["anthropic"]
    world.ledger.limits = SpendLimits(caps, world.ledger.limits.monthly, MODELS)
    now = datetime.now(UTC)
    assert await world.ledger.paused(now, "match") == "day"
    assert await world.ledger.paused(now) is None
    response = await post(world, JD)
    assert (response.status_code, response.json()["code"]) == (429, "limit_day")
    assert world.counts("match") == {}


async def test_too_long_text_counts_nothing(make_world: Callable[..., World]) -> None:
    world = make_world()
    events = await events_of(world, "x" * (MAX_JD_CHARS + 1))
    assert events == [("error", {"code": "too_long", "message": MESSAGES["too_long"]})]
    assert world.counts("match") == {} and world.script.calls == []
    assert (await post(world, "x" * 50_001)).status_code == 422
    assert (await events_of(world, "Python " * (MAX_JD_CHARS // 7)))[-1] == ("done", {})


async def test_failed_reservation_leaves_the_count_alone(
    make_world: Callable[..., World], monkeypatch: pytest.MonkeyPatch
) -> None:
    world = make_world()

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AIPaused("day")

    monkeypatch.setattr(world.ledger, "reserve", refuse)
    assert error_of(await events_of(world, JD)) == {
        "code": "limit_day",
        "message": MESSAGES["limit_day"],
    }
    assert world.counts("match") == {}
    assert world.rows("SELECT * FROM match_log") == [] and world.script.calls == []

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise psycopg.OperationalError("database went away")

    monkeypatch.setattr(world.ledger, "reserve", broken)
    assert error_of(await events_of(world, JD))["code"] == "unavailable"
    assert world.counts("match") == {}


async def test_not_a_job_gives_the_count_back(make_world: Callable[..., World]) -> None:
    world = make_world(script=Script(requirements=[]), match_limit_token=1)
    events = await events_of(world, "What's the weather like?")
    assert events == [
        ("stage", {"stage": "reading", "count": None}),
        ("error", {"code": "not_a_job", "message": MESSAGES["not_a_job"]}),
    ]
    assert world.counts("match") == {"token": 0, "ip": 0, "site": 0}
    # Reading it still cost money and is on the record.
    [(source, cost, providers)] = world.rows("SELECT source, cost_usd, providers FROM match_log")
    assert (source, providers) == ("text", ["anthropic"]) and cost > 0
    assert dict(world.rows("SELECT provider, spent_usd FROM spend_daily")) == {"anthropic": cost}
    # With a limit of 1 a day, the visitor can still run a real match.
    world.script.requirements = ["Python"]
    assert (await events_of(world, JD))[-1] == ("done", {})
    assert world.counts("match") == {"token": 1, "ip": 1, "site": 1}


async def test_refund_clears_a_full_site_mark(make_world: Callable[..., World]) -> None:
    world = make_world(script=Script(requirements=[]), match_limit_site=1)
    await events_of(world, "not a job")
    # A second visitor, after the refund: the site has its one match back.
    world.script.requirements = ["Python"]
    other, _ = sign_token(world.settings.session_secret.get_secret_value(), time.time())
    assert (await events_of(world, JD, ip="192.0.2.9", token=other))[-1] == ("done", {})


# --- 4. deadlines ---------------------------------------------------------------------------


def test_app_uses_spec_stage_timeouts_and_a_120_second_request() -> None:
    settings = guard_settings("postgresql://nobody@127.0.0.1:1/none", openai_api_key="sk-test")
    db = Database(make_pool(settings.database_url.get_secret_value()))
    ledger = Ledger(db, SpendLimits.from_settings(settings), MAX_OUTPUT_TOKENS)
    service = MatchService.for_app(settings, db, ledger, Limits(db, settings, ledger))
    assert service.timeout_s == MATCH_TIMEOUT_S == 120.0
    deps = service._make_deps()
    assert dict(deps.stage_timeouts) == dict(DEFAULT_STAGE_TIMEOUTS)
    assert dict(DEFAULT_STAGE_TIMEOUTS) == {
        "reading": 25.0,
        "matching": 35.0,
        "checking": 25.0,
        "writing": 25.0,
    }
    assert ledger.worst("match") == match_worst_for(settings)


async def test_whole_request_deadline_settles_what_was_spent(
    make_world: Callable[..., World],
) -> None:
    script = Script()
    started = asyncio.Event()

    async def slow_match(messages: list[BaseMessage]) -> Any:
        started.set()
        await asyncio.sleep(10)
        return {"reasoning": "", "found": False, "sections": []}

    script.match = slow_match  # type: ignore[method-assign]
    world = make_world(script=script, timeout_s=0.5)
    events = await events_of(world, JD)
    assert events[-1] == ("error", {"code": "timeout", "message": MESSAGES["timeout"]})
    assert started.is_set()
    # Counted (no refund), reservation settled: the read at what the fake reported (its
    # worst, it reports no usage) plus the cancelled match call at its worst.
    assert world.counts("match") == {"token": 1, "ip": 1, "site": 1}
    assert world.rows("SELECT * FROM spend_reservations") == []
    [(cost,)] = world.rows("SELECT cost_usd FROM match_log")
    spent = dict(world.rows("SELECT provider, spent_usd FROM spend_daily"))
    assert spent == {"anthropic": cost} and cost > 0


async def test_stage_timeout_on_reading_is_unavailable(make_world: Callable[..., World]) -> None:
    script = Script()

    async def slow_read(messages: list[BaseMessage]) -> Any:
        await asyncio.sleep(5)
        return {}

    script.read = slow_read  # type: ignore[method-assign]
    world = make_world(script=script, stage={"reading": 0.2})
    assert error_of(await events_of(world, JD))["code"] == "unavailable"
    assert world.counts("match") == {"token": 1, "ip": 1, "site": 1}  # not refunded
    assert world.rows("SELECT * FROM spend_reservations") == []


# --- 5. the rest of the error table ----------------------------------------------------------


class Raising:
    """A stand-in compiled graph that raises `error` after its first stage event."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield "custom", {"stage": "reading", "count": None}
        raise self.error


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ModelsUnavailable("read: every provider failed"), "unavailable"),
        (AIPaused("day"), "limit_day"),
        (AIPaused("month"), "paused_month"),
        (match_api.ProviderBudgetExhausted("anthropic"), "paused_month"),
        (RuntimeError("boom with job ad text"), "internal"),
    ],
)
async def test_graph_errors_map_to_codes_and_keep_the_count(
    make_world: Callable[..., World], error: BaseException, code: str, caplog
) -> None:
    world = make_world()
    world.service._graph = Raising(error)
    events = await events_of(world, JD)
    assert events == [
        ("stage", {"stage": "reading", "count": None}),
        ("error", {"code": code, "message": MESSAGES[code]}),
    ]
    assert world.counts("match") == {"token": 1, "ip": 1, "site": 1}
    assert len(world.rows("SELECT * FROM match_log")) == 1
    assert "job ad text" not in caplog.text and "Kubernetes" not in caplog.text


async def test_logs_have_no_job_ad(make_world: Callable[..., World], caplog) -> None:
    world = make_world()
    with caplog.at_level("INFO"):
        await events_of(world, JD)
    assert "match done" in caplog.text
    for text in ("Python", "Kubernetes", "benefits", "guide text"):
        assert text not in caplog.text


async def test_bad_bodies_are_422_and_count_nothing(make_world: Callable[..., World]) -> None:
    world = make_world()
    async with world.client() as http:
        for body in ({}, {"input": ""}, {"input": "  "}, {"input": "x", "extra": 1}):
            assert (await http.post("/api/match", json=body)).status_code == 422
    assert world.counts("match") == {} and world.counts("fetch") == {}


async def test_no_token_is_401(make_world: Callable[..., World]) -> None:
    world = make_world()
    async with world.client(token="nope") as http:
        assert (await http.post("/api/match", json={"input": JD})).status_code == 401


async def test_three_concurrent_matches_all_finish(make_world: Callable[..., World]) -> None:
    world = make_world(daily=SPEC_DAILY)

    async def one(ip: str) -> list[Event]:
        token, _ = sign_token(world.settings.session_secret.get_secret_value(), time.time())
        async with world.client(ip=ip, token=token) as http:
            return parse_sse((await http.post("/api/match", json={"input": JD})).text)

    results = await asyncio.gather(*(one(f"192.0.2.{n}") for n in (1, 2, 3)))
    await world.service.wait_idle()
    assert [r[-1] for r in results] == [("done", {})] * 3
    assert len(world.rows("SELECT * FROM match_log")) == 3
    assert world.rows("SELECT * FROM spend_reservations") == []


def test_reserved_once_per_request_window_is_longer_than_the_request() -> None:
    from app.guard.ledger import RESERVATION_TTL_S

    assert RESERVATION_TTL_S > MATCH_TIMEOUT_S
    assert timedelta(seconds=RESERVATION_TTL_S) > timedelta(seconds=MATCH_TIMEOUT_S)
