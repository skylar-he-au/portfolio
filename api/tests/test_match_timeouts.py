"""Time limits of the job match (SPEC "出错有退路", "职位对照" 时间): a call that never answers,
a spend reservation that never returns, and a server that accepts the request and goes
silent all end at the stage's deadline, and the stages fit in the 120-second request.

Round 2 (r2k): one item of the combined eval run stalled for about 15 minutes. These tests
pin each layer the limit has to cover; the reservation tests failed before `ask_json` put
the spend reservation inside the per-call time limit.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import pytest
from fakes import FakeJsonChat
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from test_match_graph import (
    JD,
    Script,
    one_section_each,
    reqs,
    requirement_of,
    section,
)

from app.api.match import MATCH_TIMEOUT_S
from app.config import Settings
from app.guard.ledger import current_scope
from app.match import graph as match_graph
from app.match import models as mm
from app.match import prompts
from app.match.graph import (
    DEFAULT_STAGE_TIMEOUTS,
    MatchDeps,
    build_match_graph,
    check_step,
    match_one,
    read_step,
    write_step,
)
from app.match.models import StepProvider
from app.match.state import Finding, Requirement, SectionRef
from app.qa.models import ChatProvider
from app.rag.types import ModelsUnavailable

pytestmark = pytest.mark.anyio

STAGE = 0.3  # every stage's deadline in these tests, seconds
SLACK = 0.4  # what a stage may run over its deadline (scheduling, the fallback's bookkeeping)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def forever(*_: Any) -> Any:
    await asyncio.Event().wait()


class HangingScope:
    """A `SpendScope` whose reservation top-up never returns: the provider row lock held
    by another match, or a database that stopped answering."""

    def __init__(self) -> None:
        self.ensured = 0
        self.finished = 0

    async def ensure(self, provider: str, worst: Decimal) -> None:
        self.ensured += 1
        await forever()

    def finish(self, provider: str, worst: Decimal, actual: Decimal | None) -> None:
        self.finished += 1


def deps_for(script: Callable[[list[BaseMessage]], Any], **stage: float) -> MatchDeps:
    chat = FakeJsonChat(reply=script)
    gemini = FakeJsonChat(reply=script)
    return MatchDeps(
        ChatProvider("anthropic", "claude-haiku-5-5", chat),
        ChatProvider("google", "gemini-3.5-flash-lite", gemini),
        one_section_each(["Python", "SQL"]),
        stage_timeouts={  # type: ignore[arg-type]
            "reading": STAGE,
            "matching": STAGE,
            "checking": STAGE,
            "writing": STAGE,
            **stage,
        },
        call_timeout=30.0,  # longer than any stage: the stage deadline must end the call
    )


@asynccontextmanager
async def hang(where: str) -> AsyncIterator[Callable[[list[BaseMessage]], Any]]:
    """The model replies (`Script`) with the hang `where` says: "model" = every call
    never answers; "reservation" = every spend reservation never returns."""
    script = Script(reqs("Python"))
    if where == "model":
        for name in ("read", "match", "check", "write"):
            setattr(script, name, forever)
        yield script
        return
    token = current_scope.set(HangingScope())  # type: ignore[arg-type]
    try:
        yield script
    finally:
        current_scope.reset(token)


REQ = Requirement(id="r1", text="Python", query="q Python")
SEC = section("roleradar", "sec-python")
FOUND = Finding(
    requirement_id="r1",
    status="found",
    sections=[SectionRef(url=SEC.url, title=SEC.heading)],
    provider="anthropic",
    ms=1,
)


async def elapsed(call: Callable[[], Any]) -> tuple[float, Any]:
    started = time.monotonic()
    try:
        async with asyncio.timeout(STAGE + 5):  # a backstop, so a missed deadline fails fast
            result: Any = await call()
    except Exception as error:  # noqa: BLE001 - the test looks at what came out
        result = error
    return time.monotonic() - started, result


@pytest.mark.parametrize("where", ["model", "reservation"])
async def test_reading_ends_at_its_deadline(where: str) -> None:
    async with hang(where) as script:
        took, result = await elapsed(lambda: read_step(JD, deps_for(script)))
    assert isinstance(result, ModelsUnavailable)
    assert took < STAGE + SLACK


@pytest.mark.parametrize("where", ["model", "reservation"])
async def test_one_match_ends_at_its_deadline_as_unchecked(where: str) -> None:
    async with hang(where) as script:
        took, finding = await elapsed(lambda: match_one(REQ, [SEC], deps_for(script)))
    assert isinstance(finding, Finding) and finding.status == "unchecked"
    assert took < STAGE + SLACK


@pytest.mark.parametrize("where", ["model", "reservation"])
async def test_checking_ends_at_its_deadline_and_unchecks(where: str) -> None:
    async with hang(where) as script:
        took, result = await elapsed(
            lambda: check_step([REQ], [FOUND], {"r1": [SEC]}, deps_for(script))
        )
    assert [f.status for f in result.findings] == ["unchecked"]
    assert [v.shows for v in result.verdicts] == ["missing"]
    assert took < STAGE + SLACK


@pytest.mark.parametrize("where", ["model", "reservation"])
async def test_writing_ends_at_its_deadline(where: str) -> None:
    async with hang(where) as script:
        took, result = await elapsed(
            lambda: write_step([REQ], [FOUND], {"r1": [SEC]}, "en", False, deps_for(script))
        )
    assert isinstance(result, ModelsUnavailable)
    assert took < STAGE + SLACK


async def test_the_reservation_is_settled_when_its_call_times_out() -> None:
    """A call cut off after its reservation went through is charged at its worst case."""

    class SlowCall(HangingScope):
        async def ensure(self, provider: str, worst: Decimal) -> None:
            self.ensured += 1  # reserved at once; the call itself hangs

    scope = SlowCall()
    token = current_scope.set(scope)  # type: ignore[arg-type]
    try:
        script = Script(reqs("Python"))
        script.read = forever  # type: ignore[method-assign]
        took, result = await elapsed(lambda: read_step(JD, deps_for(script)))
    finally:
        current_scope.reset(token)
    assert isinstance(result, ModelsUnavailable) and took < STAGE + SLACK
    assert scope.ensured == scope.finished == 1  # Gemini got no time left: never reserved


async def test_a_run_whose_match_and_check_hang_ends_in_matching_plus_checking_time() -> None:
    """read and write answer; the SQL match agent and the check hang. The run goes through
    all four steps: SQL unchecked at the matching deadline, Python unchecked at the
    checking deadline (the check never answered), so nothing is left for write to call."""
    script = Script(reqs("Python", "SQL"))
    answer = script.match

    async def match(messages: list[BaseMessage]) -> Any:
        if requirement_of(messages) == "SQL":
            return await forever()
        return await answer(messages)

    script.match = match  # type: ignore[method-assign]
    script.check = forever  # type: ignore[method-assign]
    deps = deps_for(script)
    started = time.monotonic()
    final: dict[str, Any] = {}
    stages: list[str] = []
    async for mode, data in build_match_graph(deps).astream(
        {"jd": JD}, stream_mode=["custom", "values"]
    ):
        if mode == "values":
            final = data
        elif "stage" in data:
            stages.append(data["stage"])
    took = time.monotonic() - started
    assert stages == ["reading", "matching", "checking", "writing"]
    assert [f.status for f in final["findings"]] == ["found", "unchecked"]
    assert [f.status for f in final["checked_findings"]] == ["unchecked", "unchecked"]
    assert final["result"].unchecked == 2
    assert 2 * STAGE <= took < 2 * STAGE + 2 * SLACK


def test_stages_fit_in_the_request_deadline() -> None:
    """SPEC: fetch 8 s + read 25 + match 35 + check 25 + write 25 = 118 s <= 120 s."""
    assert dict(DEFAULT_STAGE_TIMEOUTS) == {
        "reading": 25.0,
        "matching": 35.0,
        "checking": 25.0,
        "writing": 25.0,
    }
    assert 8.0 + sum(DEFAULT_STAGE_TIMEOUTS.values()) <= MATCH_TIMEOUT_S == 120.0


# --- each provider's own client timeout and retries -----------------------------------------


def settings() -> Settings:
    return Settings(
        anthropic_api_key=SecretStr("sk-ant-test"),
        gemini_api_key=SecretStr("gemini-test"),
        openai_api_key=SecretStr("sk-openai-test"),
    )  # type: ignore[call-arg]


HAIKU = mm.ModelChoice("anthropic", "claude-haiku-5-5", "off")
HAIKU_LOW = mm.ModelChoice("anthropic", "claude-haiku-5-5", "low")
FLASH = mm.ModelChoice("google", "gemini-3.5-flash-lite", "minimal")
FLASH_LOW = mm.ModelChoice("google", "gemini-3.5-flash-lite", "low")
LUNA = mm.ModelChoice("openai", "gpt-6-luna", "none")
LUNA_LOW = mm.ModelChoice("openai", "gpt-6-luna", "low")
CHOICES = [HAIKU, HAIKU_LOW, FLASH, FLASH_LOW, LUNA, LUNA_LOW]


def client_timeout(chat: Any) -> Any:
    if isinstance(chat, ChatAnthropic):
        return chat.default_request_timeout
    if isinstance(chat, ChatOpenAI):
        return chat.request_timeout
    return chat.timeout


@pytest.mark.parametrize("choice", CHOICES, ids=str)
def test_every_client_sets_its_timeout_and_retries_within_the_shortest_stage(
    choice: mm.ModelChoice,
) -> None:
    chat = mm.build_chat(choice, settings(), 1024)
    timeout, retries = mm.client_limits(choice)
    assert client_timeout(chat) == timeout
    assert chat.max_retries == retries
    # every attempt together fits in the shortest stage, with room for a retry's backoff
    assert mm.client_attempts(choice) * timeout <= mm.CLIENT_BUDGET_S
    assert mm.CLIENT_BUDGET_S + 2.0 <= min(DEFAULT_STAGE_TIMEOUTS.values())
    assert timeout <= match_graph.CALL_TIMEOUT_S  # SPEC: 30 s per model call


def test_the_default_steps_use_these_clients() -> None:
    steps = mm.build_step_providers(mm.models_for(settings()), settings())
    for providers in steps.values():
        for provider in providers:
            assert isinstance(provider, StepProvider)
            choice = mm.ModelChoice(provider.name, provider.model, provider.effort)
            assert client_timeout(provider.chat) == mm.client_limits(choice)[0]


class SilentServer:
    """Accepts HTTP requests and never answers (or sends half a response and stalls)."""

    def __init__(self, partial: bool) -> None:
        self.partial = partial
        self.requests = 0
        self.server: asyncio.Server | None = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                self.requests += 1
                length = next(
                    (
                        int(line.split(b":", 1)[1])
                        for line in head.split(b"\r\n")
                        if line.lower().startswith(b"content-length:")
                    ),
                    0,
                )
                await reader.readexactly(length)
                if self.partial:
                    writer.write(
                        b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                        b'content-length: 500\r\n\r\n{"id": '
                    )
                    await writer.drain()
                await forever()
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
            writer.close()

    async def __aenter__(self) -> str:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def __aexit__(self, *_: Any) -> None:
        assert self.server is not None
        self.server.close()


def point_at(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """`build_chat` with every client's base URL set to `url`."""
    for name, cls, value in (
        ("ChatAnthropic", ChatAnthropic, url),
        ("ChatOpenAI", ChatOpenAI, f"{url}/v1"),
        ("ChatGoogleGenerativeAI", ChatGoogleGenerativeAI, url),
    ):
        monkeypatch.setattr(mm, name, lambda cls=cls, value=value, **kw: cls(base_url=value, **kw))


@pytest.mark.parametrize("partial", [False, True], ids=["silent", "half-sent"])
@pytest.mark.parametrize("choice", [HAIKU, FLASH, LUNA], ids=str)
async def test_the_client_timeout_ends_a_call_to_a_silent_server(
    choice: mm.ModelChoice, partial: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the graph's deadline the client's own timeout ends the call, after the
    number of attempts it is set to (Gemini: the seconds reach the SDK as milliseconds).

    What the client raises depends on its transport: google-genai uses aiohttp when it is
    installed (the eval group brings it in with ragas), and aiohttp's own timeout is a
    `TimeoutError`; with httpx (CI, the Docker image) it is the SDK's error. So the test
    asks the backstop whether it fired instead of looking at the exception's type."""
    monkeypatch.setattr(mm, "RETRY_TIMEOUT_S", 0.3)
    monkeypatch.setattr(mm, "CLIENT_BUDGET_S", 0.6)
    server = SilentServer(partial)
    async with server as url:
        point_at(monkeypatch, url)
        chat = mm.build_chat(choice, settings(), 64)
        structured = chat.with_structured_output(prompts.CheckOut, method="json_schema")
        started = time.monotonic()
        backstop = asyncio.timeout(10)  # only a backstop: must not be what fires
        with pytest.raises(Exception) as caught:
            async with backstop:
                await structured.ainvoke([("user", "hi")])
        took = time.monotonic() - started
    assert not backstop.expired(), caught.value
    assert server.requests == mm.client_attempts(choice)
    timeout, _ = mm.client_limits(choice)
    assert took < mm.client_attempts(choice) * timeout + 1.5  # + the retry's backoff


@pytest.mark.parametrize("partial", [False, True], ids=["silent", "half-sent"])
async def test_ask_json_ends_a_silent_provider_at_the_stage_deadline_and_falls_back(
    partial: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real client whose server goes silent: the stage deadline ends it (the client's
    own timeout is longer) and the fallback gets what is left."""
    server = SilentServer(partial)
    async with server as url:
        point_at(monkeypatch, url)
        silent = StepProvider(
            "anthropic", HAIKU.model, mm.build_chat(HAIKU, settings(), 64), effort="off"
        )
        answers = ChatProvider(
            "google",
            FLASH.model,
            FakeJsonChat(reply=lambda _: {"verdicts": [{"n": 1, "shows": True}]}),
        )
        started = time.monotonic()
        out, used = await match_graph.ask_json(
            "check",
            [silent, answers],
            prompts.CheckOut,
            prompts.check_messages([(1, REQ, SEC)], prompts.new_marker()),
            deadline=time.monotonic() + 1.0,
            call_timeout=0.5,
        )
        took = time.monotonic() - started
    assert used is answers and out.verdicts[0].shows
    assert server.requests == 1
    assert 0.5 <= took < 1.0


# --- the API judges of the eval runners (step 7 7t) --------------------------------------------

JUDGE_LIMIT = 0.2  # every judge's time limit in these tests, seconds


async def test_a_match_judge_that_never_answers_is_a_judge_error_not_a_zero(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`evals/match_run.py`: the alignment call and the Faithfulness score each end at their
    limit; the item keeps its row with the score missing (`judge_errors`), and the summary
    counts it as missing, not as 0. Before the fix both waited forever."""
    import json

    from test_evals_match import EvalScript, FakeAligner, FakeScorer, _evaluate, eval_deps

    from evals import match_run

    class Hangs:
        calls = 0

        async def __call__(self, *args: Any) -> Any:
            self.calls += 1
            await forever()

    monkeypatch.setattr(match_run, "ALIGN_TIMEOUT_S", JUDGE_LIMIT)
    monkeypatch.setattr(match_run, "FAITH_TIMEOUT_S", JUDGE_LIMIT)
    for aligner, scorer, errors in (
        (Hangs(), FakeScorer(0.9), ["align: TimeoutError"]),
        (FakeAligner(), Hangs(), ["faithfulness: TimeoutError"]),
    ):
        out = tmp_path / str(len(errors[0]))
        started = time.monotonic()
        async with asyncio.timeout(5):  # only a backstop
            outcome = await _evaluate(out, eval_deps(EvalScript()), aligner=aligner, scorer=scorer)
        took = time.monotonic() - started
        row = outcome.rows[0]
        assert outcome.status == "complete" and row["status"] == "ok"
        assert row["judge_errors"] == errors
        summary = json.loads((out / "summary" / "run-1.json").read_text(encoding="utf-8"))
        if errors[0].startswith("align"):
            assert row["scores"] is None and row["faithfulness"] == 0.9
            assert summary["scores"]["tune"]["faithfulness"]["missing"] == 0
        else:
            assert row["scores"] is not None and row["faithfulness"] is None
            assert summary["scores"]["tune"]["faithfulness"]["missing"] == 1
        assert took < JUDGE_LIMIT + 1.0


async def test_a_qa_judge_that_never_answers_leaves_the_scores_missing(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`evals/run.py`: the RAGAS scorer of one sample ends at `SCORE_TIMEOUT_S` and the
    sample's scores are missing, like any judge failure."""
    from test_evals_run import run_eval, summary

    from evals import run as qa_run
    from evals.types import METRICS

    async def hangs(*args: Any) -> Any:
        await forever()

    monkeypatch.setattr(qa_run, "SCORE_TIMEOUT_S", JUDGE_LIMIT)
    started = time.monotonic()
    async with asyncio.timeout(5):  # only a backstop
        outcome = await run_eval(tmp_path, scorer=hangs, no_ragas=False)
    took = time.monotonic() - started
    row = outcome.rows[0]
    assert outcome.status == "complete" and row["status"] == "ok"
    assert row["ragas"] == dict.fromkeys(METRICS)
    assert summary(outcome.rows)["ragas_nulls"] == len(METRICS)
    assert took < JUDGE_LIMIT + 1.0
