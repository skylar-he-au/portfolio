"""`POST /api/match`: a job ad (link or text) in, a "where to look" guide out as SSE.

    {"input": "a Greenhouse / Lever link, or the job ad text"}  ->  text/event-stream

Events (contract in `docs/briefs/step6.md`): `stage` at the start of each stage, `notes`
whenever the shared notes change (the whole notes each time), `result` once checked, then
`done`; or `error` (`code`, `message`) at any point, after which nothing else is sent.

Order of the guards, cheapest first:

1. Before the stream (HTTP errors, like `/api/ask`): Worker secret and session token
   (403 / 401), body (422), the in-process sieve (`Limits.sieve`: a known full count or the
   per-IP window, 429), then one read-only look: is the AI resting for a match
   (`Ledger.paused(now, "match")`) and does this visitor have a match left today
   (`Limits.peek`)? If not: 429 `limit_day` / 503 `paused_month`, and no job ad is fetched.
2. In the stream: a link is fetched only after its own count (`admit("fetch")`: per token
   and per IP 3 a minute and 10 a day, the site 100 a day; failures count too), with the
   fetch's 8 s. Pasted text over 20,000 characters is `too_long`. Then the match count and
   the spending reservation, in one transaction (`admit("match", hold=reserve)`): both or
   neither. Then the graph (`app.match.graph`), each stage with its own timeout, the whole
   request within `MATCH_TIMEOUT_S`.
3. A job ad with no requirements (`not_a_job`) gives the match count back; what reading it
   cost is still booked. Every request that got a reservation is settled and gets one
   `match_log` row (no job ad, requirements or result).

The job ad and the models' reasoning stay in this request's memory. Logs carry usage only
(timings, counts, providers, error types).
"""

import asyncio
import logging
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol, cast

import httpx
import psycopg
from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
from openai import APIError as OpenAIError
from psycopg import AsyncConnection
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.config import Settings
from app.db import Database
from app.guard.ledger import AIPaused, Ledger, ProviderBudgetExhausted, SpendScope, current_scope
from app.guard.limits import Admission, Limits
from app.guard.prices import cost_usd
from app.guard.session import REJECTIONS, Rejected, Session, require_session
from app.guard.usage import estimate_input_tokens
from app.match import models as match_models
from app.match import prompts
from app.match.fetch import FetchError, JobRef, fetch_job, parse_job_url
from app.match.graph import (
    DEFAULT_STAGE_TIMEOUTS,
    MAX_LINKS,
    MAX_QUERY_CHARS,
    MAX_SECTIONS,
    MatchDeps,
    build_match_deps,
    build_match_graph,
)
from app.match.state import MAX_REQUIREMENT_CHARS, MAX_REQUIREMENTS, Guide, Requirement, Stage
from app.qa.models import MAX_OUTPUT_TOKENS
from app.rag.types import ModelsUnavailable, Section
from app.records import MatchRecord, MatchRecords

logger = logging.getLogger(__name__)

MATCH_TIMEOUT_S = 120.0
MAX_JD_CHARS = 20_000
# A body longer than this is a 422 (it can't be a job ad worth reading); between
# MAX_JD_CHARS and this the visitor gets `too_long` with a sentence to act on.
MAX_BODY_CHARS = 50_000
# Longest current project-page section is 2,156 characters (2026-11-19); a longer one is
# covered by the normal top-up before the call.
SECTION_CHARS_ESTIMATE = 2_400

Code = Literal[
    "fetch_failed",
    "fetch_limit",
    "not_a_job",
    "too_long",
    "unavailable",
    "timeout",
    "limit_day",
    "paused_month",
    "internal",
]

MESSAGES: dict[str, str] = {
    "fetch_failed": "Couldn't read that link — please paste the job ad text.",
    "fetch_limit": "Too many links in a row — please paste the job ad text.",
    "not_a_job": "This doesn't look like a job ad — please paste the job description.",
    "too_long": "That's longer than 20,000 characters — please paste the main part.",
    "unavailable": "The job match isn't available right now — please try again later.",
    "timeout": "That took too long — please try again.",
    "limit_day": REJECTIONS["limit_day"][1],
    "paused_month": REJECTIONS["paused_month"][1],
    "internal": "Something went wrong — please try again.",
}

# Failures of a service we depend on (both model providers, the embedding API, the
# database): the visitor can only wait and retry.
UNAVAILABLE = (ModelsUnavailable, psycopg.Error, PoolTimeout, OpenAIError)
DB_ERRORS = (psycopg.Error, PoolTimeout, OSError, TimeoutError)

type Event = tuple[str, dict[str, Any]]


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_BODY_CHARS)
    ]


# --- the worst path the reservation covers --------------------------------------------------


def match_worst(
    answer_model: str,
    check_model: str,
    embed_model: str,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Decimal]:
    """`match_worst_steps` for the models before round 2: Claude (`answer_model`) reads,
    matches and writes, Gemini (`check_model`) checks, every call at `max_output_tokens`."""
    return match_worst_steps(
        match_models.default_models(answer_model, check_model, max_output_tokens), embed_model
    )


def match_worst_steps(models: match_models.MatchModels, embed_model: str) -> dict[str, Decimal]:
    """What one match can cost on its main path, per provider pool, from the real prompts
    at their largest: the read step reads a 20,000-character ad, 12 (`MAX_REQUIREMENTS`)
    match agents see 6 sections each, the write step writes the guide (12 findings x 3
    sections), the check checks 36 (requirement, section) pairs; OpenAI embeds 12 search
    phrases. Each step is charged to its first model's provider at that model's price and
    the step's own output cap. A switch of provider is not reserved here: it tops up before its call
    (`SpendScope.ensure`)."""
    marker = "0" * 16
    requirement = Requirement(
        id="r1", text="x" * MAX_REQUIREMENT_CHARS, query="x" * MAX_QUERY_CHARS
    )
    section = Section(
        "/projects/" + "x" * 30, "x" * 40, "x" * 80, "x" * SECTION_CHARS_ESTIMATE, 0.0
    )
    numbers = range(1, MAX_REQUIREMENTS + 1)
    inputs: dict[match_models.Step, tuple[int, int]] = {  # step -> (calls, input tokens)
        "read": (1, estimate_input_tokens(prompts.read_messages("x" * MAX_JD_CHARS, marker))),
        "match": (
            MAX_REQUIREMENTS,
            estimate_input_tokens(
                prompts.match_messages(requirement, [section] * MAX_SECTIONS, marker)
            ),
        ),
        "check": (
            1,
            estimate_input_tokens(
                prompts.check_messages(
                    [(n, requirement, section) for n in range(1, MAX_REQUIREMENTS * MAX_LINKS + 1)],
                    marker,
                )
            ),
        ),
        "write": (
            1,
            estimate_input_tokens(
                prompts.write_messages(
                    [(n, requirement, [section] * MAX_LINKS) for n in numbers], "en", marker
                )
            ),
        ),
    }
    worst = dict.fromkeys(match_models.PROVIDERS, Decimal(0))
    for step in match_models.STEPS:
        calls, input_tokens = inputs[step]
        first = models[step].first
        one = cost_usd(
            first.model,
            {"input_tokens": input_tokens, "output_tokens": models[step].max_output_tokens},
        )
        worst[first.provider] += calls * one
    embed = (MAX_REQUIREMENTS * MAX_QUERY_CHARS + 2) // 3
    worst["openai"] += cost_usd(embed_model, {"input_tokens": embed, "output_tokens": 0})
    return worst


def match_worst_for(
    settings: Settings, models: match_models.MatchModels | None = None
) -> dict[str, Decimal]:
    return match_worst_steps(models or match_models.models_for(settings), settings.embed_model)


# --- the service ----------------------------------------------------------------------------


class MatchLimits(Protocol):
    def sieve(self, kind: str, session: Session) -> None: ...
    async def peek(self, kind: str, session: Session) -> None: ...
    async def refund(self, admission: Admission, session: Session) -> None: ...
    async def admit(
        self,
        kind: str,
        session: Session,
        *,
        sieved: bool = False,
        hold: Callable[[AsyncConnection], Awaitable[Any]] | None = None,
    ) -> Admission: ...


class RecordStore(Protocol):
    async def write(self, record: MatchRecord) -> None: ...


@dataclass
class _Run:
    """What one request holds; owned by its pipeline task, finalised once."""

    match_id: uuid.UUID
    source: Literal["greenhouse", "lever", "text"]
    session: Session
    admission: Admission | None = None
    scope: SpendScope | None = None
    language: str | None = None
    refund: bool = False
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class _Failed:
    code: str
    what: str


def default_deps(
    settings: Settings, db: Database, stage_timeouts: Mapping[Stage, float]
) -> MatchDeps:
    """Real models, retrieval and embedding (needs the dev or production keys)."""
    # Imported here: building the embedder needs a key, and tests never get this far.
    from app.rag.embed import AzureEmbedder

    return build_match_deps(settings, db, AzureEmbedder(settings), stage_timeouts)


class MatchService:
    def __init__(
        self,
        limits: MatchLimits,
        ledger: Ledger,
        make_deps: Callable[[], MatchDeps],
        *,
        records: RecordStore | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = MATCH_TIMEOUT_S,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.limits = limits
        self.ledger = ledger
        self._make_deps = make_deps
        self._records = records
        self._client = client
        self.timeout_s = timeout_s
        self._clock = clock
        self._graph: Any = None
        self._graph_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    @classmethod
    def for_app(
        cls, settings: Settings, db: Database, ledger: Ledger, limits: Limits
    ) -> "MatchService":
        # Tests may hand create_app a stand-in ledger; a real one learns the match path.
        if isinstance(ledger, Ledger):
            ledger.set_worst("match", match_worst_for(settings))
        timeouts = dict(DEFAULT_STAGE_TIMEOUTS)
        return cls(
            limits,
            ledger,
            lambda: default_deps(settings, db, timeouts),
            records=MatchRecords(db),
        )

    async def wait_idle(self) -> None:
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await self.wait_idle()

    async def _get_graph(self) -> Any:
        if self._graph is None:
            async with self._graph_lock:
                if self._graph is None:
                    self._graph = build_match_graph(self._make_deps())
        return self._graph

    # Before the stream: raises `Rejected` (an HTTP error) and never adds to a count.
    async def precheck(self, session: Session) -> None:
        self.limits.sieve("match", session)
        try:
            paused = await self.ledger.paused(self._clock(), "match")
        except DB_ERRORS:
            raise Rejected("unavailable") from None
        if paused is not None:
            raise Rejected("limit_day" if paused == "day" else "paused_month")
        await self.limits.peek("match", session)

    async def stream(self, text: str, ref: JobRef | None, session: Session) -> AsyncIterator[Event]:
        """(event, data) pairs for one match; always ends with `done` or `error`."""
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        deadline = loop.time() + self.timeout_s
        run = _Run(uuid.uuid4(), ref.source if ref is not None else "text", session)
        queue: asyncio.Queue[Event | _Failed | object] = asyncio.Queue()
        task = asyncio.create_task(self._run(run, text, ref, queue))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        try:
            while True:
                try:
                    async with asyncio.timeout_at(deadline):
                        item = await queue.get()
                except TimeoutError:
                    yield self._error_event("timeout", started, "Deadline")
                    return
                if isinstance(item, _Failed):
                    yield self._error_event(item.code, started, item.what)
                    return
                if item is _DONE:
                    yield "done", {}
                    return
                yield cast(Event, item)
        finally:
            if not task.done():
                task.cancel()

    async def _run(
        self,
        run: _Run,
        text: str,
        ref: JobRef | None,
        queue: asyncio.Queue[Event | _Failed | object],
    ) -> None:
        try:
            try:
                outcome = await self._pipeline(run, text, ref, queue)
            except Exception as error:  # noqa: BLE001 - every failure becomes an `error` event
                outcome = _Failed(_code(error), _describe(error))
            queue.put_nowait(outcome)
        finally:
            # The response cancels this task after a terminal event, a timeout or a
            # disconnect: shield the cleanup so settlement, refund and record always run.
            cleanup = asyncio.create_task(self._finalize(run))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue

    async def _pipeline(
        self,
        run: _Run,
        text: str,
        ref: JobRef | None,
        queue: asyncio.Queue[Event | _Failed | object],
    ) -> object:
        graph = await self._get_graph()

        if ref is not None:
            queue.put_nowait(("stage", {"stage": "fetching", "count": None}))
            try:
                await self.limits.admit("fetch", run.session, sieved=True)
            except Rejected as rejected:
                code = (
                    "fetch_limit" if rejected.code in ("limit_day", "slow_down") else "unavailable"
                )
                return _Failed(code, f"Fetch admission {rejected.code}")
            try:
                async with self._http() as client:
                    job = await fetch_job(ref, client)
            except FetchError as error:
                return _Failed("fetch_failed", f"FetchError {error.code}")
            jd, truncated = job.text, job.truncated
        else:
            if len(text) > MAX_JD_CHARS:
                return _Failed("too_long", "TooLong")
            jd, truncated = text, False

        holder: list[SpendScope] = []

        async def reserve(conn: AsyncConnection) -> None:
            holder.append(await self.ledger.reserve("match", run.match_id, conn=conn))

        try:
            run.admission = await self.limits.admit("match", run.session, sieved=True, hold=reserve)
        except Rejected as rejected:
            code = (
                rejected.code if rejected.code in ("limit_day", "paused_month") else "unavailable"
            )
            return _Failed(code, f"Admission {rejected.code}")
        except AIPaused as paused:
            return _Failed("limit_day" if paused.kind == "day" else "paused_month", "AIPaused")
        run.scope = holder[0]
        current_scope.set(run.scope)

        final: dict[str, Any] = {}
        not_a_job = False
        async for mode, data in graph.astream({"jd": jd}, stream_mode=["custom", "values"]):
            if mode == "values":
                final = data
                run.language = data.get("language", run.language)
                continue
            if "stage" in data:
                queue.put_nowait(("stage", {"stage": data["stage"], "count": data.get("count")}))
            elif "notes" in data:
                notes = data["notes"]
                queue.put_nowait(
                    (
                        "notes",
                        {
                            "requirements": notes["requirements"],
                            "findings": notes["findings"],
                            "truncated": truncated,
                        },
                    )
                )
            elif data.get("error") == "not_a_job":
                not_a_job = True

        guide: Guide | None = final.get("result")
        if not_a_job or guide is None:
            run.refund = True
            return _Failed("not_a_job", "NotAJob")
        requirements: list[Requirement] = final.get("requirements") or []
        queue.put_nowait(("result", _result(run.match_id, guide, requirements)))
        run.counts = {
            "requirements": len(requirements),
            "items": len(guide.items),
            "not_covered": len(guide.not_covered),
            "unchecked": guide.unchecked,
        }
        return _DONE

    def _http(self) -> Any:
        if self._client is not None:
            return _Borrowed(self._client)
        # Redirects are refused inside `fetch_job` too; the client never follows them either.
        return httpx.AsyncClient(follow_redirects=False)

    async def _finalize(self, run: _Run) -> None:
        if run.scope is not None:
            try:
                await run.scope.settle()
            except Exception as error:  # noqa: BLE001 - still attempt the refund and record
                logger.warning("match settle failed: %s", type(error).__name__)
        if run.refund and run.admission is not None:
            try:
                await self.limits.refund(run.admission, run.session)
            except Exception as error:  # noqa: BLE001 - still attempt the record
                logger.warning("match refund failed: %s", type(error).__name__)
        if run.admission is not None and self._records is not None and not run.session.patrol:
            scope = run.scope
            try:
                await self._records.write(
                    MatchRecord(
                        id=run.match_id,
                        source=run.source,
                        job_language=run.language,
                        prompt_version=prompts.PROMPT_VERSION,
                        providers=scope.providers() if scope else [],
                        cost_usd=scope.total_usd() if scope else Decimal(0),
                        ip_hash=run.admission.ip_hash,
                    )
                )
            except Exception as error:  # noqa: BLE001 - records must not escape the task
                logger.warning("match record failed: %s", type(error).__name__)
        if run.counts:
            logger.info(
                "match done: source=%s requirements=%d items=%d not_covered=%d unchecked=%d "
                "providers=%s",
                run.source,
                run.counts["requirements"],
                run.counts["items"],
                run.counts["not_covered"],
                run.counts["unchecked"],
                ",".join(run.scope.providers()) if run.scope else "",
            )

    @staticmethod
    def _error_event(code: str, started: float, what: str) -> Event:
        ms = round((time.perf_counter() - started) * 1000)
        logger.warning("match failed: code=%s error=%s ms=%d", code, what, ms)
        return "error", {"code": code, "message": MESSAGES[code]}


_DONE = object()


class _Borrowed:
    """An injected client as an async context manager that does not close it."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self._client

    async def __aexit__(self, *exc: object) -> None:
        return None


def _result(match_id: uuid.UUID, guide: Guide, requirements: list[Requirement]) -> dict[str, Any]:
    texts = {r.id: r.text for r in requirements}
    return {
        "match_id": str(match_id),
        "items": [
            {
                "requirement_id": item.requirement_id,
                "requirement": texts.get(item.requirement_id, ""),
                "text": item.text,
                "sections": [{"url": s.url, "title": s.title} for s in item.sections],
            }
            for item in guide.items
        ],
        "not_covered": list(guide.not_covered),
        "unchecked": guide.unchecked,
        "top8": guide.top8,
        "language": guide.language,
    }


def _code(error: BaseException) -> str:
    if isinstance(error, ProviderBudgetExhausted):
        return "paused_month"
    if isinstance(error, AIPaused):
        return "limit_day" if error.kind == "day" else "paused_month"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, UNAVAILABLE):
        return "unavailable"
    return "internal"


def _describe(error: BaseException) -> str:
    """Exception type and where it was raised: never the message (it can quote the job ad,
    a model's output or a database host)."""
    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return type(error).__name__
    last = frames[-1]
    return f"{type(error).__name__} at {last.filename.rsplit('/', 1)[-1]}:{last.lineno}"


def match_router(service: MatchService) -> APIRouter:
    router = APIRouter()

    async def precheck(
        session: Annotated[Session, Depends(require_session)],
        body: MatchRequest,
    ) -> Session:
        await service.precheck(session)
        return session

    @router.post("/api/match", response_class=EventSourceResponse)
    async def match(
        body: MatchRequest,
        session: Annotated[Session, Depends(precheck)],
    ) -> AsyncIterator[ServerSentEvent]:
        ref = parse_job_url(body.input)
        async for name, data in service.stream(body.input, ref, session):
            yield ServerSentEvent(event=name, data=data)

    return router
