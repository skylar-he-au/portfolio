"""`POST /api/ask`: one question in, contract item 7's events out as SSE.

    {"question": "1-500 characters", "conversation_id": "optional"}  ->  text/event-stream

`meta` first, then the graph's events (`output.events`: `route`, `status`, `token`, `reset`,
`answer`, `check`, `trace`), then `done`; or `error` at any point, after which nothing
else is sent. FastAPI's SSE support adds a `: ping` comment every 15 s of silence.

- The graph and the checkpointer are made on the first question (the checkpointer opens its
  own pool and runs `setup()` then), so startup and `/api/ping` never touch the database.
- One question at a time per conversation (database lock): a second one while the first
  runs gets `error busy`.
- The whole request has one deadline (`ASK_TIMEOUT_S`); past it the graph is cancelled and
  `error timeout` is sent. The graph runs in its own task and the response only waits on a
  queue, so the deadline never has to cancel code that is suspended at a `yield`.
- Only a turn that reaches `done` is added to the conversation's memory (`turns`); an error
  or a timeout leaves the memory as it was.
- The graph task owns its spend scope and question log, including after a timeout or
  disconnect. It releases the lock before sending the terminal event.
- Logs carry usage only (timings, providers, models, error types), never the question,
  the answer or the conversation id.
"""

import asyncio
import functools
import logging
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Protocol

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from openai import APIError as OpenAIError
from pydantic import BaseModel, ConfigDict, StringConstraints
from psycopg_pool import PoolTimeout

from app.config import Settings
from app.db import Database
from app.guard.ledger import AIPaused, Ledger, ProviderBudgetExhausted, SpendScope, current_scope
from app.guard.limits import Admission
from app.guard.session import REJECTIONS, Rejected, Session, require_session
from app.qa import memory
from app.qa.graph import GraphDeps, build_deps, build_graph
from app.qa.memory import Checkpoints, Conversations, StoredTurn
from app.qa.output import events
from app.qa.prompts import PROMPT_VERSION
from app.rag.types import ModelsUnavailable
from app.records import QuestionRecord, Records

logger = logging.getLogger(__name__)

ASK_TIMEOUT_S = 120.0
MAX_QUESTION_CHARS = 500

MESSAGES = {
    "busy": "Please wait for the current answer to finish.",
    "unavailable": ("The assistant is unavailable right now — please try again later or email me."),
    "timeout": "That took too long — please try again.",
    "internal": "Something went wrong — please try again.",
    "limit_day": REJECTIONS["limit_day"][1],
    "paused_month": REJECTIONS["paused_month"][1],
}

# Failures of a service we depend on (both model providers, the embedding API, the
# database): the visitor can only wait and retry, so they are `unavailable`.
UNAVAILABLE = (ModelsUnavailable, psycopg.Error, OpenAIError)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]
    conversation_id: Annotated[str, StringConstraints(max_length=100)] | None = None


class ConversationStore(Protocol):
    async def resolve(self, requested: str | None) -> str: ...
    async def acquire(self, conversation_id: str, owner: uuid.UUID) -> bool: ...
    async def release(self, conversation_id: str, owner: uuid.UUID) -> None: ...


class RecordStore(Protocol):
    async def write(self, record: QuestionRecord) -> None: ...


class CheckpointStore(Protocol):
    async def saver(self) -> Any: ...
    async def close(self) -> None: ...


@dataclass
class _Failed:
    error: BaseException


_DONE = object()


def default_deps(settings: Settings, db: Database) -> GraphDeps:
    """Real models, retrieval and embedding (needs the dev or production keys)."""
    # Imported here: building the embedder needs a key, and tests never get this far.
    from app.rag.embed import OpenAIEmbedder
    from app.rag.retrieve import retrieve

    embedder = OpenAIEmbedder(settings)
    return build_deps(settings, functools.partial(retrieve, db, embedder), embedder)


class AskService:
    def __init__(
        self,
        conversations: ConversationStore,
        checkpoints: CheckpointStore,
        make_deps: Callable[[], GraphDeps],
        *,
        records: RecordStore | None = None,
        ledger: Ledger | None = None,
        timeout_s: float = ASK_TIMEOUT_S,
    ) -> None:
        self._conversations = conversations
        self._checkpoints = checkpoints
        self._make_deps = make_deps
        self._records = records
        self._ledger = ledger
        self.timeout_s = timeout_s
        self._graph: Any = None
        self._graph_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    @classmethod
    def for_app(
        cls, settings: Settings, db: Database, checkpoint_pool: Any, ledger: Ledger
    ) -> "AskService":
        checkpoints = Checkpoints(checkpoint_pool)
        return cls(
            Conversations(db),
            checkpoints,
            functools.partial(default_deps, settings, db),
            records=Records(db, checkpoints),
            ledger=ledger,
        )

    async def reserve(self, turn_id: uuid.UUID) -> SpendScope | None:
        if self._ledger is None:
            return None
        try:
            return await self._ledger.reserve_question(turn_id)
        except AIPaused as error:
            raise Rejected("limit_day" if error.kind == "day" else "paused_month") from None
        except (psycopg.Error, PoolTimeout, OSError, TimeoutError):
            raise Rejected("unavailable") from None

    async def wait_idle(self) -> None:
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await self._checkpoints.close()

    async def _get_graph(self) -> Any:
        if self._graph is None:
            async with self._graph_lock:
                if self._graph is None:
                    saver = await self._checkpoints.saver()
                    self._graph = build_graph(self._make_deps(), checkpointer=saver)
        return self._graph

    async def stream(
        self,
        question: str,
        conversation_id: str | None,
        admission: Admission | None = None,
        *,
        scope: SpendScope | None = None,
        turn_id: uuid.UUID | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """(event, data) pairs for one question; always ends with `done` or `error`."""
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        deadline = loop.time() + self.timeout_s
        turn_id = turn_id if turn_id is not None else uuid.uuid4()
        task: asyncio.Task[None] | None = None
        held: str | None = None
        owner = uuid.uuid4()
        try:
            try:
                async with asyncio.timeout_at(deadline):
                    graph = await self._get_graph()
                    conversation = await self._conversations.resolve(conversation_id)
                    if await self._conversations.acquire(conversation, owner):
                        held = conversation
            except Exception as error:  # noqa: BLE001 - every failure becomes an `error` event
                yield self._error(error, started)
                return
            if held is None:
                yield self._error_event("busy", started, "Busy")
                return
            yield "meta", {"conversation_id": conversation, "turn_id": turn_id.hex}

            queue: asyncio.Queue[Any] = asyncio.Queue()
            config = {"configurable": {"thread_id": conversation}}
            task = asyncio.create_task(
                self._run(
                    graph, config, question, turn_id, queue, conversation, owner, scope, admission
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            held, scope = None, None  # the graph task now owns both resources

            while True:
                try:
                    async with asyncio.timeout_at(deadline):
                        item = await queue.get()
                except TimeoutError:
                    yield self._error_event("timeout", started, "Deadline")
                    return
                if item is _DONE or isinstance(item, _Failed):
                    yield ("done", {}) if item is _DONE else self._error(item.error, started)
                    return
                yield item
        finally:
            if task is not None and not task.done():
                task.cancel()
            if held is not None:
                try:
                    await self._conversations.release(held, owner)
                except Exception as error:  # noqa: BLE001 - cleanup must not fail the response
                    logger.warning("ask release failed: %s", type(error).__name__)
            if scope is not None:
                try:
                    await scope.settle()
                except Exception as error:  # noqa: BLE001 - cleanup must not fail the response
                    logger.warning("ask settle failed: %s", type(error).__name__)

    async def _run(
        self,
        graph: Any,
        config: dict[str, Any],
        question: str,
        turn_id: uuid.UUID,
        queue: asyncio.Queue[Any],
        conversation: str,
        owner: uuid.UUID,
        scope: SpendScope | None,
        admission: Admission | None,
    ) -> None:
        if scope is not None:
            current_scope.set(scope)
        started = time.perf_counter()
        released = False
        try:
            outcome: object = _DONE
            try:
                await self._answer(graph, config, question, turn_id, queue, started)
            except Exception as error:  # noqa: BLE001 - every graph failure becomes an event
                outcome = _Failed(error)
            try:
                await self._conversations.release(conversation, owner)
                released = True
            except Exception as error:  # noqa: BLE001 - retry release during finalisation
                logger.warning("ask release failed: %s", type(error).__name__)
            queue.put_nowait(outcome)
        finally:
            # The response cancels this task even after receiving `done`. Shield cleanup
            # and wait for it so wait_idle includes settlement and the record write.
            cleanup = asyncio.create_task(
                self._finalize(
                    graph,
                    config,
                    question,
                    turn_id,
                    conversation,
                    owner,
                    scope,
                    admission,
                    released,
                )
            )
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue

    async def _answer(
        self,
        graph: Any,
        config: dict[str, Any],
        question: str,
        turn_id: uuid.UUID,
        queue: asyncio.Queue[Any],
        started: float,
    ) -> None:
        stream = graph.astream(
            {"question": question, "history": []},
            config,
            stream_mode=["messages", "custom"],
            version="v2",
        )
        async for event in events(stream):
            queue.put_nowait(event)
        state = (await graph.aget_state(config)).values
        turn: StoredTurn = {
            "turn_id": turn_id.hex,
            "question": question,
            "answer": state.get("answer_text") or "",
            "embedding": state.get("question_embedding") or b"",
        }
        turns = memory.add_turn(state.get("turns") or [], turn)
        await graph.aupdate_state(config, {"turns": turns}, as_node="finish")
        _log_usage(state, started)

    async def _finalize(
        self,
        graph: Any,
        config: dict[str, Any],
        question: str,
        turn_id: uuid.UUID,
        conversation: str,
        owner: uuid.UUID,
        scope: SpendScope | None,
        admission: Admission | None,
        released: bool,
    ) -> None:
        if not released:
            try:
                await self._conversations.release(conversation, owner)
            except Exception as error:  # noqa: BLE001 - each finalisation step is independent
                logger.warning("ask release failed: %s", type(error).__name__)
        if scope is not None:
            try:
                await scope.settle()
            except Exception as error:  # noqa: BLE001 - still attempt the question record
                logger.warning("ask settle failed: %s", type(error).__name__)
        if self._records is not None and admission is not None:
            language = None
            try:
                state = (await graph.aget_state(config)).values
                language = state.get("language")
                # begin sets a default language; only rewrite fills the query.
                if state.get("query") == "":
                    language = None
            except Exception as error:  # noqa: BLE001 - classification may never have run
                logger.warning("ask language failed: %s", type(error).__name__)
            try:
                await self._records.write(
                    QuestionRecord(
                        id=turn_id,
                        question=question,
                        answer_language=language,
                        prompt_version=PROMPT_VERSION,
                        providers=scope.providers() if scope else [],
                        cost_usd=scope.total_usd() if scope else Decimal(0),
                        ip_hash=admission.ip_hash,
                    )
                )
            except Exception as error:  # noqa: BLE001 - records must not escape the task
                logger.warning("ask record failed: %s", type(error).__name__)

    def _error(self, error: BaseException, started: float) -> tuple[str, dict[str, Any]]:
        if isinstance(error, ProviderBudgetExhausted):
            code = "paused_month"
        elif isinstance(error, AIPaused):
            code = "limit_day" if error.kind == "day" else "paused_month"
        elif isinstance(error, TimeoutError):
            code = "timeout"
        elif isinstance(error, UNAVAILABLE):
            code = "unavailable"
        else:
            code = "internal"
        return self._error_event(code, started, _describe(error))

    @staticmethod
    def _error_event(code: str, started: float, what: str) -> tuple[str, dict[str, Any]]:
        ms = round((time.perf_counter() - started) * 1000)
        logger.warning("ask failed: code=%s error=%s ms=%d", code, what, ms)
        return "error", {"code": code, "message": MESSAGES[code]}


def _describe(error: BaseException) -> str:
    """Exception type and where it was raised: never the message (it can quote the
    question, a model's output or a database host)."""
    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return type(error).__name__
    last = frames[-1]
    return f"{type(error).__name__} at {last.filename.rsplit('/', 1)[-1]}:{last.lineno}"


def _log_usage(state: dict[str, Any], started: float) -> None:
    nodes = ",".join(
        f"{n['name']}:{n.get('provider') or '-'}/{n.get('model') or '-'}:{n['ms']}ms"
        for n in (state.get("trace") or {}).get("nodes", [])
    )
    check = state.get("check") or {}
    logger.info(
        "ask done: ms=%d private=%s off_topic=%s sections=%d memory=%d passed=%s nodes=%s",
        round((time.perf_counter() - started) * 1000),
        state.get("private"),
        state.get("off_topic"),
        len(state.get("sections") or []),
        len(state.get("history") or []),
        check.get("passed"),
        nodes,
    )


def ask_router(service: AskService) -> APIRouter:
    router = APIRouter()

    async def admit(
        request: Request,
        session: Annotated[Session, Depends(require_session)],
        body: AskRequest,
    ) -> Admission:
        return await request.app.state.limits.admit("ask", session)

    async def reserve(
        admission: Annotated[Admission, Depends(admit)],
    ) -> tuple[uuid.UUID, SpendScope | None]:
        turn_id = uuid.uuid4()
        return turn_id, await service.reserve(turn_id)

    @router.post("/api/ask", response_class=EventSourceResponse)
    async def ask(
        body: AskRequest,
        admission: Annotated[Admission, Depends(admit)],
        reservation: Annotated[tuple[uuid.UUID, SpendScope | None], Depends(reserve)],
    ) -> AsyncIterator[ServerSentEvent]:
        turn_id, scope = reservation
        async for name, data in service.stream(
            body.question, body.conversation_id, admission, scope=scope, turn_id=turn_id
        ):
            yield ServerSentEvent(event=name, data=data)

    return router
