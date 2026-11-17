"""`POST /api/ask`: one question in, contract item 7's events out as SSE.

    {"question": "1-500 characters", "conversation_id": "optional"}  ->  text/event-stream

`meta` first, then the graph's events (`output.events`: `route`, `status`, `token`, `reset`,
`answer`, `check`, `trace`), then `done`; or `error` at any point, after which nothing
else is sent. FastAPI's SSE support adds a `: ping` comment every 15 s of silence.

- The graph and the checkpointer are made on the first question (the checkpointer opens its
  own pool and runs `setup()` then), so startup and `/api/ping` never touch the database.
- One question at a time per conversation (in this process): a second one while the first
  runs gets `error busy`.
- The whole request has one deadline (`ASK_TIMEOUT_S`); past it the graph is cancelled and
  `error timeout` is sent. The graph runs in its own task and the response only waits on a
  queue, so the deadline never has to cancel code that is suspended at a `yield`.
- Only a turn that reaches `done` is added to the conversation's memory (`turns`); an error
  or a timeout leaves the memory as it was.
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
from typing import Annotated, Any, Protocol

import psycopg
from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent
from openai import APIError as OpenAIError
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.config import Settings
from app.db import Database
from app.qa import memory
from app.qa.graph import GraphDeps, build_deps, build_graph
from app.qa.memory import Checkpoints, Conversations, StoredTurn
from app.qa.output import events
from app.rag.types import ModelsUnavailable

logger = logging.getLogger(__name__)

ASK_TIMEOUT_S = 120.0
MAX_QUESTION_CHARS = 500

MESSAGES = {
    "busy": "Please wait for the current answer to finish.",
    "unavailable": ("The assistant is unavailable right now — please try again later or email me."),
    "timeout": "That took too long — please try again.",
    "internal": "Something went wrong — please try again.",
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
        timeout_s: float = ASK_TIMEOUT_S,
    ) -> None:
        self._conversations = conversations
        self._checkpoints = checkpoints
        self._make_deps = make_deps
        self.timeout_s = timeout_s
        self._graph: Any = None
        self._graph_lock = asyncio.Lock()
        # Conversations with a question running -> that question's owner token; a release
        # only counts if the token matches, so a late release can't free a newer question.
        self._active: dict[str, object] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    @classmethod
    def for_app(cls, settings: Settings, db: Database, checkpoint_pool: Any) -> "AskService":
        return cls(
            Conversations(db),
            Checkpoints(checkpoint_pool),
            functools.partial(default_deps, settings, db),
        )

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
        self, question: str, conversation_id: str | None
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """(event, data) pairs for one question; always ends with `done` or `error`."""
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        deadline = loop.time() + self.timeout_s
        turn_id = uuid.uuid4().hex
        task: asyncio.Task[None] | None = None
        conversation: str | None = None
        owner = object()
        try:
            try:
                async with asyncio.timeout_at(deadline):
                    graph = await self._get_graph()
                    conversation = await self._conversations.resolve(conversation_id)
            except Exception as error:  # noqa: BLE001 - every failure becomes an `error` event
                yield self._error(error, started)
                return
            if conversation in self._active:
                conversation = None  # not ours to release
                yield self._error_event("busy", started, "Busy")
                return
            self._active[conversation] = owner

            yield "meta", {"conversation_id": conversation, "turn_id": turn_id}

            queue: asyncio.Queue[Any] = asyncio.Queue()
            config = {"configurable": {"thread_id": conversation}}
            task = asyncio.create_task(self._run(graph, config, question, turn_id, queue))
            self._tasks.add(task)
            # The conversation is free again only when the graph has really stopped (a
            # cancelled graph may still be unwinding after this response has ended).
            task.add_done_callback(functools.partial(self._finished, conversation, owner))
            held, conversation = conversation, None

            while True:
                try:
                    async with asyncio.timeout_at(deadline):
                        item = await queue.get()
                except TimeoutError:
                    yield self._error_event("timeout", started, "Deadline")
                    return
                if item is _DONE or isinstance(item, _Failed):
                    # The graph's work is over (`_run` puts nothing after these): free the
                    # conversation now, so a next question right after `done` is not busy.
                    self._release(held, owner)
                    yield ("done", {}) if item is _DONE else self._error(item.error, started)
                    return
                yield item
        finally:
            if task is not None and not task.done():
                task.cancel()
            if conversation is not None:
                self._release(conversation, owner)

    def _release(self, conversation: str, owner: object) -> None:
        if self._active.get(conversation) is owner:
            del self._active[conversation]

    def _finished(self, conversation: str, owner: object, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        self._release(conversation, owner)

    async def _run(
        self,
        graph: Any,
        config: dict[str, Any],
        question: str,
        turn_id: str,
        queue: asyncio.Queue[Any],
    ) -> None:
        """Run the graph, then add the finished turn to the memory. Never raises."""
        started = time.perf_counter()
        try:
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
                "turn_id": turn_id,
                "question": question,
                "answer": state.get("answer_text") or "",
                "embedding": state.get("question_embedding") or b"",
            }
            turns = memory.add_turn(state.get("turns") or [], turn)
            await graph.aupdate_state(config, {"turns": turns}, as_node="finish")
            _log_usage(state, started)
            queue.put_nowait(_DONE)
        except Exception as error:  # noqa: BLE001 - every failure becomes an `error` event
            queue.put_nowait(_Failed(error))

    def _error(self, error: BaseException, started: float) -> tuple[str, dict[str, Any]]:
        if isinstance(error, TimeoutError):
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

    @router.post("/api/ask", response_class=EventSourceResponse)
    async def ask(body: AskRequest) -> AsyncIterator[ServerSentEvent]:
        async for name, data in service.stream(body.question, body.conversation_id):
            yield ServerSentEvent(event=name, data=data)

    return router
