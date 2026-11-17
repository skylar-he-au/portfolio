"""Conversation state and memory for "Ask about my projects".

- **Conversations**: the id is random and made by the server (`conversations` table, which
  only records `created_at`; step 5c deletes conversations and their checkpoints after 24
  hours). An unknown, malformed or older-than-24-hours id gets a new conversation.
- **Checkpoints**: the graph state of each conversation (`thread_id` = conversation id) is
  kept by LangGraph's `AsyncPostgresSaver` on its own pool (`autocommit`, `dict_row`, as
  the saver needs). The pool is created closed; it opens, and `setup()` runs, on the first
  question, so starting the app and `/api/ping` never touch the database.
- **Memory**: every finished turn (one that reached `done`) is stored in the state as
  `{turn_id, question, answer, embedding}`. At the start of the next turn the `recall` node
  embeds the new question and picks at most `MAX_RECALLED` earlier turns with `rank_turns`.

Ranking formula (code, not a model):

    score = cosine(question, turn) + RECENCY_BONUS * RECENCY_DECAY ** age

`age` is 0 for the latest finished turn, 1 for the one before, and so on (turns, not
minutes: a conversation lasts at most a day, and a follow-up like "how is it tested?" refers
to the previous turn however long the visitor paused). With a bonus of 0.5 and a decay of
0.7 the previous turn gets +0.5, two turns back +0.35, five back +0.12, ten back +0.01: a
recent turn wins unless an older one is clearly closer in meaning. Ties go to the newer
turn. The picked turns are handed to the graph oldest first, so they read as a conversation.
"""

import asyncio
import logging
import math
import uuid
from array import array
from collections.abc import Sequence
from typing import Any, TypedDict

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.db import CONNECT_TIMEOUT_S, POOL_WAIT_S, Database
from app.rag.types import RecalledTurn

logger = logging.getLogger(__name__)

MAX_RECALLED = 6
# Older turns are dropped from the state: bounds the checkpoint size (each turn holds a
# 1536-float embedding, ~6 KB packed) whatever the visitor does.
MAX_STORED_TURNS = 20
RECENCY_BONUS = 0.5
RECENCY_DECAY = 0.7

# Deserialize only these custom types from checkpoints (plus LangGraph's own safe list);
# anything else in a checkpoint is refused instead of imported.
CHECKPOINT_TYPES = [("app.rag.types", "Section"), ("app.rag.types", "RecalledTurn")]


def checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


# --- memory ------------------------------------------------------------------------------


class StoredTurn(TypedDict):
    turn_id: str
    question: str  # the visitor's own words
    answer: str  # the final answer text
    embedding: bytes  # the question's embedding, float32 (`pack_embedding`)


def pack_embedding(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack_embedding(data: bytes) -> list[float]:
    values = array("f")
    values.frombytes(data)
    return values.tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def rank_turns(turns: Sequence[StoredTurn], question: Sequence[float]) -> list[RecalledTurn]:
    """Every stored turn with its score (module docstring), best first."""
    newest_first = list(reversed(turns))
    scored = [
        RecalledTurn(
            turn_id=turn["turn_id"],
            question=turn["question"],
            answer=turn["answer"],
            score=cosine(question, unpack_embedding(turn["embedding"]))
            + RECENCY_BONUS * RECENCY_DECAY**age,
        )
        for age, turn in enumerate(newest_first)
    ]
    # Stable sort over newest-first input: equal scores (to 6 places; embeddings are
    # float32) keep the newer turn first.
    return sorted(scored, key=lambda t: -round(t.score, 6))


def recall(
    turns: Sequence[StoredTurn], question: Sequence[float], limit: int = MAX_RECALLED
) -> list[RecalledTurn]:
    """The `limit` best turns, oldest first."""
    best = rank_turns(turns, question)[:limit]
    order = {turn["turn_id"]: index for index, turn in enumerate(turns)}
    return sorted(best, key=lambda t: order[t.turn_id])


def add_turn(turns: Sequence[StoredTurn], turn: StoredTurn) -> list[StoredTurn]:
    """`turns` plus `turn`, keeping the newest `MAX_STORED_TURNS`."""
    return [*turns, turn][-MAX_STORED_TURNS:]


# --- conversations -----------------------------------------------------------------------


class Conversations:
    """The `conversations` table, through the app's `Database`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def resolve(self, requested: str | None) -> str:
        """The conversation to use: `requested` if it exists and is under 24 hours old,
        otherwise a new one."""
        known = _parse_id(requested)
        async with self._db.connection() as conn:
            if known is not None:
                cursor = await conn.execute(
                    "SELECT 1 FROM conversations"
                    " WHERE id = %s AND created_at > now() - interval '24 hours'",
                    (known,),
                )
                if await cursor.fetchone():
                    return str(known)
            new = uuid.uuid4()
            await conn.execute("INSERT INTO conversations (id) VALUES (%s)", (new,))
            await conn.commit()
        return str(new)


def _parse_id(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


# --- checkpoints -------------------------------------------------------------------------


def make_checkpoint_pool(database_url: str) -> AsyncConnectionPool:
    """A closed pool for the checkpointer (opened on the first question)."""
    return AsyncConnectionPool(
        conninfo=database_url,
        open=False,
        min_size=0,
        max_size=5,
        max_idle=60,
        timeout=POOL_WAIT_S,
        reconnect_timeout=30,
        check=AsyncConnectionPool.check_connection,
        # What AsyncPostgresSaver.from_conn_string sets on its own connection.
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
            "connect_timeout": CONNECT_TIMEOUT_S,
        },
    )


class Checkpoints:
    """`AsyncPostgresSaver` on its own pool, opened and set up on first use."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._saver: AsyncPostgresSaver | None = None
        self._opened = False
        self._lock = asyncio.Lock()

    async def saver(self) -> AsyncPostgresSaver:
        if self._saver is not None:
            return self._saver
        async with self._lock:
            if self._saver is None:
                if not self._opened:
                    await self._pool.open(wait=False)
                    self._opened = True
                saver = AsyncPostgresSaver(self._pool, serde=checkpoint_serde())
                await saver.setup()  # creates or migrates LangGraph's tables; idempotent
                self._saver = saver
        return self._saver

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False
            self._saver = None
