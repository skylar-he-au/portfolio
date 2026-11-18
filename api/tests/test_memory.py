"""Memory ranking, stored turns and the checkpoint serializer (no network)."""

import logging
import math
import uuid
from dataclasses import dataclass

import pytest
from langgraph.checkpoint.serde import jsonplus
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.db import Database, make_pool
from app.migrate import migrate
from app.qa import memory
from app.qa.memory import (
    MAX_STORED_TURNS,
    Conversations,
    StoredTurn,
    add_turn,
    checkpoint_serde,
    pack_embedding,
    rank_turns,
    recall,
    unpack_embedding,
)
from app.rag.types import RecalledTurn, Section

DIMS = 1536


def vector_at(cos: float) -> list[float]:
    """A unit vector whose cosine with `QUESTION` is `cos`."""
    v = [0.0] * DIMS
    v[0] = cos
    v[1] = math.sqrt(1 - cos * cos)
    return v


QUESTION = vector_at(1.0)


def turn(index: int, cos: float) -> StoredTurn:
    return {
        "turn_id": f"t{index}",
        "question": f"question {index}",
        "answer": f"answer {index}",
        "embedding": pack_embedding(vector_at(cos)),
    }


# Nine finished turns, oldest first. t0 is very old but almost the same question;
# t8 is the latest turn but unrelated.
COSINES = [0.95, 0.2, 0.6, 0.1, 0.3, 0.5, 0.05, 0.4, 0.0]
HISTORY = [turn(i, c) for i, c in enumerate(COSINES)]


def expected_score(index: int) -> float:
    # The formula in memory.py's docstring, written out again here.
    age = len(COSINES) - 1 - index
    return COSINES[index] + 0.5 * 0.7**age


def test_ranking_follows_the_formula():
    ranked = rank_turns(HISTORY, QUESTION)
    order = sorted(range(len(COSINES)), key=lambda i: -expected_score(i))
    assert [t.turn_id for t in ranked] == [f"t{i}" for i in order]
    # Written out: old-but-similar first, the unrelated latest turn still fifth.
    assert [t.turn_id for t in ranked] == ["t0", "t7", "t5", "t2", "t8", "t4", "t6", "t1", "t3"]
    for t in ranked:
        assert t.score == pytest.approx(expected_score(int(t.turn_id[1:])), abs=1e-6)
    assert ranked[0].question == "question 0" and ranked[0].answer == "answer 0"


def test_recall_keeps_at_most_six_oldest_first():
    picked = recall(HISTORY, QUESTION)
    assert [t.turn_id for t in picked] == ["t0", "t2", "t4", "t5", "t7", "t8"]
    assert all(isinstance(t, RecalledTurn) for t in picked)


def test_recall_with_few_turns_returns_all_and_none_for_none():
    assert [t.turn_id for t in recall(HISTORY[:2], QUESTION)] == ["t0", "t1"]
    assert recall([], QUESTION) == []


def test_ties_go_to_the_newer_turn():
    # t0: 0.15 + 0.5 * 0.7 = 0.5; t1 (latest): 0.0 + 0.5 = 0.5.
    tied = rank_turns([turn(0, 0.15), turn(1, 0.0)], QUESTION)
    assert [t.turn_id for t in tied] == ["t1", "t0"]
    assert tied[0].score == pytest.approx(tied[1].score, abs=1e-6)


def test_add_turn_keeps_the_newest():
    turns: list[StoredTurn] = []
    for i in range(MAX_STORED_TURNS + 3):
        turns = add_turn(turns, turn(i, 0.1))
    assert len(turns) == MAX_STORED_TURNS
    assert turns[0]["turn_id"] == "t3" and turns[-1]["turn_id"] == f"t{MAX_STORED_TURNS + 2}"


def test_embedding_round_trip_is_float32():
    v = vector_at(0.3)
    packed = pack_embedding(v)
    assert len(packed) == DIMS * 4
    assert unpack_embedding(packed) == pytest.approx(v, abs=1e-7)


# --- checkpoint serializer -----------------------------------------------------------------

STATE = {
    "sections": [Section("/about", "work-rights", "Work rights", "text", 0.5)],
    "history": [RecalledTurn("t1", "q", "a", 0.9)],
    "turns": [turn(0, 0.2)],
}


def _round_trip(serde: JsonPlusSerializer) -> object:
    return serde.loads_typed(serde.dumps_typed(STATE))


def test_default_serializer_warns_about_our_types(monkeypatch, caplog):
    # Control: shows the warning is visible to this test when it happens.
    monkeypatch.setattr(jsonplus, "_warned_unregistered_types", set())
    with caplog.at_level(logging.WARNING):
        assert _round_trip(JsonPlusSerializer()) == STATE
    assert "unregistered type app.rag.types.Section" in caplog.text


def test_checkpoint_serializer_allows_our_types_without_warning(monkeypatch, caplog):
    monkeypatch.setattr(jsonplus, "_warned_unregistered_types", set())
    with caplog.at_level(logging.WARNING):
        assert _round_trip(checkpoint_serde()) == STATE
    assert caplog.text == ""


@dataclass
class NotOnTheList:
    value: int


def test_checkpoint_serializer_refuses_other_types():
    serde = checkpoint_serde()
    restored = serde.loads_typed(serde.dumps_typed({"x": NotOnTheList(1)}))
    assert not isinstance(restored["x"], NotOnTheList)
    # The default (permissive) serializer would have rebuilt it.
    assert isinstance(
        JsonPlusSerializer().loads_typed(serde.dumps_typed({"x": NotOnTheList(1)}))["x"],
        NotOnTheList,
    )
    assert memory.CHECKPOINT_TYPES == [
        ("app.rag.types", "Section"),
        ("app.rag.types", "RecalledTurn"),
    ]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_conversation_lock_expiry_and_owner_release(db_url):
    migrate(db_url)
    db = Database(make_pool(db_url))
    conversations = Conversations(db)
    first, second = uuid.uuid4(), uuid.uuid4()
    try:
        cid = await conversations.resolve(None)
        assert await conversations.acquire(cid, first)
        assert not await conversations.acquire(cid, second)
        async with db.connection() as conn:
            cursor = await conn.execute(
                "SELECT running_owner, running_until > now(), "
                "running_until <= now() + interval '150 seconds' FROM conversations WHERE id = %s",
                (cid,),
            )
            assert await cursor.fetchone() == (first, True, True)
            await conn.execute(
                "UPDATE conversations SET running_until = now() - interval '1 second' "
                "WHERE id = %s",
                (cid,),
            )
            await conn.commit()
        assert await conversations.acquire(cid, second)
        await conversations.release(cid, first)
        async with db.connection() as conn:
            cursor = await conn.execute(
                "SELECT running_owner, running_until IS NOT NULL FROM conversations WHERE id = %s",
                (cid,),
            )
            assert await cursor.fetchone() == (second, True)
        await conversations.release(cid, second)
        await conversations.release(cid, second)
        async with db.connection() as conn:
            cursor = await conn.execute(
                "SELECT running_owner, running_until FROM conversations WHERE id = %s", (cid,)
            )
            assert await cursor.fetchone() == (None, None)
        assert await conversations.acquire(cid, first)
        assert not await conversations.acquire(str(uuid.uuid4()), first)
    finally:
        await db.close()
