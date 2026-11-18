"""Step 5c1: the database conversation lock, spend scope and 30-day question log of `/api/ask`."""

import pytest
from test_ask import (
    EXAMPLE_Q,
    MATCH,
    Models,
    collect,
    general_turn,
    make_deps,
    one,
)
from test_graph import FakeRetriever, Slow, rewrite_reply

from app.api.ask import AskService
from app.db import Database, make_pool
from app.migrate import migrate
from app.qa.memory import Checkpoints, Conversations, make_checkpoint_pool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def postgres_service(db: Database, url: str, models: Models, retriever: FakeRetriever):
    deps = make_deps(models, retriever)
    return AskService(Conversations(db), Checkpoints(make_checkpoint_pool(url)), lambda: deps)


async def test_two_services_on_one_database_share_the_conversation_lock(db_url: str):
    """The lock lives in the database, so a second instance sees the first one's question."""
    migrate(db_url)
    c, g = general_turn()
    db = Database(make_pool(db_url))
    first_service = postgres_service(
        db, db_url, Models([Slow(0.5, rewrite_reply()), c[1]], g), FakeRetriever([MATCH])
    )
    second_service = postgres_service(db, db_url, Models([], []), FakeRetriever())
    try:
        first = first_service.stream(EXAMPLE_Q, None)
        name, meta = await anext(first)
        assert name == "meta"
        busy = await collect(second_service, "Another question", meta["conversation_id"])
        assert busy == [
            ("error", {"code": "busy", "message": "Please wait for the current answer to finish."})
        ]
        rest = [event async for event in first]
        assert rest[-1] == ("done", {})
        assert one(rest, "check") is not None
    finally:
        await first_service.close()
        await second_service.close()
        await db.close()
