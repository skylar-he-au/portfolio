"""Thumbs-down feedback, bounded before borrowing a database connection.

The body names either an answer (`turn_id`, marks `question_log`) or a job match
(`match_id`, marks `match_log`): exactly one of the two, otherwise 422."""

import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Request, Response
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, model_validator

from app.guard.limits import SYDNEY
from app.guard.session import Rejected, Session, require_session


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: uuid.UUID | None = None
    match_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> "FeedbackRequest":
        if (self.turn_id is None) == (self.match_id is None):
            raise ValueError("send exactly one of turn_id and match_id")
        return self


def feedback_router(clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> APIRouter:
    router = APIRouter()
    counts: dict[bytes, int] = {}
    marked_day: date | None = None

    @router.post("/api/feedback", status_code=204)
    async def feedback(
        request: Request,
        session: Annotated[Session, Depends(require_session)],
        body: FeedbackRequest,
    ) -> Response:
        nonlocal marked_day
        day = clock().astimezone(SYDNEY).date()
        if day != marked_day:
            counts.clear()
            marked_day = day
        if session.token_key not in counts and len(counts) >= 10_000:
            raise Rejected("slow_down")
        counts[session.token_key] = counts.get(session.token_key, 0) + 1
        if counts[session.token_key] > 20:
            raise Rejected("slow_down")
        if not request.app.state.limits.allow_db_query():
            raise Rejected("slow_down")
        try:
            async with request.app.state.db.connection() as conn:
                # Two fixed statements: the table name never comes from the request.
                if body.turn_id is not None:
                    await conn.execute(
                        "UPDATE question_log SET thumbs_down = true WHERE id = %s "
                        "AND created_at > now() - interval '30 days'",
                        (body.turn_id,),
                    )
                else:
                    await conn.execute(
                        "UPDATE match_log SET thumbs_down = true WHERE id = %s "
                        "AND created_at > now() - interval '30 days'",
                        (body.match_id,),
                    )
                await conn.commit()
        except (psycopg.Error, PoolTimeout, OSError, TimeoutError):
            raise Rejected("unavailable") from None
        return Response(status_code=204)

    return router
