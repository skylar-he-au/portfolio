"""Best-effort question and job-match records and retention; never log record contents."""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal
from uuid import UUID

from app import retention
from app.db import Database
from app.qa.memory import Checkpoints

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionRecord:
    id: UUID
    question: str
    answer_language: str | None
    prompt_version: str
    providers: list[str]
    cost_usd: Decimal
    ip_hash: bytes


class Records:
    def __init__(self, db: Database, checkpoints: Checkpoints) -> None:
        self._db = db
        self._checkpoints = checkpoints

    async def write(self, record: QuestionRecord) -> None:
        try:
            async with self._db.connection() as conn:
                await conn.execute(
                    "INSERT INTO question_log "
                    "(id, question, answer_language, prompt_version, providers, cost_usd, ip_hash) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (
                        record.id,
                        record.question,
                        record.answer_language,
                        record.prompt_version,
                        record.providers,
                        record.cost_usd,
                        record.ip_hash,
                    ),
                )
                await conn.commit()
        except Exception as error:  # noqa: BLE001 - records cannot fail the response
            logger.warning("question record failed: %s", type(error).__name__)

        # Each sweep gets its own boundary: a broken saver must not stop key cleanup.
        for sweep in (
            lambda: retention.delete_old_questions(self._db, limit=20),
            self._delete_conversations,
            lambda: retention.delete_old_daily_keys(self._db),
            lambda: retention.delete_old_counters(self._db),
        ):
            try:
                await sweep()
            except Exception as error:  # noqa: BLE001 - retention is best effort
                logger.warning("question retention failed: %s", type(error).__name__)

    async def _delete_conversations(self) -> int:
        saver = await self._checkpoints.saver()
        return await retention.delete_expired_conversations(self._db, saver, limit=20)


@dataclass(frozen=True)
class MatchRecord:
    """One `match_log` row (SPEC "职位对照"): no job ad, requirements or result."""

    id: UUID
    source: Literal["greenhouse", "lever", "text"]
    job_language: str | None
    prompt_version: str
    providers: list[str]
    cost_usd: Decimal
    ip_hash: bytes


class MatchRecords:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def write(self, record: MatchRecord) -> None:
        try:
            async with self._db.connection() as conn:
                await conn.execute(
                    "INSERT INTO match_log "
                    "(id, source, job_language, prompt_version, providers, cost_usd, ip_hash) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (
                        record.id,
                        record.source,
                        record.job_language,
                        record.prompt_version,
                        record.providers,
                        record.cost_usd,
                        record.ip_hash,
                    ),
                )
                await conn.commit()
        except Exception as error:  # noqa: BLE001 - records cannot fail the response
            logger.warning("match record failed: %s", type(error).__name__)

        for sweep in (
            lambda: retention.delete_old_matches(self._db, limit=20),
            lambda: retention.delete_old_daily_keys(self._db),
            lambda: retention.delete_old_counters(self._db),
        ):
            try:
                await sweep()
            except Exception as error:  # noqa: BLE001 - retention is best effort
                logger.warning("match retention failed: %s", type(error).__name__)
