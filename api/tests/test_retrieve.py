import asyncio
import math
from pathlib import Path

import psycopg
import pytest
from fakes import DIMENSIONS, FakeEmbedder, unit_vector
from pgvector import Vector
from pgvector.psycopg import register_vector, register_vector_async

from app.db import Database, make_pool
from app.ingest import ingest
from app.migrate import migrate
from app.rag.retrieve import retrieve
from app.rag.types import Section

QUERY = "quasar OR nebula"


def seed_hybrid(db_url: str, *, all_keywords: bool = False) -> None:
    """32 distinct cosine ranks; the unique quasar hit is outside the top 20."""
    migrate(db_url)
    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        version = conn.execute(
            "INSERT INTO section_versions DEFAULT VALUES RETURNING version"
        ).fetchone()[0]
        conn.execute("INSERT INTO content_state (current_version) VALUES (%s)", (version,))
        for index in range(32):
            theta = (index + 1) * 0.02
            vector = [0.0] * DIMENSIONS
            vector[0], vector[1] = math.cos(theta), math.sin(theta)
            text = "ordinary content"
            if all_keywords:
                # ts_rank_cd sums term occurrences, giving distinct keyword ranks.
                text = "nebula " * (32 - index)
            elif index == 25:
                text = "quasar quasar quasar"
            elif index == 4:
                text = "nebula"
            conn.execute(
                "INSERT INTO sections "
                "(version, page, section_id, heading, text, position, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    version,
                    "/about" if index % 2 == 0 else "/projects/example",
                    f"s{index}",
                    f"Section {index}",
                    text,
                    index // 2,
                    Vector(vector),
                ),
            )


def test_hybrid_fuses_fts_only_vector_only_and_shared_hits(db_url: str) -> None:
    seed_hybrid(db_url)

    async def run() -> None:
        db = Database(make_pool(db_url))
        fake = FakeEmbedder({QUERY: unit_vector()})
        try:
            found = await retrieve(db, fake, QUERY)
            # s4 has vector rank 5 and keyword rank 2. s25 is keyword-only,
            # at cosine rank 26; s0 is vector-only. Equal scores use position.
            expected = [
                ("s4", 1 / (60 + 5) + 1 / (60 + 2)),
                ("s0", 1 / (60 + 1)),
                ("s25", 1 / (60 + 1)),
                ("s1", 1 / (60 + 2)),
                ("s2", 1 / (60 + 3)),
                ("s3", 1 / (60 + 4)),
            ]
            assert [section.section_id for section in found] == [item[0] for item in expected]
            for section, (_, score) in zip(found, expected, strict=True):
                assert isinstance(section, Section)
                assert section.score == pytest.approx(score)
                assert section.url == f"{section.page}#{section.section_id}"
            assert found[1].text == "ordinary content"
            assert found[2].text == "quasar quasar quasar"
            assert fake.calls == [[QUERY]]
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize("k", [0, 1, 3, 6, 50])
def test_k_and_twenty_semantic_candidates(db_url: str, k: int) -> None:
    seed_hybrid(db_url)

    async def run() -> None:
        db = Database(make_pool(db_url))
        try:
            found = await retrieve(db, FakeEmbedder({QUERY: unit_vector()}), QUERY, k=k)
            # Union of vector ranks 1..20 and keyword-only s25 has 21 members.
            order = [4, 0, 25, 1, 2, 3, *range(5, 20)]
            assert [section.section_id for section in found] == [f"s{i}" for i in order[:k]]
            assert len(found) == min(k, 21)
        finally:
            await db.close()

    asyncio.run(run())


def test_keyword_branch_also_has_twenty_candidates(db_url: str) -> None:
    seed_hybrid(db_url, all_keywords=True)

    async def run() -> None:
        db = Database(make_pool(db_url))
        try:
            found = await retrieve(db, FakeEmbedder({"nebula": unit_vector()}), "nebula", k=50)
            assert [section.section_id for section in found] == [f"s{i}" for i in range(20)]
            assert [section.score for section in found] == pytest.approx(
                [2 / (60 + index + 1) for index in range(20)]
            )
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize("pages", [None, [], ["/about"], ["/projects/example"]])
def test_page_filter_applies_before_both_rankings(db_url: str, pages: list[str] | None) -> None:
    seed_hybrid(db_url)

    async def run() -> None:
        db = Database(make_pool(db_url))
        fake = FakeEmbedder({QUERY: unit_vector()})
        try:
            found = await retrieve(db, fake, QUERY, pages=pages, k=50)
            if pages is None:
                assert len(found) == 21
                assert {section.page for section in found} == {"/about", "/projects/example"}
            elif not pages:
                assert found == []
            else:
                assert {section.page for section in found} == set(pages)
                # Filtering after LIMIT 20 would lose six vector candidates.
                assert len(found) == 16
                if pages == ["/about"]:
                    assert found[0].section_id == "s4"
                    assert found[0].score == pytest.approx(1 / (60 + 3) + 1 / (60 + 1))
                else:
                    assert found[0].section_id == "s25"
                    assert found[0].score == pytest.approx(1 / (60 + 13) + 1 / (60 + 1))
            assert fake.calls == [[QUERY]]
        finally:
            await db.close()

    asyncio.run(run())


def test_empty_database_returns_no_sections(db_url: str) -> None:
    migrate(db_url)

    async def run() -> None:
        db = Database(make_pool(db_url))
        fake = FakeEmbedder()
        try:
            assert await retrieve(db, fake, "anything") == []
            assert fake.calls == [["anything"]]
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize("query", ["", " ", "\t\n"])
def test_blank_query_never_calls_embedder_or_database(query: str) -> None:
    async def run() -> None:
        # An unusable pool proves there is no database access for blank queries.
        db = Database(object())
        fake = FakeEmbedder(fail_on_call=1)
        try:
            assert await retrieve(db, fake, query) == []
            assert fake.calls == []
        finally:
            await db.close()

    asyncio.run(run())


def test_stop_words_still_return_vector_results(db_url: str) -> None:
    seed_hybrid(db_url)

    async def run() -> None:
        db = Database(make_pool(db_url))
        query = "the and of"
        fake = FakeEmbedder({query: unit_vector()})
        try:
            found = await retrieve(db, fake, query)
            assert [section.section_id for section in found] == [f"s{i}" for i in range(6)]
            assert [section.score for section in found] == pytest.approx(
                [1 / (60 + i + 1) for i in range(6)]
            )
            assert fake.calls == [[query]]
        finally:
            await db.close()

    asyncio.run(run())


def test_only_current_version_is_searched_after_second_ingest(db_url: str, tmp_path: Path) -> None:
    migrate(db_url)
    path = tmp_path / "about.md"

    def write_page(heading: str) -> None:
        path.write_text(
            f"---\ntitle: About\nrag: true\n---\n## {heading}\nBody\n", encoding="utf-8"
        )

    async def run() -> None:
        db = Database(make_pool(db_url))
        fake = FakeEmbedder({QUERY: unit_vector()})
        try:
            await asyncio.to_thread(write_page, "Old only")
            old = await ingest(db, fake, tmp_path)
            await asyncio.to_thread(write_page, "New only")
            new = await ingest(db, fake, tmp_path)
            assert new.version > old.version
            found = await retrieve(db, fake, QUERY, k=50)
            assert [section.section_id for section in found] == ["new-only"]

            # Restore an old version as a stronger match, without changing the
            # current pointer. This catches retrieval that forgets the join even
            # though normal ingest deletes historical rows.
            async with db.connection() as conn, conn.transaction():
                await register_vector_async(conn)
                await conn.execute(
                    "INSERT INTO section_versions (version) OVERRIDING SYSTEM VALUE VALUES (%s)",
                    (old.version,),
                )
                await conn.execute(
                    "INSERT INTO sections "
                    "(version, page, section_id, heading, text, position, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        old.version,
                        "/about",
                        "old-only",
                        "Old only",
                        "quasar nebula",
                        0,
                        Vector(unit_vector()),
                    ),
                )
            found = await retrieve(db, fake, QUERY, k=50)
            assert [section.section_id for section in found] == ["new-only"]
        finally:
            await db.close()

    asyncio.run(run())


def test_database_errors_propagate_unwrapped(db_url: str) -> None:
    migrate(db_url)

    async def run() -> None:
        db = Database(make_pool(db_url))
        try:
            async with db.connection() as conn:
                await conn.execute("DROP TABLE sections")
            with pytest.raises(psycopg.errors.UndefinedTable):
                await retrieve(db, FakeEmbedder(), "query")
        finally:
            await db.close()

    asyncio.run(run())
