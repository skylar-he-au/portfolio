"""Hybrid cosine/full-text retrieval with reciprocal rank fusion."""

from collections.abc import Sequence

from pgvector import Vector
from pgvector.psycopg import register_vector_async

from app.db import Database
from app.rag.types import Embedder, Section

SEARCH_SQL = """
WITH current_sections AS (
    SELECT s.*
    FROM sections s
    JOIN content_state c ON c.current_version = s.version
    WHERE c.id AND (%(pages)s::text[] IS NULL OR s.page = ANY(%(pages)s::text[]))
), semantic_search AS (
    SELECT page, section_id,
           ROW_NUMBER() OVER (
               ORDER BY embedding <=> %(embedding)s, position, page, section_id
           ) AS rank
    FROM current_sections
    ORDER BY embedding <=> %(embedding)s, position, page, section_id
    LIMIT 20
), keyword_search AS (
    SELECT page, section_id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(tsv, query) DESC, position, page, section_id
           ) AS rank
    FROM current_sections, websearch_to_tsquery('english', %(query)s) query
    WHERE tsv @@ query
    ORDER BY ts_rank_cd(tsv, query) DESC, position, page, section_id
    LIMIT 20
), fused AS (
    SELECT page, section_id,
           COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k USING (page, section_id)
)
SELECT c.page, c.section_id, c.heading, c.text, f.score
FROM fused f
JOIN current_sections c USING (page, section_id)
ORDER BY f.score DESC, c.position, c.page, c.section_id
LIMIT %(limit)s
"""


async def retrieve(
    db: Database,
    embedder: Embedder,
    query: str,
    *,
    k: int = 6,
    pages: Sequence[str] | None = None,
) -> list[Section]:
    if not query.strip():
        return []
    vectors = await embedder.embed([query])
    async with db.connection() as conn:
        await register_vector_async(conn)
        cursor = await conn.execute(
            SEARCH_SQL,
            {
                "embedding": Vector(vectors[0]),
                "query": query,
                "pages": list(pages) if pages is not None else None,
                "limit": k,
            },
        )
        rows = await cursor.fetchall()
    return [
        Section(page, section_id, heading, text, float(score))
        for page, section_id, heading, text, score in rows
    ]
