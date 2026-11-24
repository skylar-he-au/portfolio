"""Load eligible Markdown sections atomically: python -m app.ingest."""

import argparse
import asyncio
import logging
import re
import sys
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from openai import RateLimitError
from pgvector import Vector
from pgvector.psycopg import register_vector_async

from app.config import get_settings
from app.db import Database, make_pool
from app.rag.embed import AzureEmbedder, estimate_tokens
from app.rag.types import Embedder, Section

logger = logging.getLogger("app.ingest")
BATCH_SIZE = 64
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
# Pages write {{email}}; the web page turns it into a link built by a script. The
# address itself is never stored, so the indexed text names the link instead.
EMAIL_PLACEHOLDER = "{{email}}"
EMAIL_TEXT = "the email link on the About page"


class IngestError(ValueError):
    """An ingest validation error safe to show at the command line."""


@dataclass(frozen=True)
class ContentPage:
    title: str
    sections: list[Section]


@dataclass(frozen=True)
class Pacing:
    """Keeps ingest under an Azure deployment's tokens-per-minute limit.

    Each request carries at most half the TPM (by the high `estimate_tokens` count),
    the next one waits for the following window, and a 429 that outlasts the SDK's own
    retries waits for the server's `retry-after` (or a window) and tries again.
    """

    tokens_per_minute: int
    window_s: float = 60.0
    retries: int = 4

    @property
    def batch_tokens(self) -> int:
        return max(1, self.tokens_per_minute // 2)


def plan_batches(texts: list[str], pacing: Pacing | None) -> list[list[str]]:
    """Consecutive batches of at most BATCH_SIZE texts and, with pacing, at most
    `batch_tokens` estimated tokens; a text over the limit on its own goes alone."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = estimate_tokens([text])
        full = len(current) >= BATCH_SIZE or (
            pacing is not None and current_tokens + tokens > pacing.batch_tokens
        )
        if current and full:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def _retry_after_s(error: RateLimitError) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            return float(headers[name]) / scale
        except (KeyError, TypeError, ValueError):
            continue
    return None


async def _embed_paced(
    embedder: Embedder,
    batches: list[list[str]],
    pacing: Pacing | None,
    sleep: Callable[[float], Awaitable[None]],
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for index, batch in enumerate(batches):
        if pacing is not None and index > 0:
            await sleep(pacing.window_s)
        attempt = 0
        while True:
            try:
                vectors = await embedder.embed(batch)
                break
            except RateLimitError as error:
                attempt += 1
                if pacing is None or attempt > pacing.retries:
                    raise
                wait = max(_retry_after_s(error) or 0.0, pacing.window_s)
                logger.warning(
                    "embedding batch %d/%d rate limited; retry %d/%d in %.0f s",
                    index + 1,
                    len(batches),
                    attempt,
                    pacing.retries,
                    wait,
                )
                await sleep(wait)
        if len(vectors) != len(batch):
            raise IngestError("embedder returned the wrong number of vectors")
        embeddings.extend(vectors)
    return embeddings


@dataclass(frozen=True)
class IngestResult:
    version: int
    pages: int
    sections: int


def slugify(heading: str) -> str:
    """Slug rendered heading text using github-slugger's Unicode rules."""
    heading = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    heading = heading.replace("`", "").replace("*", "")
    # Preserve underscores within words, but remove paired emphasis delimiters.
    heading = re.sub(r"(?<!\w)(_+)(\S(?:.*?\S)?)\1(?!\w)", r"\2", heading)
    return "".join(
        char
        for char in heading.lower()
        if char in " -"
        or unicodedata.category(char)[0] in "LM"
        or unicodedata.category(char) in {"Nd", "Pc"}
    ).replace(" ", "-")


def split_sections(body: str, page: str) -> list[Section]:
    """Split on level-two headings outside backtick and tilde fences."""
    sections: list[Section] = []
    used: set[str] = set()
    suffixes: dict[str, int] = {}
    heading: str | None = None
    lines: list[str] = []
    fence_char = ""
    fence_length = 0

    def finish() -> None:
        if heading is None:
            return
        base = slugify(heading)
        section_id = base
        while section_id in used:
            suffixes[base] = suffixes.get(base, 0) + 1
            section_id = f"{base}-{suffixes[base]}"
        used.add(section_id)
        start, end = 0, len(lines)
        while start < end and not lines[start].strip():
            start += 1
        while end > start and not lines[end - 1].strip():
            end -= 1
        text = "\n".join(lines[start:end]).replace(EMAIL_PLACEHOLDER, EMAIL_TEXT)
        sections.append(Section(page, section_id, heading, text))

    for line in body.splitlines():
        fence = FENCE_RE.match(line)
        if fence:
            marker, rest = fence.groups()
            if fence_char:
                if marker[0] == fence_char and len(marker) >= fence_length and not rest.strip():
                    fence_char = ""
            elif marker[0] != "`" or "`" not in rest:
                fence_char, fence_length = marker[0], len(marker)
        if not fence_char and line.startswith("## "):
            finish()
            heading = re.sub(r"\s+#+\s*$", "", line[3:]).strip()
            lines = []
        elif heading is not None:
            lines.append(line)
    finish()
    return sections


def read_page(path: Path, content_dir: Path) -> ContentPage | None:
    """Read the simple key/value front matter used by this content tree."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return None
    metadata: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line == "---":
            body = "\n".join(lines[index + 1 :])
            break
        key, separator, value = line.partition(":")
        if separator:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            metadata[key.strip()] = value
    else:
        return None
    if metadata.get("rag", "").lower() != "true":
        return None
    if metadata.get("draft", "").lower() == "true":
        return None
    if not metadata.get("title"):
        raise IngestError("rag content must have a title")
    page = "/" + path.relative_to(content_dir).with_suffix("").as_posix()
    return ContentPage(metadata["title"], split_sections(body, page))


def load_content(content_dir: Path) -> list[ContentPage]:
    return [
        page
        for path in sorted(content_dir.rglob("*.md"))
        if (page := read_page(path, content_dir)) is not None
    ]


async def ingest(
    db: Database,
    embedder: Embedder,
    content_dir: Path,
    *,
    pacing: Pacing | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> IngestResult:
    """Embed every eligible section, then swap the stored version in one transaction.

    Without `pacing` (tests with a fake embedder) texts go in BATCH_SIZE batches with no
    waiting; the command line always paces for the configured deployment."""
    pages = await asyncio.to_thread(load_content, content_dir)
    records = [
        (section, position, f"{page.title} — {section.heading}\n\n{section.text}")
        for page in pages
        for position, section in enumerate(page.sections)
    ]
    if not records:
        raise IngestError("content directory contains no eligible sections")
    batches = plan_batches([text for _, _, text in records], pacing)
    if pacing is not None:
        logger.info(
            "embedding %d sections in %d batches (about %d tokens, %d per minute)",
            len(records),
            len(batches),
            estimate_tokens([text for _, _, text in records]),
            pacing.batch_tokens,
        )
    embeddings = await _embed_paced(embedder, batches, pacing, sleep)

    async with db.connection() as conn, conn.transaction():
        await register_vector_async(conn)
        # Serialize writers even on the first run, when content_state has no row.
        await conn.execute("LOCK TABLE content_state IN EXCLUSIVE MODE")
        cursor = await conn.execute("INSERT INTO section_versions DEFAULT VALUES RETURNING version")
        version = (await cursor.fetchone())[0]
        async with conn.cursor() as cursor:
            await cursor.executemany(
                "INSERT INTO sections "
                "(version, page, section_id, heading, text, position, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        version,
                        section.page,
                        section.section_id,
                        section.heading,
                        section.text,
                        position,
                        Vector(vector),
                    )
                    for (section, position, _), vector in zip(records, embeddings, strict=True)
                ],
            )
        await conn.execute(
            "INSERT INTO content_state (id, current_version) VALUES (true, %s) "
            "ON CONFLICT (id) DO UPDATE SET current_version = EXCLUDED.current_version",
            (version,),
        )
        await conn.execute("DELETE FROM sections WHERE version <> %s", (version,))
        await conn.execute("DELETE FROM section_versions WHERE version <> %s", (version,))

    result = IngestResult(version, len(pages), len(records))
    logger.info(
        "ingested version %d: %d pages, %d sections", result.version, result.pages, result.sections
    )
    return result


async def _run(content_dir: Path | None) -> None:
    settings = get_settings()
    embedder = AzureEmbedder(settings)
    db = Database(make_pool(settings.database_url.get_secret_value()))
    try:
        await ingest(
            db,
            embedder,
            content_dir if content_dir is not None else settings.content_dir,
            pacing=Pacing(settings.azure_openai_tpm),
        )
    finally:
        await db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-dir", type=Path, help="Markdown content directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_run(args.content_dir))
    except IngestError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI errors must never expose keys or database URLs
        logger.error("ingest failed: %s", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
