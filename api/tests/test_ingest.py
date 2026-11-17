import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
import pytest
from fakes import FakeEmbedder, unit_vector
from psycopg_pool import PoolTimeout

import app.ingest as ingest_module
from app.config import Settings
from app.db import POOL_WAIT_S, Database, make_pool
from app.ingest import IngestError, ingest, read_page, slugify, split_sections
from app.migrate import migrate
from app.rag.embed import OpenAIEmbedder
from app.rag.retrieve import retrieve

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "content"
PAGES = ["/about", "/privacy", "/projects/roleradar", "/projects/oncallops"]


def section_ids() -> dict[str, list[dict[str, str]]]:
    return json.loads((ROOT / "web/tests/fixtures/section-ids.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("page", PAGES)
def test_real_sections_match_web_ids(page: str) -> None:
    parsed = read_page(CONTENT / f"{page.lstrip('/')}.md", CONTENT)
    assert parsed is not None
    actual = [(section.heading, section.section_id) for section in parsed.sections]
    expected = [(item["heading"], item["id"]) for item in section_ids()[page]]
    assert actual == expected


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("Hello, WORLD!", "hello-world"),
        ("Tests: evals & prompts?", "tests-evals--prompts"),
        ("A - B", "a---b"),
        (" A  B ", "-a--b-"),
        ("Crème 中文 \uff11\uff12", "crème-中文-\uff11\uff12"),
        ("e\u0301lan", "e\u0301lan"),
        ("\u2460 \u00b2 \u00bd", "--"),
        ("snake_case undertie\u203f", "snake_case-undertie\u203f"),
        ("**Bold** _emphasis_ `code` [a link](https://example.com)", "bold-emphasis-code-a-link"),
    ],
)
def test_slugify(heading: str, expected: str) -> None:
    assert slugify(heading) == expected


def test_duplicate_slugs_avoid_generated_and_literal_collisions() -> None:
    sections = split_sections("## Foo\n## Foo\n## Foo-1\n## Foo\n## Foo-1", "/test")
    assert [section.section_id for section in sections] == [
        "foo",
        "foo-1",
        "foo-1-1",
        "foo-2",
        "foo-1-2",
    ]


def test_split_ignores_preamble_subheadings_and_fenced_headings() -> None:
    body = """Preamble is dropped.
```md
## Not a section
```
## First ###

    keep indentation
### Child
```markdown
## Inside backticks
~~~
```

## Second

~~~md
## Inside tildes
```
~~~
tail

"""
    sections = split_sections(body, "/test")
    assert [(section.heading, section.section_id) for section in sections] == [
        ("First", "first"),
        ("Second", "second"),
    ]
    assert sections[0].text == (
        "    keep indentation\n### Child\n```markdown\n## Inside backticks\n~~~\n```"
    )
    assert sections[1].text == "~~~md\n## Inside tildes\n```\n~~~\ntail"


@pytest.mark.parametrize("filename", ["home.md", "examples.md", "LICENSE"])
def test_ineligible_real_files_are_skipped(filename: str) -> None:
    assert read_page(CONTENT / filename, CONTENT) is None


def test_quoted_front_matter_and_markdown_heading(tmp_path: Path) -> None:
    path = tmp_path / "page.md"
    path.write_text(
        '---\ntitle: "A title: with colon"\nrag: true\n---\n## **Bold** _text_\n\nBody\n',
        encoding="utf-8",
    )
    parsed = read_page(path, tmp_path)
    assert parsed is not None
    assert parsed.title == "A title: with colon"
    assert parsed.sections[0].heading == "**Bold** _text_"
    assert parsed.sections[0].section_id == "bold-text"


def test_openai_embedder_uses_configured_client(monkeypatch: pytest.MonkeyPatch) -> None:
    options: dict[str, object] = {}
    calls: list[list[str]] = []
    vectors = [unit_vector(), unit_vector(1)]

    class StubEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            options.update(kwargs)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return vectors

    monkeypatch.setattr("app.rag.embed.OpenAIEmbeddings", StubEmbeddings)
    settings = Settings(
        _env_file=None,
        database_url="postgresql://unused",
        openai_api_key="stub-key",
        embed_model="configured-model",
        embed_dims=1536,
    )
    embedder = OpenAIEmbedder(settings)
    assert asyncio.run(embedder.embed(["one", "two"])) == vectors
    assert calls == [["one", "two"]]
    assert options == {
        "model": "configured-model",
        "dimensions": 1536,
        "api_key": settings.openai_api_key,
        "timeout": 30,
        "max_retries": 2,
        "check_embedding_ctx_length": False,
    }


def test_openai_embedder_sends_text_without_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real client sends raw strings and never loads a tiktoken encoding."""
    sent: list[object] = []

    class StubResponse:
        def model_dump(self) -> dict[str, object]:
            return {"data": [{"embedding": unit_vector()}]}

    class StubAsyncClient:
        async def create(self, *, input: object, **_kwargs: object) -> StubResponse:
            sent.append(input)
            return StubResponse()

    def no_tiktoken(*_args: object, **_kwargs: object) -> None:
        pytest.fail("tiktoken must not be loaded")

    monkeypatch.setattr("tiktoken.encoding_for_model", no_tiktoken)
    monkeypatch.setattr("tiktoken.get_encoding", no_tiktoken)
    settings = Settings(
        _env_file=None, database_url="postgresql://unused", openai_api_key="stub-key"
    )
    embedder = OpenAIEmbedder(settings)
    assert embedder._client.check_embedding_ctx_length is False
    embedder._client.async_client = StubAsyncClient()
    assert asyncio.run(embedder.embed(["What did Skylar build?"])) == [unit_vector()]
    assert sent == [["What did Skylar build?"]]


@pytest.mark.parametrize("key", [None, ""])
def test_openai_embedder_requires_key(key: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_client(**_kwargs: object) -> None:
        pytest.fail("client must not be created without a key")

    monkeypatch.setattr("app.rag.embed.OpenAIEmbeddings", unexpected_client)
    settings = Settings(_env_file=None, database_url="postgresql://unused", openai_api_key=key)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        OpenAIEmbedder(settings)


class StubPool:
    def __init__(self) -> None:
        self.conn = object()
        self.opens: list[bool] = []
        self.timeouts: list[float] = []
        self.returned = 0
        self.closed = False
        self.error: Exception | None = None

    async def open(self, *, wait: bool) -> None:
        self.opens.append(wait)

    @asynccontextmanager
    async def connection(self, **options: float) -> AsyncIterator[object]:
        self.timeouts.append(options["timeout"])
        if self.error is not None:
            raise self.error
        try:
            yield self.conn
        finally:
            self.returned += 1

    async def close(self) -> None:
        self.closed = True


def test_connection_is_lazy_and_returns_borrowed_connection() -> None:
    async def run() -> None:
        pool = StubPool()
        db = Database(pool)
        assert pool.opens == []
        try:
            for _ in range(2):
                async with db.connection() as conn:
                    assert conn is pool.conn
            assert pool.opens == [False]
            assert pool.timeouts == [POOL_WAIT_S, POOL_WAIT_S]
            assert pool.returned == 2
        finally:
            await db.close()
        assert pool.closed

    asyncio.run(run())


@pytest.mark.parametrize("error", [psycopg.OperationalError("private"), PoolTimeout("private")])
def test_connection_propagates_errors_unwrapped(error: Exception) -> None:
    async def run() -> None:
        pool = StubPool()
        db = Database(pool)
        try:
            pool.error = error
            with pytest.raises(type(error)) as caught:
                async with db.connection():
                    pytest.fail("borrow should fail")
            assert caught.value is error
            pool.error = None
            with pytest.raises(type(error)) as caught:
                async with db.connection():
                    raise error
            assert caught.value is error
            assert pool.returned == 1
        finally:
            await db.close()

    asyncio.run(run())


async def stored_state(
    db: Database,
) -> tuple[int | None, list[int], list[tuple[int, str, str, str, str, int]]]:
    async with db.connection() as conn:
        cursor = await conn.execute("SELECT current_version FROM content_state")
        current = await cursor.fetchone()
        cursor = await conn.execute("SELECT version FROM section_versions ORDER BY version")
        versions = [row[0] for row in await cursor.fetchall()]
        cursor = await conn.execute(
            "SELECT version, page, section_id, heading, text, position "
            "FROM sections ORDER BY page, position"
        )
        rows = await cursor.fetchall()
    current_version = current[0] if current else None
    return current_version, versions, rows


def test_sections_migration_starts_empty_without_vector_index(db_url: str) -> None:
    migrate(db_url)
    with psycopg.connect(db_url) as conn:
        counts = conn.execute(
            "SELECT (SELECT count(*) FROM section_versions), "
            "(SELECT count(*) FROM content_state), (SELECT count(*) FROM sections)"
        ).fetchone()
        assert counts == (0, 0, 0)
        embedding_type = conn.execute(
            "SELECT format_type(atttypid, atttypmod), attnotnull FROM pg_attribute "
            "WHERE attrelid = 'sections'::regclass AND attname = 'embedding'"
        ).fetchone()
        assert embedding_type == ("vector(1536)", True)
        generated = conn.execute(
            "SELECT attgenerated FROM pg_attribute "
            "WHERE attrelid = 'sections'::regclass AND attname = 'tsv'"
        ).fetchone()
        assert generated == ("s",)
        indexes = conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'sections'")
        assert all("USING hnsw" not in row[0] and "USING ivfflat" not in row[0] for row in indexes)


def test_ingest_real_content_embedding_format_and_safe_logs(
    db_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    migrate(db_url)
    caplog.set_level(logging.INFO, logger="app.ingest")

    async def run() -> None:
        db = Database(make_pool(db_url))
        fake = FakeEmbedder()
        try:
            result = await ingest(db, fake, CONTENT)
            current, versions, rows = await stored_state(db)
            expected = section_ids()
            assert {row[1] for row in rows} == set(PAGES)
            assert len(rows) == result.sections == sum(len(items) for items in expected.values())
            assert result.pages == 4
            assert current == result.version
            assert versions == [result.version]
            for page in PAGES:
                positions = [row[5] for row in rows if row[1] == page]
                assert positions == list(range(len(expected[page])))
            about = read_page(CONTENT / "about.md", CONTENT)
            assert about is not None
            assert len(fake.calls) == 1
            assert fake.calls[0][0] == f"About Skylar — About me\n\n{about.sections[0].text}"
            messages = [
                record.getMessage() for record in caplog.records if record.name == "app.ingest"
            ]
            assert messages == [f"ingested version {result.version}: 4 pages, {len(rows)} sections"]
            for row in rows:
                assert row[4] not in caplog.text
        finally:
            await db.close()

    asyncio.run(run())


def test_ingest_replaces_previous_version(db_url: str) -> None:
    migrate(db_url)

    async def run() -> None:
        db = Database(make_pool(db_url))
        try:
            first = await ingest(db, FakeEmbedder(), CONTENT)
            second = await ingest(db, FakeEmbedder(), CONTENT)
            current, versions, rows = await stored_state(db)
            assert second.version > first.version
            assert current == second.version
            assert versions == [second.version]
            assert len(rows) == first.sections == second.sections
            assert {row[0] for row in rows} == {second.version}
        finally:
            await db.close()

    asyncio.run(run())


def test_second_embedding_batch_failure_preserves_queryable_version(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrate(db_url)

    async def run() -> None:
        db = Database(make_pool(db_url))
        try:
            await ingest(db, FakeEmbedder(), CONTENT)
            before = await stored_state(db)
            monkeypatch.setattr(ingest_module, "BATCH_SIZE", 1)
            failing = FakeEmbedder(fail_on_call=2)
            with pytest.raises(RuntimeError, match="fake embedding failure"):
                await ingest(db, failing, CONTENT)
            assert len(failing.calls) == 2
            assert await stored_state(db) == before
            found = await retrieve(db, FakeEmbedder(), "work rights")
            assert found
            old_text = {(row[1], row[2]): row[4] for row in before[2]}
            assert all(
                old_text[(section.page, section.section_id)] == section.text for section in found
            )
        finally:
            await db.close()

    asyncio.run(run())


def test_database_write_failure_rolls_back_version_and_sections(db_url: str) -> None:
    migrate(db_url)

    async def run() -> None:
        db = Database(make_pool(db_url))
        try:
            recording = FakeEmbedder()
            await ingest(db, recording, CONTENT)
            before = await stored_state(db)
            # First insert is valid; the second violates vector(1536).
            broken = FakeEmbedder({recording.calls[0][1]: [1.0]})
            with pytest.raises(psycopg.Error):
                await ingest(db, broken, CONTENT)
            assert await stored_state(db) == before
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "body", [None, "rag: false\n---\n## Ignored", "rag: true\n---\nNo heading"]
)
def test_no_sections_never_deletes_current_content(
    db_url: str, tmp_path: Path, body: str | None
) -> None:
    migrate(db_url)
    if body is not None:
        (tmp_path / "empty.md").write_text(f"---\ntitle: Empty\n{body}\n", encoding="utf-8")

    async def run() -> None:
        db = Database(make_pool(db_url))
        try:
            await ingest(db, FakeEmbedder(), CONTENT)
            before = await stored_state(db)
            fake = FakeEmbedder()
            with pytest.raises(IngestError, match="no eligible sections"):
                await ingest(db, fake, tmp_path)
            assert fake.calls == []
            assert await stored_state(db) == before
        finally:
            await db.close()

    asyncio.run(run())


def test_embedding_failure_happens_before_database_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "page.md").write_text(
        "---\ntitle: Test\nrag: true\n---\n## First\nOne\n## Second\nTwo\n", encoding="utf-8"
    )
    monkeypatch.setattr(ingest_module, "BATCH_SIZE", 1)

    async def run() -> None:
        db = Database(object())
        fake = FakeEmbedder(fail_on_call=2)
        try:
            with pytest.raises(RuntimeError, match="fake embedding failure"):
                await ingest(db, fake, tmp_path)
            assert len(fake.calls) == 2
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize("override", [None, Path("override-content")])
@pytest.mark.parametrize("fail", [False, True])
def test_run_uses_settings_and_closes_database(
    override: Path | None, fail: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://unused",
        content_dir=Path("configured-content"),
        openai_api_key=None,
    )
    fake = FakeEmbedder()
    pool = StubPool()
    db = Database(pool)
    urls: list[str] = []
    received: list[Path] = []

    def make_test_pool(url: str) -> StubPool:
        urls.append(url)
        return pool

    def make_test_db(actual_pool: StubPool) -> Database:
        assert actual_pool is pool
        return db

    def make_test_embedder(actual_settings: Settings) -> FakeEmbedder:
        assert actual_settings is settings
        return fake

    async def load(actual_db: Database, actual_embedder: FakeEmbedder, directory: Path) -> None:
        assert actual_db is db
        assert actual_embedder is fake
        received.append(directory)
        async with actual_db.connection():
            if fail:
                raise IngestError("content directory contains no eligible sections")

    monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_module, "make_pool", make_test_pool)
    monkeypatch.setattr(ingest_module, "Database", make_test_db)
    monkeypatch.setattr(ingest_module, "OpenAIEmbedder", make_test_embedder)
    monkeypatch.setattr(ingest_module, "ingest", load)
    if fail:
        with pytest.raises(IngestError):
            asyncio.run(ingest_module._run(override))
    else:
        asyncio.run(ingest_module._run(override))
    assert urls == [settings.database_url.get_secret_value()]
    assert received == [settings.content_dir if override is None else override]
    assert pool.closed


@pytest.mark.parametrize("content_dir", [None, Path("alternate-content")])
def test_cli_passes_content_directory_and_returns_success(
    content_dir: Path | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[Path | None] = []

    async def run(directory: Path | None) -> None:
        received.append(directory)

    monkeypatch.setattr(ingest_module, "_run", run)
    args = [] if content_dir is None else ["--content-dir", str(content_dir)]
    assert ingest_module.main(args) == 0
    assert received == [content_dir]


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("postgresql://secret:password@private-host/db"),
        RuntimeError("private section text and key"),
        IngestError("content directory contains no eligible sections"),
    ],
)
def test_cli_errors_are_one_line_and_do_not_leak_secrets(
    error: Exception, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def run(_directory: Path | None) -> None:
        raise error

    monkeypatch.setattr(ingest_module, "_run", run)
    assert ingest_module.main([]) == 1
    messages = [record.getMessage() for record in caplog.records if record.name == "app.ingest"]
    assert len(messages) == 1
    assert "\n" not in messages[0]
    assert "private" not in messages[0]
    assert "password" not in messages[0]
    if isinstance(error, IngestError):
        assert "no eligible sections" in messages[0]
    else:
        assert type(error).__name__ in messages[0]
