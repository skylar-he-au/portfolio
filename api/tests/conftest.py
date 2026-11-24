"""Shared fixtures.

Database tests need `TEST_DATABASE_URL` (a superuser URL to a throwaway server,
e.g. the `step2-api-testdb` container). Each test gets its own fresh database.
Without the variable those tests are skipped with the reason printed (`-rs`);
with `REQUIRE_DB_TESTS=1` (set in CI) a missing variable is an error instead.
"""

import os
import socket
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

# `app.main` builds a module-level app at import time; give it a harmless URL.
os.environ.setdefault("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
os.environ.setdefault("ORIGIN_SECRET", "test-origin")
os.environ.setdefault("SESSION_SECRET", "test-session")
os.environ.setdefault("TURNSTILE_SECRET", "1x0000000000000000000000000000000AA")
os.environ.setdefault("TURNSTILE_HOSTNAMES", "example.com")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini")
os.environ.setdefault("OPENAI_API_KEY", "test-openai")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-azure")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-small")
os.environ.setdefault("PATROL_SECRET", "test-patrol")
for period, cap in (("DAILY", "1"), ("MONTHLY", "10")):
    for provider in ("ANTHROPIC", "GOOGLE", "OPENAI"):
        os.environ.setdefault(f"SPEND_{period}_USD_{provider}", cap)

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
SKIP_REASON = "TEST_DATABASE_URL is not set; database tests skipped"


@pytest.fixture
def admin_url() -> str:
    if not TEST_DATABASE_URL:
        if os.environ.get("REQUIRE_DB_TESTS") == "1":
            pytest.fail("REQUIRE_DB_TESTS=1 but TEST_DATABASE_URL is not set")
        pytest.skip(SKIP_REASON)
    return TEST_DATABASE_URL


@pytest.fixture
def db_url(admin_url: str) -> Iterator[str]:
    """URL of a fresh, empty database, dropped after the test."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    params = conninfo_to_dict(admin_url)
    params["dbname"] = name
    try:
        yield make_conninfo(**params)
    finally:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def refused_url() -> str:
    """A URL whose port has nothing listening (connection refused)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"postgresql://nobody:secret-pw@127.0.0.1:{port}/none"


@pytest.fixture
def hanging_url() -> Iterator[str]:
    """A URL whose server accepts TCP but never answers."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(8)  # the kernel completes handshakes; nobody reads or writes
        port = s.getsockname()[1]
        yield f"postgresql://nobody:secret-pw@127.0.0.1:{port}/none"
