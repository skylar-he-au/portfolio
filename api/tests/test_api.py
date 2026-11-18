import time

import psycopg
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings
from app.main import create_app
from app.migrate import migrate


def settings(url: str, env: str = "dev") -> Settings:
    return Settings(_env_file=None, database_url=SecretStr(url), app_env=env)  # type: ignore[arg-type]


class ExplodingPool:
    """Fails the test on any use: proves a code path never touches the pool."""

    touched: list[str]

    def __init__(self) -> None:
        object.__setattr__(self, "touched", [])

    def __getattr__(self, name: str):
        self.touched.append(name)
        raise AssertionError(f"pool.{name} was used")


# --- /api/ping ---------------------------------------------------------------


def test_ping_ok_with_unreachable_database(refused_url: str) -> None:
    with TestClient(create_app(settings(refused_url))) as client:
        resp = client.get("/api/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_startup_and_ping_never_touch_pool(refused_url: str) -> None:
    pool = ExplodingPool()
    for env in ("dev", "prod"):
        app = create_app(settings(refused_url, env), pool=pool)
        with TestClient(app) as client:  # runs startup and shutdown
            assert client.get("/api/ping").status_code == 200
    assert pool.touched == []


def test_default_pause_source_is_the_app_ledger(refused_url):
    app = create_app(settings(refused_url))
    assert app.state.limits._pause_source is app.state.ledger
    assert app.state.ledger.db is app.state.db


# --- /api/health -------------------------------------------------------------


def test_health_absent_outside_dev(refused_url: str) -> None:
    with TestClient(create_app(settings(refused_url, "prod"))) as client:
        assert client.get("/api/health").status_code == 404
        # No schema or docs pages either.
        assert client.get("/api/openapi.json").status_code == 404
        assert client.get("/api/docs").status_code == 404


def test_health_503_when_connection_refused(refused_url: str) -> None:
    with TestClient(create_app(settings(refused_url))) as client:
        start = time.monotonic()
        resp = client.get("/api/health")
        elapsed = time.monotonic() - start
    assert resp.status_code == 503
    assert resp.json() == {"status": "degraded", "db": "down", "pgvector": None}
    body = resp.text
    for leak in ("secret-pw", "nobody", "127.0.0.1", "postgresql", "refused"):
        assert leak not in body
    assert elapsed < 3


def test_health_503_within_3s_when_server_hangs(hanging_url: str) -> None:
    with TestClient(create_app(settings(hanging_url))) as client:
        start = time.monotonic()
        resp = client.get("/api/health")
        elapsed = time.monotonic() - start
    assert resp.status_code == 503
    assert resp.json() == {"status": "degraded", "db": "down", "pgvector": None}
    assert elapsed < 3, f"took {elapsed:.2f}s"


def test_health_ok_reports_pgvector_version(db_url: str) -> None:
    migrate(db_url)
    with psycopg.connect(db_url) as conn:
        expected = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname='vector'"
        ).fetchone()
    assert expected is not None
    with TestClient(create_app(settings(db_url))) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": "ok", "pgvector": expected[0]}


def test_health_503_when_pgvector_missing(db_url: str) -> None:
    # Database reachable but migrations not run yet.
    with TestClient(create_app(settings(db_url))) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 503
    assert resp.json() == {"status": "degraded", "db": "ok", "pgvector": None}
