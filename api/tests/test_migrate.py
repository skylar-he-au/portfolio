import os
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest

from app.migrate import DEFAULT_DIR, LOCK_KEY, MigrationError, discover, migrate

API_DIR = Path(__file__).resolve().parent.parent


def versions(url: str) -> list[str]:
    with psycopg.connect(url) as conn:
        return [
            r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")
        ]


def table_exists(url: str, name: str) -> bool:
    with psycopg.connect(url) as conn:
        return conn.execute("SELECT to_regclass(%s) IS NOT NULL", (name,)).fetchone()[0]


def advisory_lock_held(url: str) -> bool:
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'"
            " AND ((classid::bigint << 32) | objid::bigint) = %s",
            (LOCK_KEY,),
        ).fetchone()
        return row[0] > 0


def run_cli(url: str, directory: Path) -> subprocess.Popen[str]:
    env = {**os.environ, "DATABASE_URL": url}
    return subprocess.Popen(
        [sys.executable, "-m", "app.migrate", "--dir", str(directory)],
        cwd=API_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


# --- no database needed ------------------------------------------------------


def test_discover_sorts_and_validates(tmp_path: Path) -> None:
    (tmp_path / "0002_b.sql").write_text("SELECT 1;")
    (tmp_path / "0001_a.sql").write_text("SELECT 1;")
    (tmp_path / "README.md").write_text("ignored")
    assert [v for v, _ in discover(tmp_path)] == ["0001", "0002"]

    (tmp_path / "0002_dup.sql").write_text("SELECT 1;")
    with pytest.raises(MigrationError, match="duplicate"):
        discover(tmp_path)


def test_discover_rejects_bad_name(tmp_path: Path) -> None:
    (tmp_path / "1_init.sql").write_text("SELECT 1;")
    with pytest.raises(MigrationError, match="bad migration file name"):
        discover(tmp_path)


# --- against a real database --------------------------------------------------


def test_empty_database_then_rerun_is_noop(db_url: str) -> None:
    expected = [version for version, _ in discover(DEFAULT_DIR)]
    assert migrate(db_url) == expected
    assert versions(db_url) == expected
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT 1 FROM pg_extension WHERE extname='vector'").fetchone()

    assert migrate(db_url) == []
    assert versions(db_url) == expected


def test_concurrent_runs_apply_once(db_url: str, tmp_path: Path) -> None:
    # Slow migration with a visible side effect; both runs start before it ends.
    (tmp_path / "0001_slow.sql").write_text(
        "CREATE TABLE runs (n int);\nSELECT pg_sleep(1);\nINSERT INTO runs VALUES (1);\n"
    )
    procs = [run_cli(db_url, tmp_path) for _ in range(2)]
    outputs = [p.communicate(timeout=30)[0] for p in procs]
    codes = [p.returncode for p in procs]

    assert codes == [0, 0], outputs
    assert versions(db_url) == ["0001"]
    with psycopg.connect(db_url) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    joined = "\n".join(outputs)
    assert joined.count("applying 0001_slow.sql") == 1, joined


def test_failing_migration_rolls_back_and_resumes(db_url: str, tmp_path: Path) -> None:
    (tmp_path / "0001_ok.sql").write_text("CREATE TABLE a (id int);")
    bad = tmp_path / "0002_bad.sql"
    # First statement succeeds, second fails: the first must be rolled back too.
    bad.write_text("CREATE TABLE b (id int);\nSELEC broken;")
    (tmp_path / "0003_after.sql").write_text("CREATE TABLE c (id int);")

    proc = run_cli(db_url, tmp_path)
    out = proc.communicate(timeout=30)[0]
    assert proc.returncode != 0, out
    assert "0002_bad.sql failed" in out
    assert versions(db_url) == ["0001"]
    assert table_exists(db_url, "a")
    assert not table_exists(db_url, "b")  # rolled back
    assert not table_exists(db_url, "c")  # never ran
    assert not advisory_lock_held(db_url)

    bad.write_text("CREATE TABLE b (id int);")
    proc = run_cli(db_url, tmp_path)
    out = proc.communicate(timeout=30)[0]
    assert proc.returncode == 0, out
    assert versions(db_url) == ["0001", "0002", "0003"]
    assert table_exists(db_url, "b") and table_exists(db_url, "c")


def test_lock_released_after_success(db_url: str) -> None:
    migrate(db_url)
    assert not advisory_lock_held(db_url)
