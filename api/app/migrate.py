"""Apply `migrations/NNNN_name.sql` files in order.

Usage: python -m app.migrate [--dir PATH]

- Each file runs in its own transaction; its version (the `NNNN` prefix) is
  recorded in `schema_migrations` in that same transaction.
- The whole run holds a session-level `pg_advisory_lock`, so two instances
  started together apply each file once. Session locks need a direct
  connection: on Neon use the non-pooled URL (no `-pooler` in the host).
- The first failing file is rolled back and nothing after it runs; the exit
  code is non-zero. Fix the file and run again to continue from there.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import psycopg

from app.config import get_settings

logger = logging.getLogger("app.migrate")

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "migrations"
# Fixed key for pg_advisory_lock; any constant works if nothing else uses it.
LOCK_KEY = 7_412_093_551_208_330_241
LOCK_WAIT = "120s"
FILE_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


class MigrationError(Exception):
    pass


def discover(directory: Path) -> list[tuple[str, Path]]:
    """Return (version, path) pairs sorted by version; reject bad or duplicate names."""
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix != ".sql":
            continue
        match = FILE_RE.match(path.name)
        if not match:
            raise MigrationError(f"bad migration file name: {path.name} (want NNNN_name.sql)")
        version = match.group(1)
        if version in found:
            raise MigrationError(
                f"duplicate migration number {version}: {found[version].name}, {path.name}"
            )
        found[version] = path
    return sorted(found.items())


def migrate(database_url: str, directory: Path = DEFAULT_DIR) -> list[str]:
    """Apply pending migrations; return the versions applied in this run."""
    migrations = discover(directory)
    applied_now: list[str] = []
    with psycopg.connect(database_url, autocommit=True, connect_timeout=10) as conn:
        conn.execute(f"SET lock_timeout = '{LOCK_WAIT}'")
        conn.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
        conn.execute("RESET lock_timeout")
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version text PRIMARY KEY,"
                " applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
            for version, path in migrations:
                if version in done:
                    continue
                logger.info("applying %s", path.name)
                try:
                    with conn.transaction():
                        # No parameters, so psycopg sends the file as one simple
                        # query and multiple statements are allowed.
                        conn.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
                        conn.execute(
                            "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
                        )
                except psycopg.Error as exc:
                    raise MigrationError(f"{path.name} failed: {exc}") from exc
                applied_now.append(version)
        finally:
            # If the connection broke, the server already dropped the lock.
            if not conn.closed:
                conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
    return applied_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.migrate", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="migrations directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        applied = migrate(get_settings().database_url.get_secret_value(), args.dir)
    except MigrationError as exc:
        logger.error("%s", exc)
        return 1
    except psycopg.Error as exc:
        # Connection or lock errors: report the kind, not the message (it can hold the host).
        logger.error("migration run failed: %s", type(exc).__name__)
        return 1
    logger.info("applied %d migration(s): %s", len(applied), ", ".join(applied) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
