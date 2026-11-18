"""App logs reach the current stderr once; pool retries stay quiet."""

import logging

from app.logging_setup import configure_logging


def test_logging_is_idempotent_and_uses_current_stderr(capsys):
    configure_logging()
    configure_logging()
    logging.getLogger("app.api.ask").info("test-app-info")
    logging.getLogger("psycopg.pool").warning("test-pool-warning")
    logging.getLogger("psycopg.pool").error("test-pool-error")
    stderr = capsys.readouterr().err
    assert stderr.count("test-app-info") == 1
    assert "INFO app.api.ask: test-app-info" in stderr
    assert "test-pool-warning" not in stderr
    assert stderr.count("test-pool-error") == 1


def test_logging_preserves_propagation(caplog):
    configure_logging()
    logging.getLogger("app.cleanup").warning("test-cleanup-warning")
    assert [record.message for record in caplog.records] == ["test-cleanup-warning"]
