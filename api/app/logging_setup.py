"""App diagnostics on the current stderr, without noisy pool retry warnings."""

import logging
import sys
from typing import TextIO


class _StderrHandler(logging.StreamHandler):
    def __init__(self) -> None:
        # StreamHandler.__init__ assigns a fixed stream; resolve it on each emit instead.
        logging.Handler.__init__(self)

    @property
    def stream(self) -> TextIO:
        return sys.stderr


def configure_logging() -> None:
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    handler = next((h for h in app_logger.handlers if isinstance(h, _StderrHandler)), None)
    if handler is None:
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        app_logger.addHandler(handler)
    pool_logger = logging.getLogger("psycopg.pool")
    pool_logger.setLevel(logging.ERROR)
    if not any(isinstance(h, _StderrHandler) for h in pool_logger.handlers):
        pool_logger.addHandler(handler)
