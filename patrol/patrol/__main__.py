"""`uv run python -m patrol --base-url https://<domain>`: exit 0 green, 1 red, 2 not run.

Needs PATROL_SECRET (signs X-Patrol) and PATROL_GEMINI_API_KEY (the agent's own model, the dev
Gemini project). Neither, nor the session token, is ever printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from patrol.checks import base_origin
from patrol.run import format_report, run_patrol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m patrol", description=__doc__)
    parser.add_argument("--base-url", required=True, help="the site, e.g. https://example.com")
    args = parser.parse_args(argv)
    try:
        base = base_origin(args.base_url)
    except ValueError as e:
        print(f"patrol: {e}", file=sys.stderr)
        return 2
    secret = os.environ.get("PATROL_SECRET", "").strip()
    gemini_key = os.environ.get("PATROL_GEMINI_API_KEY", "").strip()
    missing = [
        name
        for name, value in (("PATROL_SECRET", secret), ("PATROL_GEMINI_API_KEY", gemini_key))
        if not value
    ]
    if missing:
        print(f"patrol: missing {', '.join(missing)}", file=sys.stderr)
        return 2
    report = asyncio.run(run_patrol(base, secret, gemini_key))
    print(format_report(report), flush=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
