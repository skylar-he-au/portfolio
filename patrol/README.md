# patrol

Post-deploy browser patrol: a browser-use agent walks the live site like a recruiter (ask an
example question, open a cited project page, match a fixed job ad). Code, not the agent,
decides green or red.

The home page has one box, `section#ask`: an example button sends its question; the same box
turns into "Match job" when the pasted text is longer than 500 characters (the fixed job ad is
633, `tests/test_page_contract.py` checks it). The patrol reads `#ask .ask-turn:not([data-restored])
.ask-answer` (turns read back from `sessionStorage` are not counted) and `#ask .match-result`;
done means the 👎 button is there. If the page's structure changes, change `SNAPSHOT_JS` in
`patrol/run.py` and the steps in `patrol/task.py` together.

```bash
uv sync --locked
uv run playwright install chromium
PATROL_SECRET=... PATROL_GEMINI_API_KEY=... uv run python -m patrol --base-url http://localhost:4321
```

Exit 0 green, 1 red (the output names the step), 2 not run (bad arguments or missing variables).
Unit tests: `uv run pytest` (no model, no browser).

Against a local site (a few cents per run: the agent and the site's own answers use the dev keys):
the site must be the current `main` (`site/scripts/dev-up.sh` rebuilds web), and `PATROL_SECRET`
must be the same value the api has in `site/.env`.
