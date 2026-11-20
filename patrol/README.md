# patrol

Post-deploy browser patrol: a browser-use agent walks the live site like a recruiter (ask an
example question, open a cited project page, match a fixed job ad). Code, not the agent,
decides green or red.

```bash
uv sync --locked
uv run playwright install chromium
PATROL_SECRET=... PATROL_GEMINI_API_KEY=... uv run python -m patrol --base-url http://localhost:4321
```

Exit 0 green, 1 red (the output names the step), 2 not run (bad arguments or missing variables).
Unit tests: `uv run pytest` (no model, no browser).
