# portfolio

Skylar He's personal site: write-ups of my projects, plus an assistant that answers recruiters' questions about them with links to the page sections it used. Right now this is the skeleton: the pages render and the backend and database run locally. The assistant comes next.

## Repository layout

```
.
├── content/             Page text in Markdown (one file per page)
├── web/                 Frontend: Astro 7, React 19, TypeScript (static output)
├── api/                 Backend: Python 3.12, FastAPI, psycopg 3, managed with uv
│   ├── app/             Application code (app.main:app, app.migrate)
│   ├── migrations/      Plain SQL migrations, applied in order (NNNN_name.sql)
│   └── tests/
├── docker-compose.yml   Local stack: db -> migrate -> api -> web
└── .github/workflows/   CI for api (ruff, pytest) and web (check, tests, build)
```

## Run it locally

Requires Docker with Compose v2.

```bash
cp .env.example .env
./scripts/dev-up.sh
```

The script starts the stack in the background with a temporary Docker config to avoid Docker Desktop credential helper hangs. Then open http://localhost:4321.

If Docker works normally on your machine, `docker compose up` is also an option.

| Service | What it does | Address |
|---|---|---|
| `db` | Postgres 17 with pgvector 0.8.0 | `localhost:5433` |
| `migrate` | Applies `api/migrations/*.sql`, then exits | — |
| `api` | FastAPI backend | http://localhost:8000/api/ping |
| `web` | Astro dev server, reloads when `web/` or `content/` changes | http://localhost:4321 |

All ports are bound to `127.0.0.1` only. If one is already taken on your machine, set `DB_PORT`, `API_PORT` or `WEB_PORT` (in `.env` or the shell) before `docker compose up`.

With `APP_ENV=dev`, `GET /api/health` also reports whether the database is reachable and which pgvector version is installed. Outside dev that route does not exist.

Stop with `./scripts/dev-up.sh --down`; this uses the same workaround and keeps the local database volume. Direct `docker compose down` can also hit the credential helper. `docker compose down -v` stops everything and deletes the local database volume.

### Screenshots

With the frontend dependencies and Chrome installed, run from the repo root:

```bash
node web/scripts/shot.mjs http://localhost:4321 --width 375 --height 900 --scheme dark --out screenshots/mobile-dark.png
```

Use `--full` to capture the full page. Output supports `.png`, `.jpg` and `.jpeg`. Set `SHOT_BROWSER_PATH` to use a specific browser executable instead of installed Chrome.

## Load the content for Ask

After `docker compose up` starts the database and applies migrations, run on the host:

```bash
cd api
uv run python -m app.ingest
```

Set `OPENAI_API_KEY` in the repo-root `.env`; the command reads `DATABASE_URL` from
the same file (the compose `localhost:5433` address) and loads `content/` pages with
`rag: true`. Re-running creates a new version and removes the old one. Run this on
the host because the API container image has no `content/` directory.

If you changed `DB_PORT` (for example to 5434 because 5433 is taken), set `DATABASE_URL` to point at that port when running the ingest.

## Run the tests

Backend (needs [uv](https://docs.astral.sh/uv/); the database tests use the compose `db` and create and drop their own temporary databases):

```bash
cd api
uv sync --locked
uv run ruff check . && uv run ruff format --check .
TEST_DATABASE_URL=postgresql://portfolio:portfolio@localhost:5433/portfolio uv run pytest
```

Without `TEST_DATABASE_URL` the database tests are skipped.

Frontend (needs Node 24):

```bash
cd web
npm ci
npm run check
npm test
npm run build && npm run test:dist
```

## Evals

From `api/`, the offline eval runner sends a YAML question set through the real Q&A graph,
applies code checks, and optionally scores answers with RAGAS. First run
`docker compose up -d db migrate` from the repo root, ingest the content, and configure
`.env` with the three dev keys (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`)
and `DATABASE_URL`.

```bash
uv run --no-default-groups --group eval python -m evals.run --set <qa-set.yaml> --out <private-results-dir> \
  --ledger <ledger.jsonl> --pass-cap <usd> --total-cap <usd> \
  --credit-pass-cap <usd> --credit-total-cap <usd>
```

All four caps are required, there are no defaults; the numbers live in the project docs,
not here. Copy them from the docs repo's command. The `eval` group is installed on demand;
CI does not install it.

Options: `--split tune|held_out|all` (default `tune`), `--only id1,id2`, `--repeats N`
(default `1`), `--route default|gemini-answer` (default `default`), and `--no-ragas`.
Raw JSONL results and a metadata sidecar go to `--out` and must stay private. Add
`--summary-out ../content/evals` only for baseline/final runs to publish aggregates without
question or answer text. An incomplete run keeps its raw results and writes no summary.
Exit codes: `0` complete, `1` incomplete, `2` invalid arguments/set or missing eval modules.

## License

The code is under the [MIT License](LICENSE). The text and images in `content/` are © Skylar He, all rights reserved, and are not covered by the MIT License (see [content/LICENSE](content/LICENSE)).
