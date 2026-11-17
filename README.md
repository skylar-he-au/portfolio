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
docker compose up
```

Then open http://localhost:4321.

| Service | What it does | Address |
|---|---|---|
| `db` | Postgres 17 with pgvector 0.8.0 | `localhost:5433` |
| `migrate` | Applies `api/migrations/*.sql`, then exits | — |
| `api` | FastAPI backend | http://localhost:8000/api/ping |
| `web` | Astro dev server, reloads when `web/` or `content/` changes | http://localhost:4321 |

All ports are bound to `127.0.0.1` only. If one is already taken on your machine, set `DB_PORT`, `API_PORT` or `WEB_PORT` (in `.env` or the shell) before `docker compose up`.

With `APP_ENV=dev`, `GET /api/health` also reports whether the database is reachable and which pgvector version is installed. Outside dev that route does not exist.

`docker compose down -v` stops everything and deletes the local database volume.

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

## License

The code is under the [MIT License](LICENSE). The text and images in `content/` are © Skylar He, all rights reserved, and are not covered by the MIT License (see [content/LICENSE](content/LICENSE)).
