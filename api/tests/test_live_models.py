# ruff: noqa: RUF001 - Chinese test strings use full-width punctuation
"""Real model calls with the dev keys: Haiku 5.5 takes our parameters, structured output
works on both providers, and nothing silently falls back.

Opt-in only, so a plain `uv run pytest` never spends money even with keys in `.env`:

    uv run pytest -m live -s tests/test_live_models.py

Retrieval is a small keyword ranker over the real content files (3a's real retrieval and
embeddings are not used here). Prints each answer for the report.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.config import get_settings
from app.qa import output
from app.qa.graph import build_deps, build_graph
from app.rag.types import Section

pytestmark = [pytest.mark.live, pytest.mark.anyio]

CONTENT = Path(__file__).resolve().parents[2] / "content"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _live_only(request: pytest.FixtureRequest) -> None:
    if "live" not in (request.config.option.markexpr or ""):
        pytest.skip("live model test: run with `-m live`")
    settings = get_settings()
    if not (settings.anthropic_api_key and settings.gemini_api_key):
        pytest.skip("live model test: ANTHROPIC_API_KEY / GEMINI_API_KEY not set")


def _slug(heading: str) -> str:
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading.lower()).strip())


def _sections(path: Path, page: str) -> list[Section]:
    body = path.read_text().split("---", 2)[2]
    out = []
    for chunk in re.split(r"^## ", body, flags=re.M)[1:]:
        heading, _, text = chunk.partition("\n")
        out.append(Section(page, _slug(heading.strip()), heading.strip(), text.strip()))
    return out


ALL = (
    _sections(CONTENT / "about.md", "/about")
    + _sections(CONTENT / "projects" / "roleradar.md", "/projects/roleradar")
    + _sections(CONTENT / "projects" / "oncallops.md", "/projects/oncallops")
)


class KeywordRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str] | None]] = []

    async def __call__(self, query: str, *, pages: Sequence[str] | None = None) -> list[Section]:
        self.calls.append((query, None if pages is None else list(pages)))
        words = set(re.findall(r"[a-z]{3,}", query.lower()))
        pool = [s for s in ALL if pages is None or s.page in pages]

        def score(s: Section) -> int:
            return len(words & set(re.findall(r"[a-z]{3,}", (s.heading + " " + s.text).lower())))

        ranked = sorted(pool, key=score, reverse=True)[:6]
        return [Section(s.page, s.section_id, s.heading, s.text, float(score(s))) for s in ranked]


async def _ask(question: str) -> tuple[list[tuple[str, dict]], dict, KeywordRetriever]:
    retriever = KeywordRetriever()
    graph = build_graph(build_deps(get_settings(), retriever), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "live"}}
    stream = graph.astream(
        {"question": question, "history": []},
        config,
        stream_mode=["messages", "custom"],
        version="v2",
    )
    events = [e async for e in output.events(stream)]
    state = (await graph.aget_state(config)).values
    print(f"\n=== {question}")
    print("route:", json.dumps(state.get("private")), state.get("language"))
    print("retrieved with:", retriever.calls)
    for name, data in events:
        if name in ("route", "reset", "answer", "check"):
            print(f"{name}: {json.dumps(data, ensure_ascii=False)}")
    print("private_check:", state.get("private_check"))
    print("nodes:", [(n["name"], n["provider"], n["ms"]) for n in state["trace"]["nodes"]])
    return events, state, retriever


def _providers(state: dict) -> dict[str, str]:
    return {n["name"]: n["provider"] for n in state["trace"]["nodes"]}


async def test_live_example_question():
    events, state, _ = await _ask("How does RoleRadar decide which jobs match a user?")
    assert [n for n, _ in events if n == "token"], "general answers stream"
    assert state["check"] is not None
    # No silent fallback: Claude rewrote and answered, Gemini checked.
    assert _providers(state) == {
        "rewrite": "anthropic",
        "retrieve": "openai",
        "answer": "anthropic",
        "check": "google",
    }
    assert any(c["url"].startswith("/projects/roleradar#") for c in state["citations"])


async def test_live_private_question():
    events, state, retriever = await _ask("Can you work full-time in Australia?")
    assert state["private"] is True
    assert retriever.calls[0][1] == ["/about"]
    assert [n for n, _ in events if n == "token"] == []
    assert _providers(state)["answer"] == "anthropic"
    assert _providers(state)["check"] == "google"


async def test_live_chinese_question():
    _events, state, retriever = await _ask("RoleRadar 怎么判断哪些职位适合用户？")
    assert state["language"] == "zh"
    assert retriever.calls[0][0].isascii(), "the search query is English"
    assert re.search(r"[一-鿿]", state["answer_text"]), "the answer is Chinese"
    assert _providers(state)["answer"] == "anthropic"
