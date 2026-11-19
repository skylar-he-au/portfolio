"""The job-match graph: several agents that share one set of notes (`state.MatchState`).

    START -> read -+-> (0 requirements) -> END            stream: {"error": "not_a_job"}
                   +-> search -> Send x N -> match -+-> write -> check -> END
                                  (one per requirement, in parallel)

- read (Claude Haiku 5.5, Gemini as fallback): reads the job ad, lists at most 8
  requirements, each with an English search phrase. Code numbers them r1..r8, trims them
  and sets `top8` when the ad had more.
- search (code): one batch retrieval for all requirements: one embedding call, one
  connection, project pages only (`PROJECT_PAGE_PREFIX`; not About, Privacy, Home).
- match (one agent per requirement, run in parallel with `Send`): sees only its own
  requirement and the sections found for it, writes reasoning first (kept in a local
  variable, never stored or streamed), then found / not found and which sections. Code turns
  section numbers into `page#section_id` links. Both providers failing or the stage running
  out of time makes that one requirement `unchecked`; the run goes on.
- write (Claude): runs once every match agent has written its finding (LangGraph waits for
  the whole parallel step). It sees the numbered findings and writes one sentence per
  finding number; the links, "not covered" and the counts come from code.
- check (Gemini 3.5 Flash-Lite, `match-v1`): one call for every link of the guide; a link
  it does not pass is removed, an item left without links becomes `unchecked` (see `check`).
  Then the notes are sent once more with those changes (`check.checked_findings`), so the
  panel and the guide agree.

Who reads and writes which note is fixed in `NOTE_ACCESS` and enforced by `guard_node`.
Every model call uses structured output with `method="json_schema"`. Each stage has its own
deadline (`MatchDeps.stage_timeouts`); switching provider uses what is left of it.
The stream (LangGraph `custom` mode) carries only `{"stage", "count"}`, `{"notes": ...}`
(validated fields only, see `state.notes_snapshot`) and `{"error": "not_a_job"}`.
`ModelsUnavailable` from read or write, `AIPaused` and `ProviderBudgetExhausted` propagate
to the caller (6b2 maps them to SSE errors).
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from pydantic import BaseModel

from app.config import Settings
from app.db import Database
from app.guard.ledger import AIPaused, ProviderBudgetExhausted
from app.guard.usage import estimate_input_tokens, metered, raise_if_budget_exhausted
from app.match import check as match_check
from app.match import prompts
from app.match.state import (
    MAX_REQUIREMENT_CHARS,
    MAX_REQUIREMENTS,
    Finding,
    Guide,
    GuideItem,
    MatchInput,
    MatchState,
    MatchTask,
    NotesBoard,
    Requirement,
    SectionRef,
    Stage,
    known_findings,
    notes_snapshot,
)
from app.qa.models import CALL_TIMEOUT_S, MAX_OUTPUT_TOKENS, ChatProvider, build_providers
from app.rag.retrieve import SEARCH_SQL
from app.rag.types import Embedder, ModelsUnavailable, Section

logger = logging.getLogger(__name__)

PROJECT_PAGE_PREFIX = "/projects/"
MAX_SECTIONS = 6  # retrieved per requirement, as in the Q&A graph
MAX_LINKS = 3  # sections a finding may link to
MAX_QUERY_CHARS = 200
MAX_GUIDE_TEXT_CHARS = 300

# SPEC "职位对照" 时间: reading 25 s, matching 35 s (parallel), writing 25 s, checking 25 s.
DEFAULT_STAGE_TIMEOUTS: Mapping[Stage, float] = {
    "reading": 25.0,
    "matching": 35.0,
    "writing": 25.0,
    "checking": 25.0,
}


# --- retrieval: one embedding call and one connection for every requirement -----------------

PROJECT_PAGES_SQL = """
SELECT DISTINCT s.page
FROM sections s
JOIN content_state c ON c.current_version = s.version
WHERE c.id AND starts_with(s.page, %(prefix)s)
ORDER BY s.page
"""


class BatchRetriever(Protocol):
    """Retrieval as the match graph uses it: db and embedder bound; one result list per query,
    in order."""

    async def __call__(
        self, queries: Sequence[str], *, page_prefix: str
    ) -> list[list[Section]]: ...


async def retrieve_many(
    db: Database,
    embedder: Embedder,
    queries: Sequence[str],
    *,
    page_prefix: str,
    k: int = MAX_SECTIONS,
) -> list[list[Section]]:
    """`app.rag.retrieve.retrieve` for several queries: the same SQL (`SEARCH_SQL`) per
    query, all vectors from one `embed` call, all queries on one connection, limited to the
    current pages whose path starts with `page_prefix`."""
    if not queries:
        return []
    vectors = await embedder.embed(list(queries))
    results: list[list[Section]] = []
    async with db.connection() as conn:
        await register_vector_async(conn)
        cursor = await conn.execute(PROJECT_PAGES_SQL, {"prefix": page_prefix})
        pages = [row[0] for row in await cursor.fetchall()]
        for query, vector in zip(queries, vectors, strict=True):
            if not query.strip():
                results.append([])
                continue
            cursor = await conn.execute(
                SEARCH_SQL,
                {"embedding": Vector(vector), "query": query, "pages": pages, "limit": k},
            )
            rows = await cursor.fetchall()
            results.append(
                [
                    Section(page, section_id, heading, text, float(score))
                    for page, section_id, heading, text, score in rows
                ]
            )
    return results


def bind_search(db: Database, embedder: Embedder) -> BatchRetriever:
    async def search(queries: Sequence[str], *, page_prefix: str) -> list[list[Section]]:
        return await retrieve_many(db, embedder, queries, page_prefix=page_prefix)

    return search


# --- dependencies ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchDeps:
    claude: ChatProvider
    gemini: ChatProvider
    search: BatchRetriever
    stage_timeouts: Mapping[Stage, float] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_TIMEOUTS)
    )
    call_timeout: float = CALL_TIMEOUT_S  # cap on any one model call inside a stage


def build_match_deps(
    settings: Settings,
    db: Database,
    embedder: Embedder,
    stage_timeouts: Mapping[Stage, float] | None = None,
) -> MatchDeps:
    claude, gemini = build_providers(settings)
    return MatchDeps(
        claude,
        gemini,
        bind_search(db, embedder),
        stage_timeouts=dict(stage_timeouts or DEFAULT_STAGE_TIMEOUTS),
    )


# --- model calls ----------------------------------------------------------------------------


async def ask_json[T: BaseModel](
    node: str,
    order: Sequence[ChatProvider],
    schema: type[T],
    messages: list[BaseMessage],
    *,
    deadline: float,
    call_timeout: float = CALL_TIMEOUT_S,
) -> tuple[T, ChatProvider]:
    """Structured output (`method="json_schema"`) from the first provider in `order` that
    succeeds before `deadline` (a `time.monotonic()` value shared by the whole stage).

    Like `app.qa.models.structured`, except the explicit method and that a switch to the
    next provider gets only what is left of the stage. Provider budget errors propagate
    (`raise_if_budget_exhausted`); if every provider was refused by the spending guard
    (`AIPaused`) that propagates too; anything else ends in `ModelsUnavailable`.
    """
    errors: list[BaseException] = []
    for provider in order:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            errors.append(TimeoutError(f"{node}: stage deadline passed"))
            break
        try:
            async with metered(
                provider.name,
                provider.model,
                input_tokens=estimate_input_tokens(messages),
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ) as meter:
                async with asyncio.timeout(min(remaining, call_timeout)):
                    chat = provider.chat.with_structured_output(schema, method="json_schema")
                    if meter.callbacks:
                        raw = await chat.ainvoke(
                            messages, config=RunnableConfig(callbacks=list(meter.callbacks))
                        )
                    else:
                        raw = await chat.ainvoke(messages)
            if raw is None:
                raise ValueError("empty reply")
            value = raw if isinstance(raw, schema) else schema.model_validate(raw)
            return value, provider
        except Exception as error:  # noqa: BLE001 - any provider failure means: try the next one
            await raise_if_budget_exhausted(provider.name, error)
            logger.warning(
                "match model call failed: node=%s provider=%s model=%s error=%s",
                node,
                provider.name,
                provider.model,
                type(error).__name__,
            )
            errors.append(error)
    if errors and all(isinstance(e, AIPaused) for e in errors):
        raise errors[-1]
    raise ModelsUnavailable(f"{node}: every provider failed") from (errors[-1] if errors else None)


# --- who reads and writes which note --------------------------------------------------------


class NoteAccessError(Exception):
    """A node tried to write a note it does not own."""


# node -> (fields it reads, fields it writes). `match` reads its `Send` payload (`MatchTask`)
# instead of the state. Writes outside the set raise; reads outside it are not passed in.
NOTE_ACCESS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "read": (frozenset({"jd"}), frozenset({"requirements", "language", "top8", "board"})),
    "search": (frozenset({"requirements"}), frozenset({"retrieved", "match_deadline"})),
    "match": (frozenset({"requirement", "sections", "deadline", "board"}), frozenset({"findings"})),
    "write": (
        frozenset({"requirements", "findings", "retrieved", "language", "top8"}),
        frozenset({"result"}),
    ),
    "check": (frozenset({"result", "requirements", "retrieved", "board"}), frozenset({"result"})),
}

type NodeFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def guard_node(name: str, fn: NodeFn) -> NodeFn:
    """`fn` as graph node `name`: it gets only the fields it may read, and may return only
    the fields it may write (else `NoteAccessError`)."""
    reads, writes = NOTE_ACCESS[name]

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        view = {key: value for key, value in state.items() if key in reads}
        update = await fn(view)
        extra = set(update) - writes
        if extra:
            raise NoteAccessError(f"{name} may not write {sorted(extra)}")
        return update

    node.__name__ = name
    return node


# --- small helpers --------------------------------------------------------------------------


def _language(code: str) -> str:
    code = code.strip().lower().replace("_", "-")
    head = code.split("-")[0]
    if 2 <= len(head) <= 3 and head.isascii() and head.isalpha():
        return head
    return "en"


def _one_line(text: str, limit: int, marker: str = "") -> str:
    if marker:
        text = text.replace(marker, "")
    return " ".join(text.split())[:limit].strip()


def _requirements(out: prompts.ReadOut, marker: str) -> tuple[list[Requirement], bool]:
    """Code-checked requirements (trimmed, de-duplicated, numbered r1..r8) and `top8`."""
    seen: set[str] = set()
    cleaned: list[tuple[str, str]] = []
    for item in out.requirements:
        text = _one_line(item.text, MAX_REQUIREMENT_CHARS, marker)
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        query = _one_line(item.query, MAX_QUERY_CHARS, marker) or text
        cleaned.append((text, query))
    top8 = len(cleaned) > MAX_REQUIREMENTS
    requirements = [
        Requirement(id=f"r{n}", text=text, query=query)
        for n, (text, query) in enumerate(cleaned[:MAX_REQUIREMENTS], start=1)
    ]
    return requirements, top8


def _links(numbers: Sequence[int], sections: Sequence[Section]) -> list[SectionRef]:
    """The model's section numbers as links: only 1..len(sections), no repeats, at most 3."""
    links: list[SectionRef] = []
    for n in dict.fromkeys(numbers):
        if 1 <= n <= len(sections):
            section = sections[n - 1]
            links.append(SectionRef(url=section.url, title=section.heading))
        if len(links) == MAX_LINKS:
            break
    return links


def _sections_by_url(retrieved: Mapping[str, list[Section] | None]) -> dict[str, Section]:
    return {s.url: s for found in retrieved.values() if found for s in found}


def _ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


# --- the graph ------------------------------------------------------------------------------


def build_match_graph(deps: MatchDeps) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the graph. Input: `{"jd": str}`. No checkpointer: nothing is stored."""
    answer_order = (deps.claude, deps.gemini)
    check_order = (deps.gemini, deps.claude)
    timeouts = deps.stage_timeouts

    def deadline(stage: Stage) -> float:
        return time.monotonic() + timeouts[stage]

    async def read(view: dict[str, Any]) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"stage": "reading", "count": None})
        marker = prompts.new_marker()
        out, _ = await ask_json(
            "read",
            answer_order,
            prompts.ReadOut,
            prompts.read_messages(view["jd"], marker),
            deadline=deadline("reading"),
            call_timeout=deps.call_timeout,
        )
        requirements, top8 = _requirements(out, marker)
        board = NotesBoard(requirements)
        if requirements:
            writer({"notes": board.snapshot()})
        else:
            writer({"error": "not_a_job"})
        return {
            "requirements": requirements,
            "language": _language(out.language),
            "top8": top8,
            "board": board,
        }

    async def search(view: dict[str, Any]) -> dict[str, Any]:
        requirements: list[Requirement] = view["requirements"]
        get_stream_writer()({"stage": "matching", "count": len(requirements)})
        match_deadline = deadline("matching")
        retrieved: dict[str, list[Section] | None]
        try:
            async with asyncio.timeout(max(match_deadline - time.monotonic(), 0)):
                found = await deps.search(
                    [r.query for r in requirements], page_prefix=PROJECT_PAGE_PREFIX
                )
            if len(found) != len(requirements):
                raise ValueError("one result list per query expected")
            retrieved = {
                r.id: list(sections)[:MAX_SECTIONS]
                for r, sections in zip(requirements, found, strict=True)
            }
        except (AIPaused, ProviderBudgetExhausted):
            raise
        except Exception as error:  # noqa: BLE001 - no search: every requirement is unchecked
            logger.warning("match search failed: error=%s", type(error).__name__)
            retrieved = {r.id: None for r in requirements}
        return {"retrieved": retrieved, "match_deadline": match_deadline}

    def fan_out(state: MatchState) -> list[Send]:
        retrieved = state.get("retrieved") or {}
        return [
            Send(
                "match",
                MatchTask(
                    requirement=r,
                    sections=retrieved.get(r.id),
                    deadline=state["match_deadline"],
                    board=state["board"],
                ),
            )
            for r in state["requirements"]
        ]

    async def match(task: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        requirement: Requirement = task["requirement"]
        sections: list[Section] | None = task["sections"]
        provider_name: str | None = None
        links: list[SectionRef] = []
        status: Literal["found", "not_found", "unchecked"]
        if sections is None:
            status = "unchecked"
        elif not sections:
            status = "not_found"  # nothing on the project pages to look at: no model call
        else:
            try:
                out, provider = await ask_json(
                    "match",
                    answer_order,
                    prompts.MatchOut,
                    prompts.match_messages(requirement, sections, prompts.new_marker()),
                    deadline=task["deadline"],
                    call_timeout=deps.call_timeout,
                )
            except ModelsUnavailable:
                status = "unchecked"
            else:
                provider_name = provider.name
                # out.reasoning is dropped here: it is never stored, streamed or logged.
                links = _links(out.sections, sections) if out.found else []
                if not out.found:
                    status = "not_found"
                elif links:
                    status = "found"
                else:
                    status = "unchecked"  # "found" without a valid link: nothing to show
        finding = Finding(
            requirement_id=requirement.id,
            status=status,
            sections=links,
            provider=provider_name,
            ms=_ms(started),
        )
        board: NotesBoard = task["board"]
        board.add(finding)
        get_stream_writer()({"notes": board.snapshot()})
        return {"findings": [finding]}

    async def write(view: dict[str, Any]) -> dict[str, Any]:
        get_stream_writer()({"stage": "writing", "count": None})
        requirements: list[Requirement] = view["requirements"]
        findings = {
            f.requirement_id: f for f in known_findings(requirements, view.get("findings") or [])
        }
        by_url = _sections_by_url(view.get("retrieved") or {})
        found = [
            (r, findings[r.id])
            for r in requirements
            if r.id in findings and findings[r.id].status == "found"
        ]
        not_covered = [
            r.text
            for r in requirements
            if r.id in findings and findings[r.id].status == "not_found"
        ]
        unchecked = sum(
            1 for r in requirements if r.id not in findings or findings[r.id].status == "unchecked"
        )
        texts: dict[int, str] = {}
        if found:
            marker = prompts.new_marker()
            numbered = [
                (n, r, [by_url[s.url] for s in f.sections if s.url in by_url])
                for n, (r, f) in enumerate(found, start=1)
            ]
            out, _ = await ask_json(
                "write",
                answer_order,
                prompts.WriteOut,
                prompts.write_messages(numbered, view["language"], marker),
                deadline=deadline("writing"),
                call_timeout=deps.call_timeout,
            )
            for item in out.items:
                text = _one_line(item.text, MAX_GUIDE_TEXT_CHARS, marker)
                if 1 <= item.finding <= len(found) and item.finding not in texts and text:
                    texts[item.finding] = text
                # any other finding number is dropped: links come only from match findings
        items = [
            GuideItem(
                requirement_id=r.id,
                text=texts.get(n) or "; ".join(s.title for s in f.sections),
                sections=f.sections,
            )
            for n, (r, f) in enumerate(found, start=1)
        ]
        guide = Guide(
            items=items,
            not_covered=not_covered,
            unchecked=unchecked,
            top8=view["top8"],
            language=view["language"],
        )
        return {"result": guide}

    async def check(view: dict[str, Any]) -> dict[str, Any]:
        guide: Guide | None = view.get("result")
        if guide is None or not guide.items:
            return {}
        get_stream_writer()({"stage": "checking", "count": len(guide.items)})
        pairs = match_check.check_pairs(
            guide, view["requirements"], _sections_by_url(view.get("retrieved") or {})
        )
        checked = await run_check(guide, pairs)
        # The panel shows the notes; send them once more so they agree with the checked guide.
        board: NotesBoard = view["board"]
        get_stream_writer()(
            {
                "notes": notes_snapshot(
                    board.requirements, match_check.checked_findings(board.findings, checked)
                )
            }
        )
        return {"result": checked}

    async def run_check(guide: Guide, pairs: list[tuple[int, Requirement, Section]]) -> Guide:
        if not pairs:
            return match_check.fail_all(guide)
        try:
            out, _ = await ask_json(
                "check",
                check_order,
                prompts.CheckOut,
                prompts.check_messages(pairs, prompts.new_marker()),
                deadline=deadline("checking"),
                call_timeout=deps.call_timeout,
            )
        except ModelsUnavailable:
            return match_check.fail_all(guide)
        return match_check.apply_verdicts(guide, pairs, out.verdicts)

    def after_read(state: MatchState) -> str:
        return "search" if state["requirements"] else END

    builder = StateGraph(MatchState, input_schema=MatchInput)
    builder.add_node("read", cast(Any, guard_node("read", read)))
    builder.add_node("search", cast(Any, guard_node("search", search)))
    builder.add_node("match", cast(Any, guard_node("match", match)))
    builder.add_node("write", cast(Any, guard_node("write", write)))
    builder.add_node("check", cast(Any, guard_node("check", check)))
    builder.add_edge(START, "read")
    builder.add_conditional_edges("read", after_read, ["search", END])
    builder.add_conditional_edges("search", fan_out, ["match"])
    builder.add_edge("match", "write")
    builder.add_edge("write", "check")
    builder.add_edge("check", END)
    return builder.compile()
