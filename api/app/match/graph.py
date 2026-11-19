"""The job-match graph: several agents that share one set of notes (`state.MatchState`).

    START -> read -+-> (0 requirements) -> END            stream: {"error": "not_a_job"}
                   +-> search -> Send x N -> match -+-> check -> write -> END
                                  (one per requirement, in parallel)

- read (Claude Haiku 5.5, Gemini as fallback): reads the job ad, lists at most 12
  requirements, each with an English search phrase. Code numbers them r1..r12, trims them
  and sets `top8` when the ad had more (the name predates round 2, when the limit was 8).
- search (code): one batch retrieval for all requirements: one embedding call, one
  connection, project pages only (`PROJECT_PAGE_PREFIX`; not About, Privacy, Home).
- match (one agent per requirement, run in parallel with `Send`): sees only its own
  requirement and the sections found for it, writes reasoning first (kept in a local
  variable, never stored or streamed), then found / not found and which sections. Code turns
  section numbers into `page#section_id` links. Both providers failing or the stage running
  out of time makes that one requirement `unchecked`; the run goes on.
- check (Gemini 3.5 Flash-Lite, `match-v1`): runs once every match agent has written its
  finding (LangGraph waits for the whole parallel step). One call for every link of every
  `found` finding; a link it does not pass is removed, a finding left without links becomes
  `unchecked` (see `check`). It writes `checked_findings` and sends the notes once with
  them, so the panel shows what the guide will.
- write (Claude): sees only the checked findings, numbered, and writes one sentence per
  finding number, so its sentences can only mention sections the check kept; the links,
  "not covered" and the counts come from code. Nothing changes the links after it.

The models named above are the defaults: each step's own models, thinking level and output
cap come from `app.match.models` (`MatchDeps.steps`). read, match, check and write are
module-level functions (`read_step`, `match_one`, `check_step`, `write_step`); the graph's
nodes call them, and the eval replays call the same ones outside the graph.

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
from app.match import models as match_models
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
    PairVerdict,
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

# SPEC "职位对照" 时间: reading 25 s, matching 35 s (parallel), checking 25 s, writing 25 s.
DEFAULT_STAGE_TIMEOUTS: Mapping[Stage, float] = {
    "reading": 25.0,
    "matching": 35.0,
    "checking": 25.0,
    "writing": 25.0,
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

type StepOrders = Mapping[match_models.Step, Sequence[ChatProvider]]


@dataclass(frozen=True)
class MatchDeps:
    """What the steps need. `steps` gives each step its own providers in order
    (`app.match.models`); without it (or for a step it leaves out) the order is the one
    from before round 2: read, match, write on `claude` then `gemini`, check the other way
    round. `claude` and `gemini` stay for that and for older callers."""

    claude: ChatProvider
    gemini: ChatProvider
    search: BatchRetriever
    stage_timeouts: Mapping[Stage, float] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_TIMEOUTS)
    )
    call_timeout: float = CALL_TIMEOUT_S  # cap on any one model call inside a stage
    steps: StepOrders | None = None

    def order(self, step: match_models.Step) -> Sequence[ChatProvider]:
        if self.steps is not None and step in self.steps:
            return self.steps[step]
        if step == "check":
            return (self.gemini, self.claude)
        return (self.claude, self.gemini)


def build_match_deps(
    settings: Settings,
    db: Database,
    embedder: Embedder,
    stage_timeouts: Mapping[Stage, float] | None = None,
    models: match_models.MatchModels | None = None,
) -> MatchDeps:
    """Deps for the product (`models` None: `match_models.models_for(settings)`) or for an
    eval run with other models per step."""
    claude, gemini = build_providers(settings)
    steps = match_models.build_step_providers(models or match_models.models_for(settings), settings)
    return MatchDeps(
        claude,
        gemini,
        bind_search(db, embedder),
        stage_timeouts=dict(stage_timeouts or DEFAULT_STAGE_TIMEOUTS),
        steps=steps,
    )


# --- model calls ----------------------------------------------------------------------------


def _call_config(
    node: str, meter_callbacks: Sequence[Any], config: RunnableConfig | None
) -> RunnableConfig | None:
    """The config for one model call: the meter's usage recorder plus the caller's
    callbacks and metadata (an eval replay passes its `CallLog` here), with
    `langgraph_node` set so a call made outside the graph is labelled like one inside it.
    None when there is nothing to pass (the call then inherits the graph's config)."""
    if config is None:
        return RunnableConfig(callbacks=list(meter_callbacks)) if meter_callbacks else None
    out = RunnableConfig(**config)  # type: ignore[typeddict-item]
    callbacks = [*meter_callbacks, *(config.get("callbacks") or [])]
    if callbacks:
        out["callbacks"] = callbacks
    out["metadata"] = {"langgraph_node": node, **(config.get("metadata") or {})}
    return out


async def ask_json[T: BaseModel](
    node: str,
    order: Sequence[ChatProvider],
    schema: type[T],
    messages: list[BaseMessage],
    *,
    deadline: float,
    call_timeout: float = CALL_TIMEOUT_S,
    config: RunnableConfig | None = None,
) -> tuple[T, ChatProvider]:
    """Structured output (`method="json_schema"`) from the first provider in `order` that
    succeeds before `deadline` (a `time.monotonic()` value shared by the whole stage).

    Like `app.qa.models.structured`, except the explicit method and that a switch to the
    next provider gets only what is left of the stage. OpenAI gets `strict=True`
    (Structured Outputs); its refusals (`OpenAIRefusalError`), a reply cut off at the cap
    and a reply that does not fit the schema all count as that provider failing. The
    reservation uses the provider's own output cap (`StepProvider.max_output_tokens`,
    else `MAX_OUTPUT_TOKENS`). `config` (callbacks, metadata) is passed to the call, see
    `_call_config`. Provider budget errors propagate (`raise_if_budget_exhausted`); if
    every provider was refused by the spending guard (`AIPaused`) that propagates too;
    anything else ends in `ModelsUnavailable`.
    """
    if config is not None and not isinstance(config.get("callbacks") or [], list):
        # checked before any call: inside the loop it would look like every provider failing
        raise TypeError("ask_json: pass config callbacks as a list")
    errors: list[BaseException] = []
    for provider in order:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            errors.append(TimeoutError(f"{node}: stage deadline passed"))
            break
        try:
            # The time limit is outside the meter: topping up the spend reservation waits on
            # the database (a provider row lock another match may hold) and counts as part
            # of the call, so a slow database cannot hold a stage past its deadline.
            async with (
                asyncio.timeout(min(remaining, call_timeout)),
                metered(
                    provider.name,
                    provider.model,
                    input_tokens=estimate_input_tokens(messages),
                    max_output_tokens=getattr(provider, "max_output_tokens", MAX_OUTPUT_TOKENS),
                ) as meter,
            ):
                if provider.name == "openai":
                    chat = provider.chat.with_structured_output(
                        schema, method="json_schema", strict=True
                    )
                else:
                    chat = provider.chat.with_structured_output(schema, method="json_schema")
                call_config = _call_config(node, meter.callbacks, config)
                if call_config is not None:
                    raw = await chat.ainvoke(messages, config=call_config)
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
    "check": (
        frozenset({"requirements", "findings", "retrieved"}),
        frozenset({"checked_findings", "check_verdicts"}),
    ),
    "write": (
        frozenset({"requirements", "checked_findings", "retrieved", "language", "top8"}),
        frozenset({"result"}),
    ),
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


def _requirements(
    out: prompts.ReadOut, marker: str, limit: int = MAX_REQUIREMENTS
) -> tuple[list[Requirement], bool]:
    """Code-checked requirements (trimmed, de-duplicated, numbered r1..r<limit>) and `top8`
    (the ad had more than `limit`)."""
    seen: set[str] = set()
    cleaned: list[tuple[str, str]] = []
    for item in out.requirements:
        text = _one_line(item.text, MAX_REQUIREMENT_CHARS, marker)
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        query = _one_line(item.query, MAX_QUERY_CHARS, marker) or text
        cleaned.append((text, query))
    top8 = len(cleaned) > limit
    requirements = [
        Requirement(id=f"r{n}", text=text, query=query)
        for n, (text, query) in enumerate(cleaned[:limit], start=1)
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


# --- the four model steps (the graph's nodes and the eval replays share these) -------------


type Writer = Callable[[dict[str, Any]], None]


def _no_stream(_: dict[str, Any]) -> None:
    return None


def _deadline(deps: MatchDeps, stage: Stage) -> float:
    return time.monotonic() + deps.stage_timeouts[stage]


async def read_step(
    jd: str,
    deps: MatchDeps,
    *,
    writer: Writer | None = None,
    max_requirements: int = MAX_REQUIREMENTS,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Read the ad: `{"requirements", "language", "top8", "board"}` (the read node's
    update). The graph keeps `max_requirements` at `MAX_REQUIREMENTS` (12); replays may
    try fewer (8, the round-1 limit)."""
    send = writer or _no_stream
    send({"stage": "reading", "count": None})
    marker = prompts.new_marker()
    out, _ = await ask_json(
        "read",
        deps.order("read"),
        prompts.ReadOut,
        prompts.read_messages(jd, marker),
        deadline=_deadline(deps, "reading"),
        call_timeout=deps.call_timeout,
        config=config,
    )
    requirements, top8 = _requirements(out, marker, max_requirements)
    board = NotesBoard(requirements)
    if requirements:
        send({"notes": board.snapshot()})
    else:
        send({"error": "not_a_job"})
    return {
        "requirements": requirements,
        "language": _language(out.language),
        "top8": top8,
        "board": board,
    }


async def match_one(
    requirement: Requirement,
    sections: list[Section] | None,
    deps: MatchDeps,
    *,
    deadline: float | None = None,
    board: NotesBoard | None = None,
    writer: Writer | None = None,
    config: RunnableConfig | None = None,
) -> Finding:
    """One match agent: its requirement and the sections found for it (None: the search
    failed). `deadline` defaults to a fresh matching stage. With a `board` the finding is
    added and the notes are sent."""
    started = time.monotonic()
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
                deps.order("match"),
                prompts.MatchOut,
                prompts.match_messages(requirement, sections, prompts.new_marker()),
                deadline=deadline if deadline is not None else _deadline(deps, "matching"),
                call_timeout=deps.call_timeout,
                config=config,
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
    if board is not None:
        board.add(finding)
        (writer or _no_stream)({"notes": board.snapshot()})
    return finding


@dataclass(frozen=True)
class CheckResult:
    findings: list[Finding]  # the checked findings (`checked_findings`)
    verdicts: list[PairVerdict]  # one per pair asked about, in pair order


def _pair_verdicts(
    pairs: Sequence[match_check.Pair], verdicts: Sequence[prompts.Verdict] | None
) -> list[PairVerdict]:
    """What the check said per pair, read the way `apply_verdicts` reads it: any verdict
    `shows` for the pair's number passes it; a number with no verdict (or no reply at all,
    `verdicts` None) is `missing`."""
    given = {v.n for v in verdicts or []}
    shown = {v.n for v in verdicts or [] if v.shows}
    return [
        PairVerdict(
            requirement_id=requirement.id,
            url=section.url,
            shows=(n in shown) if n in given else "missing",
        )
        for n, requirement, section in pairs
    ]


async def check_step(
    requirements: Sequence[Requirement],
    findings: Sequence[Finding],
    retrieved: Mapping[str, list[Section] | None],
    deps: MatchDeps,
    *,
    writer: Writer | None = None,
    config: RunnableConfig | None = None,
) -> CheckResult:
    """The check (see `app.match.check`): one call for every link of every `found`
    finding. Nothing found: no call, no stage, no verdicts."""
    requirements = list(requirements)
    known = known_findings(requirements, findings)
    if not any(f.status == "found" for f in known):
        return CheckResult(known, [])
    send = writer or _no_stream
    send({"stage": "checking", "count": sum(1 for f in known if f.status == "found")})
    pairs = match_check.check_pairs(known, requirements, _sections_by_url(retrieved))
    verdicts: list[prompts.Verdict] | None = None
    if not pairs:
        checked = match_check.fail_all(known)
    else:
        try:
            out, _ = await ask_json(
                "check",
                deps.order("check"),
                prompts.CheckOut,
                prompts.check_messages(pairs, prompts.new_marker()),
                deadline=_deadline(deps, "checking"),
                call_timeout=deps.call_timeout,
                config=config,
            )
        except ModelsUnavailable:
            checked = match_check.fail_all(known)
        else:
            verdicts = list(out.verdicts)
            checked = match_check.apply_verdicts(known, pairs, verdicts)
    # The panel shows the notes; send them once with the check's changes.
    send({"notes": notes_snapshot(requirements, checked)})
    return CheckResult(checked, _pair_verdicts(pairs, verdicts))


async def write_step(
    requirements: Sequence[Requirement],
    checked_findings: Sequence[Finding],
    retrieved: Mapping[str, list[Section] | None],
    language: str,
    top8: bool,
    deps: MatchDeps,
    *,
    writer: Writer | None = None,
    config: RunnableConfig | None = None,
) -> Guide:
    """The guide from the checked findings only: the write agent never sees a link the
    check rejected; links, "not covered" and the counts come from code."""
    (writer or _no_stream)({"stage": "writing", "count": None})
    findings = {f.requirement_id: f for f in known_findings(requirements, checked_findings)}
    by_url = _sections_by_url(retrieved)
    found = [
        (r, findings[r.id])
        for r in requirements
        if r.id in findings and findings[r.id].status == "found"
    ]
    not_covered = [
        r.text for r in requirements if r.id in findings and findings[r.id].status == "not_found"
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
            deps.order("write"),
            prompts.WriteOut,
            prompts.write_messages(numbered, language, marker),
            deadline=_deadline(deps, "writing"),
            call_timeout=deps.call_timeout,
            config=config,
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
    return Guide(
        items=items, not_covered=not_covered, unchecked=unchecked, top8=top8, language=language
    )


# --- the graph ------------------------------------------------------------------------------


def build_match_graph(deps: MatchDeps) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the graph. Input: `{"jd": str}`. No checkpointer: nothing is stored."""

    async def read(view: dict[str, Any]) -> dict[str, Any]:
        return await read_step(view["jd"], deps, writer=get_stream_writer())

    async def search(view: dict[str, Any]) -> dict[str, Any]:
        requirements: list[Requirement] = view["requirements"]
        get_stream_writer()({"stage": "matching", "count": len(requirements)})
        match_deadline = _deadline(deps, "matching")
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
        finding = await match_one(
            task["requirement"],
            task["sections"],
            deps,
            deadline=task["deadline"],
            board=task["board"],
            writer=get_stream_writer(),
        )
        return {"findings": [finding]}

    async def check(view: dict[str, Any]) -> dict[str, Any]:
        result = await check_step(
            view["requirements"],
            view.get("findings") or [],
            view.get("retrieved") or {},
            deps,
            writer=get_stream_writer(),
        )
        # the evals store the verdicts; an empty list when the check asked about nothing
        return {"checked_findings": result.findings, "check_verdicts": result.verdicts}

    async def write(view: dict[str, Any]) -> dict[str, Any]:
        guide = await write_step(
            view["requirements"],
            view.get("checked_findings") or [],
            view.get("retrieved") or {},
            view["language"],
            view["top8"],
            deps,
            writer=get_stream_writer(),
        )
        return {"result": guide}

    def after_read(state: MatchState) -> str:
        return "search" if state["requirements"] else END

    builder = StateGraph(MatchState, input_schema=MatchInput)
    builder.add_node("read", cast(Any, guard_node("read", read)))
    builder.add_node("search", cast(Any, guard_node("search", search)))
    builder.add_node("match", cast(Any, guard_node("match", match)))
    builder.add_node("check", cast(Any, guard_node("check", check)))
    builder.add_node("write", cast(Any, guard_node("write", write)))
    builder.add_edge(START, "read")
    builder.add_conditional_edges("read", after_read, ["search", END])
    builder.add_conditional_edges("search", fan_out, ["match"])
    builder.add_edge("match", "check")
    builder.add_edge("check", "write")
    builder.add_edge("write", END)
    return builder.compile()
