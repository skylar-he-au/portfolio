# ruff: noqa: RUF001 - Chinese test strings use full-width punctuation
"""The compiled Q&A graph with fake streaming models and a fake retriever.

Every test runs the real compiled graph through `graph.astream(..., stream_mode=["messages",
"custom"], version="v2")` and `output.events`, the same way the API does (contract item 6),
then asserts the visitor-facing events and the final state.
"""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from app.qa import output
from app.qa.graph import NOT_FOUND, PRIVATE_FALLBACK, GraphDeps, build_graph
from app.qa.models import ChatProvider
from app.rag.types import ModelsUnavailable, RecalledTurn, Section

# anyio's pytest plugin ships with anyio (a FastAPI dependency); asyncio only.
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- fakes -----------------------------------------------------------------------------


@dataclass
class Partial:
    """Stream `text`, then raise `error` (a provider dying mid-answer)."""

    text: str
    error: Exception


@dataclass
class Slow:
    """Wait `seconds`, then give `reply`."""

    seconds: float
    reply: Any


def _pieces(text: str) -> list[str]:
    return re.findall(r"\S+\s*|\s+", text)


class FakeChat(BaseChatModel):
    """Scripted streaming chat model. Each call takes the next item of `script`:
    str (streamed word by word), dict (streamed as JSON; for structured output),
    Exception (raised before any token), Partial or Slow."""

    script: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("async only")

    async def _chunks(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("FakeChat: no scripted reply left")
        reply = self.script.pop(0)
        if isinstance(reply, Slow):
            await asyncio.sleep(reply.seconds)
            reply = reply.reply
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, Partial):
            for piece in _pieces(reply.text):
                yield piece
            raise reply.error
        text = json.dumps(reply, ensure_ascii=False) if isinstance(reply, dict) else reply
        for piece in _pieces(text):
            yield piece

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for piece in self._chunks(messages):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

    async def _agenerate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        text = "".join([piece async for piece in self._chunks(messages)])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        # Like the real ones: the model's (streamed) text, then a parser.
        return self | RunnableLambda(lambda message: schema.model_validate_json(message.text))


class FakeRetriever:
    """Contract item 2's signature; returns the scripted results in order, then []."""

    def __init__(self, *results: list[Section]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, list[str] | None]] = []

    async def __call__(self, query: str, *, pages: Sequence[str] | None = None) -> list[Section]:
        self.calls.append((query, None if pages is None else list(pages)))
        return self.results.pop(0) if self.results else []


# --- data --------------------------------------------------------------------------------

MATCH = Section(
    "/projects/roleradar",
    "how-roleradar-matches-jobs",
    "How RoleRadar matches jobs",
    "RoleRadar embeds each job and the user's profile and ranks jobs by similarity.",
    score=0.032,
)
TESTS = Section(
    "/projects/roleradar",
    "how-roleradar-tests-its-ai-evals-and-prompt-versions",
    "How RoleRadar tests its AI: evals and prompt versions",
    "Every prompt has a version; evals run on a fixed set of questions.",
    score=0.016,
)
WORK_RIGHTS = Section(
    "/about",
    "work-rights",
    "Work rights",
    "About Skylar's work rights: I'm studying in Sydney and can work in Australia; "
    "happy to discuss details — email me.",
    score=0.03,
)
AVAILABILITY = Section(
    "/about",
    "availability",
    "Availability",
    "I'm looking for internships and part-time work now, and graduate programs and "
    "full-time roles from early 2027.",
    score=0.02,
)

EXAMPLE_Q = "How does RoleRadar decide which jobs match a user?"
EXAMPLE_QUERY = "How RoleRadar decides which jobs match a user"
GOOD_ANSWER = (
    "RoleRadar embeds each job and your profile, then ranks jobs by similarity [1]. "
    "Its prompts are versioned and checked with evals [2]."
)


def rewrite_reply(
    query: str = EXAMPLE_QUERY,
    *,
    language: str = "en",
    private: bool = False,
    off_topic: bool = False,
) -> dict:
    return {"language": language, "private": private, "off_topic": off_topic, "query": query}


def check_reply(unsupported: Sequence[str] = (), passed: bool | None = None) -> dict:
    ok = not unsupported if passed is None else passed
    return {"unsupported": list(unsupported), "passed": ok}


def private_check_reply(
    *, enough_info: bool, unsupported: Sequence[str] = (), passed: bool | None = None
) -> dict:
    ok = enough_info and not unsupported if passed is None else passed
    return {
        "reasoning": "The visitor asked...",
        "enough_info": enough_info,
        "unsupported": list(unsupported),
        "passed": ok,
    }


# --- harness -----------------------------------------------------------------------------


@dataclass
class Run:
    events: list[tuple[str, dict]]
    parts: list[dict]  # raw stream parts, before the filter
    state: dict
    claude: FakeChat
    gemini: FakeChat
    retriever: FakeRetriever

    @property
    def names(self) -> list[str]:
        """Event names without `status`, consecutive tokens collapsed into one."""
        out: list[str] = []
        for name, _ in self.events:
            if name == "status" or (name == "token" and out and out[-1] == "token"):
                continue
            out.append(name)
        return out

    def data(self, name: str) -> list[dict]:
        return [d for n, d in self.events if n == name]

    @property
    def tokens(self) -> str:
        return "".join(d["text"] for d in self.data("token"))

    def raw_text(self, node: str) -> str:
        return "".join(
            output.message_text(p["data"][0])
            for p in self.parts
            if p["type"] == "messages" and p["data"][1].get("langgraph_node") == node
        )

    def index(self, name: str, **match: Any) -> int:
        for i, (n, d) in enumerate(self.events):
            if n == name and all(d.get(k) == v for k, v in match.items()):
                return i
        raise AssertionError(f"no {name} {match} in {self.events}")


def make_deps(
    claude: list[Any], gemini: list[Any], retriever: FakeRetriever, *, timeout: float = 5.0
) -> tuple[GraphDeps, FakeChat, FakeChat]:
    claude_model, gemini_model = FakeChat(script=list(claude)), FakeChat(script=list(gemini))
    deps = GraphDeps(
        ChatProvider("anthropic", "claude-haiku-5-5", claude_model),
        ChatProvider("google", "gemini-2.5-flash-lite", gemini_model),
        retriever,
        embed_model="text-embedding-3-small",
        call_timeout=timeout,
    )
    return deps, claude_model, gemini_model


async def stream_turn(
    graph: Any, question: str, thread: str = "t1", history: Sequence[RecalledTurn] = ()
) -> tuple[list, list, dict]:
    """One turn through the compiled graph, the way the API streams it.

    `history` is the recalled turns; with no embedder in the deps, `recall` passes the
    input's `history` through as it is (graph.py), so tests hand the earlier turns in here.
    """
    config = {"configurable": {"thread_id": thread}}
    parts: list[dict] = []

    async def tap() -> AsyncIterator[dict]:
        stream = graph.astream(
            {"question": question, "history": list(history)},
            config,
            stream_mode=["messages", "custom"],
            version="v2",
        )
        async for part in stream:
            parts.append(part)
            yield part

    events = [e async for e in output.events(tap())]
    state = (await graph.aget_state(config)).values
    return events, parts, state


async def ask(
    question: str,
    *,
    claude: list[Any],
    gemini: list[Any],
    retriever: FakeRetriever,
    call_timeout: float = 5.0,
    history: Sequence[RecalledTurn] = (),
) -> Run:
    deps, claude_model, gemini_model = make_deps(claude, gemini, retriever, timeout=call_timeout)
    graph = build_graph(deps, checkpointer=InMemorySaver())
    events, parts, state = await stream_turn(graph, question, history=history)
    return Run(events, parts, state, claude_model, gemini_model, retriever)


def assert_statuses_paired(run: Run) -> None:
    open_nodes: list[str] = []
    for d in run.data("status"):
        if d["state"] == "start":
            open_nodes.append(d["node"])
        else:
            assert open_nodes.pop() == d["node"]
            assert isinstance(d["ms"], int)
    assert open_nodes == []


def trace_providers(run: Run) -> list[tuple[str, str | None]]:
    return [(n["name"], n["provider"]) for n in run.state["trace"]["nodes"]]


def system_prompt(model: FakeChat, call: int) -> str:
    return str(model.calls[call][0].content)


def earlier_turns(model: FakeChat, call: int) -> list[str]:
    """The `earlier-turn` blocks one model call got, each checked to carry that call's marker."""
    text = "\n".join(str(m.content) for m in model.calls[call])
    [marker] = re.findall(r"tags whose name ends in -([0-9a-f]{16})", text)
    blocks = re.findall(r"<earlier-turn-([0-9a-f]+)>\n(.*?)\n</earlier-turn-\1>", text, re.S)
    assert all(m == marker for m, _ in blocks), "an earlier turn outside this call's marker"
    return [body for _, body in blocks]


# --- streaming filter --------------------------------------------------------------------


async def test_private_question_streams_no_tokens_and_hides_other_nodes():
    run = await ask(
        "Can you work full-time in Australia?",
        claude=[rewrite_reply("work rights", private=True), "I can work in Australia [1]."],
        gemini=[private_check_reply(enough_info=True)],
        retriever=FakeRetriever([WORK_RIGHTS]),
    )
    assert run.data("token") == []
    # The private draft was written (and shown only after the check) ...
    assert len(run.claude.calls) == 2
    assert [d["text"] for d in run.data("answer")] == ["I can work in Australia [1]."]
    # ... but its model is tagged `nostream`, so not even the raw stream carries it.
    assert run.raw_text("answer_private") == ""
    # The raw stream does carry the other nodes' output: the filter is what stops it.
    assert '"enough_info"' in run.raw_text("check_private")
    assert '"query"' in run.raw_text("rewrite")


async def test_general_tokens_are_only_the_answer_node():
    run = await ask(
        EXAMPLE_Q,
        claude=[rewrite_reply(), GOOD_ANSWER],
        gemini=[check_reply()],
        retriever=FakeRetriever([MATCH, TESTS]),
    )
    assert run.tokens == GOOD_ANSWER
    assert '"query"' in run.raw_text("rewrite")
    assert '"unsupported"' in run.raw_text("check")
    assert '"query"' not in run.tokens and '"unsupported"' not in run.tokens


# --- general questions -------------------------------------------------------------------


async def test_general_failed_retry_check_reason_is_unsupported():
    draft = "RoleRadar uses GPT-9 [1]."
    run = await ask(
        EXAMPLE_Q,
        claude=[rewrite_reply(), draft, draft],
        gemini=[check_reply([draft]), check_reply([draft])],
        retriever=FakeRetriever([MATCH], [MATCH]),
    )
    [check] = run.data("check")
    assert check["passed"] is False and check["retried"] is True
    assert check["reason"] == "unsupported"
    assert run.state["check"] == check


async def test_general_question_check_passes():
    run = await ask(
        EXAMPLE_Q,
        claude=[rewrite_reply(), GOOD_ANSWER],
        gemini=[check_reply()],
        retriever=FakeRetriever([MATCH, TESTS]),
    )
    assert run.names == ["route", "token", "answer", "check", "trace"]
    assert run.index("route") > run.index("status", node="rewrite", state="end")
    assert run.data("route") == [{"private": False, "off_topic": False, "language": "en"}]
    assert run.retriever.calls == [(EXAMPLE_QUERY, None)]
    [answer] = run.data("answer")
    assert answer["text"] == GOOD_ANSWER
    assert [c["url"] for c in answer["citations"]] == [MATCH.url, TESTS.url]
    assert run.data("check") == [
        {"passed": True, "unsupported": [], "retried": False, "reason": None}
    ]
    assert_statuses_paired(run)

    state = run.state
    assert state["answer_text"] == GOOD_ANSWER
    assert state["citations"] == answer["citations"]
    assert state["check"] == {
        "passed": True,
        "unsupported": [],
        "retried": False,
        "reason": None,
    }
    assert state["private"] is False and state["language"] == "en"
    assert trace_providers(run) == [
        ("rewrite", "anthropic"),
        ("retrieve", "openai"),
        ("answer", "anthropic"),
        ("check", "google"),
    ]
    [trace] = run.data("trace")
    assert trace == state["trace"]
    assert trace["sections"] == [
        {"url": MATCH.url, "heading": MATCH.heading, "score": MATCH.score},
        {"url": TESTS.url, "heading": TESTS.heading, "score": TESTS.score},
    ]
    assert trace["memory"] == []


async def test_general_question_with_nothing_retrieved_gets_fixed_sentence():
    run = await ask(
        EXAMPLE_Q,
        claude=[rewrite_reply()],
        gemini=[],
        retriever=FakeRetriever([]),
    )
    assert run.names == ["route", "answer", "trace"]
    assert run.data("answer") == [{"text": NOT_FOUND, "citations": []}]
    assert len(run.claude.calls) == 1  # only the rewrite: the answer model was not called
    assert run.gemini.calls == []
    assert run.state["check"] is None


async def test_general_check_fails_twice_retries_once_then_shows_marked():
    first = "RoleRadar ranks jobs by similarity [1]. It was built in 2019."
    second = "RoleRadar ranks jobs by similarity [1]. It uses GPT-9."
    run = await ask(
        EXAMPLE_Q,
        claude=[rewrite_reply(), first, second],
        gemini=[check_reply(["It was built in 2019."]), check_reply(["It uses GPT-9.", "made up"])],
        retriever=FakeRetriever([MATCH], [MATCH, TESTS]),
    )
    assert run.names == ["route", "token", "answer", "reset", "token", "answer", "check", "trace"]
    assert run.data("reset") == [{"reason": "retry"}]
    assert [a["text"] for a in run.data("answer")] == [first, second]
    # Only the final check is shown; quotes not in the answer are dropped.
    assert run.data("check") == [
        {
            "passed": False,
            "unsupported": ["It uses GPT-9."],
            "retried": True,
            "reason": "unsupported",
        }
    ]
    # Searched again for what the failed sentence claimed; told the model what failed.
    assert len(run.retriever.calls) == 2
    assert "It was built in 2019." in run.retriever.calls[1][0]
    assert "It was built in 2019." in system_prompt(run.claude, 2)
    assert run.state["answer_text"] == second
    assert run.state["check"]["retried"] is True
    assert [n for n, _ in trace_providers(run)] == [
        "rewrite",
        "retrieve",
        "answer",
        "check",
        "retrieve",
        "answer",
        "check",
    ]


async def test_general_retry_that_passes():
    run = await ask(
        EXAMPLE_Q,
        claude=[rewrite_reply(), "RoleRadar was built in 2019 [1].", GOOD_ANSWER],
        gemini=[check_reply(["RoleRadar was built in 2019 [1]."]), check_reply()],
        retriever=FakeRetriever([MATCH], [MATCH, TESTS]),
    )
    assert run.data("check") == [
        {"passed": True, "unsupported": [], "retried": True, "reason": None}
    ]
    assert run.state["answer_text"] == GOOD_ANSWER


async def test_off_topic_question_is_declined_without_retrieval_or_check():
    decline = "I can only answer questions about Skylar's projects — try asking about RoleRadar."
    run = await ask(
        "What's the capital of France?",
        claude=[rewrite_reply("capital of France", off_topic=True), decline + " [1]"],
        gemini=[],
        retriever=FakeRetriever(),
    )
    assert run.names == ["route", "token", "answer", "trace"]
    assert run.data("route") == [{"private": False, "off_topic": True, "language": "en"}]
    assert run.data("answer") == [{"text": decline, "citations": []}]
    assert run.retriever.calls == []
    assert run.gemini.calls == []
    assert run.data("check") == []


# --- private questions -------------------------------------------------------------------


async def test_private_insufficient_info_check_reason_is_not_in_about():
    run = await ask(
        "What salary are you expecting?",
        claude=[rewrite_reply("salary", private=True), "I'm looking for internships [1]."],
        gemini=[private_check_reply(enough_info=False)],
        retriever=FakeRetriever([AVAILABILITY]),
    )
    [check] = run.data("check")
    assert check["passed"] is False and check["unsupported"] == []
    assert check["reason"] == "not_in_about"
    assert run.state["check"] == check


async def test_private_no_sections_check_reason_is_not_in_about():
    run = await ask(
        "Where are you based?",
        claude=[rewrite_reply("location", private=True)],
        gemini=[],
        retriever=FakeRetriever([]),
    )
    [check] = run.data("check")
    assert check["passed"] is False and check["unsupported"] == []
    assert check["reason"] == "not_in_about"
    assert run.state["check"] == check


async def test_private_unsupported_draft_check_reason_is_inferred():
    draft = "I can work full-time in Australia [1]."
    run = await ask(
        "Can you work full-time?",
        claude=[rewrite_reply("work rights", private=True), draft],
        gemini=[private_check_reply(enough_info=True, unsupported=[draft], passed=True)],
        retriever=FakeRetriever([WORK_RIGHTS]),
    )
    [check] = run.data("check")
    assert check["passed"] is False and check["unsupported"] == []
    assert check["reason"] == "inferred"
    assert run.state["check"] == check


async def test_private_question_flagged_by_rewrite_shown_only_after_check():
    draft = "I'm studying in Sydney and can work in Australia [1]."
    run = await ask(
        "What are your work rights?",
        claude=[rewrite_reply("Skylar work rights", private=True), draft],
        gemini=[private_check_reply(enough_info=True)],
        retriever=FakeRetriever([WORK_RIGHTS]),
    )
    assert run.names == ["route", "answer", "check", "trace"]
    assert run.data("token") == []
    assert run.index("answer") > run.index("status", node="check", state="end")
    assert run.retriever.calls == [("Skylar work rights", ["/about"])]
    assert run.data("answer") == [
        {
            "text": draft,
            "citations": [{"n": 1, "url": WORK_RIGHTS.url, "heading": "Work rights"}],
        }
    ]
    assert run.data("check") == [
        {"passed": True, "unsupported": [], "retried": False, "reason": None}
    ]
    assert run.state["private"] is True
    assert_statuses_paired(run)


async def test_private_by_keyword_only_still_takes_private_path():
    run = await ask(
        "Can you work full-time in Australia?",
        # The rewrite model misses it; the keyword list catches "full-time".
        claude=[rewrite_reply("work full-time in Australia"), "I can work in Australia [1]."],
        gemini=[private_check_reply(enough_info=True)],
        retriever=FakeRetriever([WORK_RIGHTS]),
    )
    assert run.data("route") == [{"private": True, "off_topic": False, "language": "en"}]
    assert run.retriever.calls[0][1] == ["/about"]
    assert run.data("token") == []


async def test_private_salary_not_on_about_page_gets_fixed_sentence():
    run = await ask(
        "What salary are you expecting?",
        claude=[
            rewrite_reply("Skylar salary expectations", private=True),
            "I'm looking for internships and part-time work now [1].",
        ],
        gemini=[private_check_reply(enough_info=False)],
        retriever=FakeRetriever([AVAILABILITY]),
    )
    assert run.names == ["route", "answer", "check", "trace"]
    assert run.data("answer") == [
        {
            "text": PRIVATE_FALLBACK,
            "citations": [{"n": 1, "url": "/about", "heading": "About Skylar"}],
        }
    ]
    assert run.data("check") == [
        {"passed": False, "unsupported": [], "retried": False, "reason": "not_in_about"}
    ]
    assert run.state["private_check"] == {"enough_info": False, "unsupported": []}
    assert run.data("token") == []


async def test_private_inference_is_flagged_and_replaced():
    inferred = "Yes, I can work full-time in Australia [1]."
    run = await ask(
        "Can you work full-time?",
        claude=[rewrite_reply("Skylar work full-time", private=True), inferred],
        gemini=[private_check_reply(enough_info=True, unsupported=[inferred])],
        retriever=FakeRetriever([WORK_RIGHTS, AVAILABILITY]),
    )
    # The check prompt tells the model that "can work in Australia" is not "full-time".
    check_prompt = system_prompt(run.gemini, 0)
    assert '"can work in Australia" does not support "can work' in check_prompt
    assert "enough_info" in check_prompt
    assert run.state["private_check"] == {"enough_info": True, "unsupported": [inferred]}
    assert run.data("answer")[0]["text"] == PRIVATE_FALLBACK
    assert run.data("check") == [
        {"passed": False, "unsupported": [], "retried": False, "reason": "inferred"}
    ]
    assert run.data("token") == []


async def test_private_check_says_passed_but_lists_a_real_quote_fails():
    draft = "I can work in Australia [1]. I prefer remote work."
    run = await ask(
        "What are your work rights?",
        claude=[rewrite_reply("work rights", private=True), draft],
        gemini=[
            private_check_reply(
                enough_info=True, unsupported=["I prefer remote work."], passed=True
            )
        ],
        retriever=FakeRetriever([WORK_RIGHTS]),
    )
    assert run.data("answer")[0]["text"] == PRIVATE_FALLBACK
    assert run.state["check"]["passed"] is False


@pytest.mark.parametrize(
    "draft",
    [
        "I don't have that on my About page.",
        "  i don’t have that on my about page  ",
        PRIVATE_FALLBACK,
        "please see the About page or email me!",
        "Please see the About page or email me [1].",
        "I don't have that on my About page. Please see the About page or email me.",
    ],
)
async def test_private_refusal_draft_becomes_fixed_sentence_without_check(draft: str):
    run = await ask(
        "Which visa are you on?",
        claude=[rewrite_reply("Skylar visa", private=True), draft],
        gemini=[],
        retriever=FakeRetriever([WORK_RIGHTS]),
    )
    assert run.names == ["route", "answer", "check", "trace"]
    assert run.data("answer") == [
        {
            "text": PRIVATE_FALLBACK,
            "citations": [{"n": 1, "url": "/about", "heading": "About Skylar"}],
        }
    ]
    assert run.data("check") == [
        {"passed": False, "unsupported": [], "retried": False, "reason": "not_in_about"}
    ]
    assert run.gemini.calls == []  # the private check never ran
    assert run.data("token") == []


@pytest.mark.parametrize(
    ("draft", "kept"),
    [
        # fixed sentence tacked on after a real answer
        (
            "I can work in Australia [1]. Please see the About page or email me.",
            "I can work in Australia [1].",
        ),
        # fixed sentence in the middle, with a citation of its own
        (
            "I can work in Australia [1]. Please see the About page or email me [1]. "
            "I'm studying in Sydney [1].",
            "I can work in Australia [1]. I'm studying in Sydney [1].",
        ),
        # in a Chinese answer, with no space after 。
        (
            "我可以在澳大利亚工作[1]。Please see the About page or email me.",
            "我可以在澳大利亚工作[1]。",
        ),
    ],
)
async def test_private_draft_drops_the_fixed_sentence_and_checks_the_rest(draft: str, kept: str):
    run = await ask(
        "What are your work rights?",
        claude=[rewrite_reply("work rights", private=True), draft],
        gemini=[private_check_reply(enough_info=True)],
        retriever=FakeRetriever([WORK_RIGHTS]),
    )
    # The private check ran on the answer without the fixed sentence, and passed it.
    checked = "\n".join(str(m.content) for m in run.gemini.calls[0])
    assert kept in checked and PRIVATE_FALLBACK not in checked
    assert run.data("answer") == [
        {"text": kept, "citations": [{"n": 1, "url": WORK_RIGHTS.url, "heading": "Work rights"}]}
    ]
    assert run.state["check"]["passed"] is True


async def test_private_prompt_asks_for_the_fixed_sentence_word_for_word():
    run = await ask(
        "Which visa are you on?",
        claude=[rewrite_reply("Skylar visa", private=True), PRIVATE_FALLBACK],
        gemini=[],
        retriever=FakeRetriever([WORK_RIGHTS]),
    )
    private_prompt = system_prompt(run.claude, 1)
    assert PRIVATE_FALLBACK in private_prompt
    assert "I don't have that on my About page" not in private_prompt


async def test_private_with_nothing_retrieved_gets_fixed_sentence_without_models():
    run = await ask(
        "Where are you based?",
        claude=[rewrite_reply("Skylar location", private=True)],
        gemini=[],
        retriever=FakeRetriever([]),
    )
    assert run.names == ["route", "answer", "check", "trace"]
    assert run.data("answer")[0]["text"] == PRIVATE_FALLBACK
    assert len(run.claude.calls) == 1 and run.gemini.calls == []


# --- fallbacks and timeouts --------------------------------------------------------------


async def test_rewrite_claude_error_falls_back_to_gemini():
    run = await ask(
        EXAMPLE_Q,
        claude=[RuntimeError("anthropic 529"), GOOD_ANSWER],
        gemini=[rewrite_reply(), check_reply()],
        retriever=FakeRetriever([MATCH, TESTS]),
    )
    assert run.names == ["route", "token", "answer", "check", "trace"]
    assert trace_providers(run)[0] == ("rewrite", "google")
    assert trace_providers(run)[2] == ("answer", "anthropic")
    end = run.data("status")[1]
    assert end["node"] == "rewrite" and end["provider"] == "google"
    assert end["model"] == "gemini-2.5-flash-lite"


async def test_answer_claude_dies_mid_stream_resets_and_gemini_answers():
    run = await ask(
        EXAMPLE_Q,
        claude=[rewrite_reply(), Partial("RoleRadar embeds each", RuntimeError("stream cut"))],
        gemini=[GOOD_ANSWER, check_reply()],
        retriever=FakeRetriever([MATCH, TESTS]),
    )
    assert run.names == ["route", "token", "reset", "token", "answer", "check", "trace"]
    assert run.data("reset") == [{"reason": "fallback"}]
    tokens_after_reset = "".join(
        d["text"] for n, d in run.events[run.index("reset") :] if n == "token"
    )
    assert tokens_after_reset == GOOD_ANSWER
    assert run.state["answer_text"] == GOOD_ANSWER


async def test_answer_claude_error_uses_gemini_for_answer_and_check():
    run = await ask(
        EXAMPLE_Q,
        claude=[rewrite_reply(), RuntimeError("anthropic 500")],
        gemini=[GOOD_ANSWER, check_reply()],
        retriever=FakeRetriever([MATCH, TESTS]),
    )
    assert run.names == ["route", "reset", "token", "answer", "check", "trace"]
    assert run.data("reset") == [{"reason": "fallback"}]
    assert trace_providers(run) == [
        ("rewrite", "anthropic"),
        ("retrieve", "openai"),
        ("answer", "google"),
        ("check", "google"),
    ]


async def test_check_gemini_error_falls_back_to_claude():
    run = await ask(
        EXAMPLE_Q,
        claude=[rewrite_reply(), GOOD_ANSWER, check_reply()],
        gemini=[RuntimeError("google 503")],
        retriever=FakeRetriever([MATCH, TESTS]),
    )
    assert run.data("check") == [
        {"passed": True, "unsupported": [], "retried": False, "reason": None}
    ]
    assert trace_providers(run)[-1] == ("check", "anthropic")


async def test_rewrite_both_providers_fail_raises_models_unavailable():
    with pytest.raises(ModelsUnavailable):
        await ask(
            EXAMPLE_Q,
            claude=[RuntimeError("anthropic down")],
            gemini=[RuntimeError("google down")],
            retriever=FakeRetriever([MATCH]),
        )


async def test_slow_model_call_counts_as_an_error():
    run = await ask(
        EXAMPLE_Q,
        claude=[Slow(2.0, rewrite_reply()), GOOD_ANSWER],
        gemini=[rewrite_reply(), check_reply()],
        retriever=FakeRetriever([MATCH, TESTS]),
        call_timeout=0.2,
    )
    assert trace_providers(run)[0] == ("rewrite", "google")
    assert run.data("check")[0]["passed"] is True


async def test_bad_structured_output_counts_as_an_error():
    run = await ask(
        EXAMPLE_Q,
        claude=["not json at all", GOOD_ANSWER],
        gemini=[rewrite_reply(), check_reply()],
        retriever=FakeRetriever([MATCH, TESTS]),
    )
    assert trace_providers(run)[0] == ("rewrite", "google")


# --- language and turns ------------------------------------------------------------------


async def test_chinese_question_searches_in_english_and_answers_in_chinese():
    answer_zh = "RoleRadar 把职位和你的资料都转成向量，按相似度排序[1]。"
    run = await ask(
        "RoleRadar 怎么判断哪些职位适合用户？",
        claude=[rewrite_reply(EXAMPLE_QUERY, language="zh"), answer_zh],
        gemini=[check_reply()],
        retriever=FakeRetriever([MATCH]),
    )
    assert run.data("route") == [{"private": False, "off_topic": False, "language": "zh"}]
    assert run.retriever.calls == [(EXAMPLE_QUERY, None)]
    assert "Write your reply in Chinese" in system_prompt(run.claude, 1)
    assert run.state["language"] == "zh"
    assert run.state["answer_text"] == answer_zh


async def test_second_turn_does_not_reuse_the_first_turns_fields():
    deps, _, _ = make_deps(
        claude=[
            rewrite_reply("salary", private=True),
            "I'm looking for internships [1].",
            rewrite_reply("capital of France", off_topic=True),
            "Sorry, I only answer questions about my projects.",
        ],
        gemini=[private_check_reply(enough_info=False)],
        retriever=FakeRetriever([AVAILABILITY]),
    )
    # One graph, one thread: the checkpointer hands turn 1's state to turn 2.
    graph = build_graph(deps, checkpointer=InMemorySaver())
    _, _, first = await stream_turn(graph, "What salary are you expecting?")
    assert first["private"] is True and first["private_check"] is not None

    events, _, second = await stream_turn(graph, "What's the capital of France?")
    assert second["question"] == "What's the capital of France?"
    assert second["private"] is False
    assert second["private_check"] is None
    assert second["check"] is None
    assert second["sections"] == []
    assert second["citations"] == []
    assert [n["name"] for n in second["trace"]["nodes"]] == ["rewrite", "answer"]
    assert [d["text"] for n, d in events if n == "answer"] == [
        "Sorry, I only answer questions about my projects."
    ]


# --- memory in the answer prompt ---------------------------------------------------------

FOLLOW_UP = "Which model does it use for that?"
FOLLOW_UP_QUERY = "Which model RoleRadar uses to decide which jobs match a user"
EARLIER = RecalledTurn("turn-1", EXAMPLE_Q, GOOD_ANSWER, 0.72)
EARLIER_BLOCK = f"Q: {EXAMPLE_Q}\nA: {GOOD_ANSWER}"


async def test_second_turn_answer_gets_the_first_turn_as_data():
    follow_up_answer = "It uses an embedding model to rank jobs by similarity [1]."
    deps, claude, gemini = make_deps(
        claude=[rewrite_reply(), GOOD_ANSWER, rewrite_reply(FOLLOW_UP_QUERY), follow_up_answer],
        gemini=[check_reply(), check_reply()],
        retriever=FakeRetriever([MATCH, TESTS], [MATCH]),
    )
    graph = build_graph(deps, checkpointer=InMemorySaver())
    _, _, first = await stream_turn(graph, EXAMPLE_Q)
    assert earlier_turns(claude, 1) == []  # turn 1 has nothing to recall

    recalled = RecalledTurn("turn-1", EXAMPLE_Q, first["answer_text"], 0.72)
    _, _, second = await stream_turn(graph, FOLLOW_UP, history=[recalled])
    # Turn 2's rewrite and answer both got turn 1's question and answer, inside the marker.
    assert earlier_turns(claude, 2) == [EARLIER_BLOCK]
    assert earlier_turns(claude, 3) == [EARLIER_BLOCK]
    answer_system = system_prompt(claude, 3)
    assert "Earlier turns" in answer_system and "never cite" in answer_system
    # The check still sees only this turn's sections, not the earlier turns.
    assert earlier_turns(gemini, 1) == []
    assert GOOD_ANSWER not in "\n".join(str(m.content) for m in gemini.calls[1])
    # Citations still point only at this turn's sections.
    assert second["answer_text"] == follow_up_answer
    assert [c["url"] for c in second["citations"]] == [MATCH.url]


async def test_retried_answer_also_gets_the_earlier_turns():
    run = await ask(
        FOLLOW_UP,
        claude=[
            rewrite_reply(FOLLOW_UP_QUERY),
            "It uses GPT-9 [1].",
            "It ranks by similarity [1].",
        ],
        gemini=[check_reply(["It uses GPT-9 [1]."]), check_reply()],
        retriever=FakeRetriever([MATCH], [MATCH, TESTS]),
        history=[EARLIER],
    )
    assert run.data("reset") == [{"reason": "retry"}]
    assert earlier_turns(run.claude, 1) == [EARLIER_BLOCK]
    assert earlier_turns(run.claude, 2) == [EARLIER_BLOCK]


async def test_private_and_off_topic_answers_get_no_earlier_turns():
    private = await ask(
        "Can you work in Australia?",
        claude=[rewrite_reply("work rights", private=True), "I can work in Australia [1]."],
        gemini=[private_check_reply(enough_info=True)],
        retriever=FakeRetriever([WORK_RIGHTS]),
        history=[EARLIER],
    )
    assert earlier_turns(private.claude, 0) == [EARLIER_BLOCK]  # the rewrite did get them
    private_prompt = "\n".join(str(m.content) for m in private.claude.calls[1])
    assert EXAMPLE_Q not in private_prompt and GOOD_ANSWER not in private_prompt

    off_topic = await ask(
        "What's the capital of France?",
        claude=[rewrite_reply("capital of France", off_topic=True), "I only answer that."],
        gemini=[],
        retriever=FakeRetriever(),
        history=[EARLIER],
    )
    off_topic_prompt = "\n".join(str(m.content) for m in off_topic.claude.calls[1])
    assert EXAMPLE_Q not in off_topic_prompt and GOOD_ANSWER not in off_topic_prompt


# --- model settings (no network) ---------------------------------------------------------


def test_model_settings_are_explicit():
    from app.config import Settings
    from app.qa.models import build_providers, gemini_thinking

    claude, gemini = build_providers(Settings(database_url="postgresql://x@127.0.0.1:1/x"))
    assert claude.chat.thinking == {"type": "disabled"}  # Haiku 5.5 thinks if omitted
    assert claude.chat.temperature is None and claude.chat.top_p is None  # Haiku 5.5 rejects
    assert claude.chat.max_tokens == gemini.chat.max_output_tokens == 1024
    assert claude.chat.default_request_timeout == gemini.chat.timeout == 30
    assert gemini_thinking("gemini-2.5-flash-lite") == {"thinking_budget": 0, "temperature": 0}
    assert gemini_thinking("gemini-3.5-flash-lite") == {"thinking_level": "minimal"}
