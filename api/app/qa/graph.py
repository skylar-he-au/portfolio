"""The Q&A graph: recall -> rewrite -> retrieve -> answer -> check, retrying a failed check once.

    begin -> recall -> rewrite -+-> (off-topic) ---------------------------> answer -> finish
                                +-> retrieve -+-> (general) -> answer -> check -+-> finish
                                |             |                  ^              |
                                |             |                  +- retrieve <- prepare_retry
                                |             |                    (failed check, first time only)
                                |             +-> (private) -> answer_private -> check_private

- General questions stream as they are written (`answer` is the only node whose tokens
  reach the visitor, see `output.answer_token`); the check then marks unsupported
  sentences. A failed check sends `reset retry` and answers once more; if that fails too
  the answer stays, marked.
- Private questions (rewrite flag or `private.looks_private`) use /about only, are never
  streamed, and are shown only if the check passes; otherwise a fixed sentence.
- Off-topic questions get a one-line decline: no retrieval, no check.
- `recall` (3b2) embeds the question and picks earlier turns of the conversation from the
  state's `turns` (see `memory`); the rewrite node uses them to resolve "it". Without an
  embedder in `GraphDeps` it passes the input `history` through unchanged.

Nodes send the `route`, `status`, `reset`, `answer`, `check` and `trace` events (contract
item 7) through `get_stream_writer()`; the API adds `meta`, `token`, `done` and `error`.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from app.config import Settings
from app.qa import memory, prompts
from app.qa.models import CALL_TIMEOUT_S, ChatProvider, build_providers, stream_text, structured
from app.qa.output import ANSWER_NODE, resolve_citations, verify_unsupported
from app.qa.private import looks_private
from app.qa.prompts import PRIVATE_FALLBACK
from app.qa.state import QAInput, QAState, turn_defaults
from app.rag.types import Embedder, Retriever, Section

NOT_FOUND = "I couldn't find that in the project pages — please email me."
# A private draft that only declines (the fixed sentence, or the refusal the v2-v6 prompt
# asked for) is sent as the fixed sentence by code, without the private check.
_REFUSALS = {"i don't have that on my about page", PRIVATE_FALLBACK}
ABOUT_CITATION = {"n": 1, "url": "/about", "heading": "About Skylar"}
ABOUT_PAGES = ["/about"]
MAX_SECTIONS = 6
MAX_QUERY_CHARS = 1000

PRIVATE_ANSWER_NODE = "answer_private"  # must never equal output.ANSWER_NODE


@dataclass(frozen=True)
class GraphDeps:
    claude: ChatProvider
    gemini: ChatProvider
    retriever: Retriever  # db and embedder already bound (3b2)
    embed_model: str  # for status/trace of the retrieve node
    call_timeout: float = CALL_TIMEOUT_S  # per model call; tests make it small
    embedder: Embedder | None = None  # for `recall` (3b2); None: no memory


def build_deps(
    settings: Settings, retriever: Retriever, embedder: Embedder | None = None
) -> GraphDeps:
    claude, gemini = build_providers(settings)
    return GraphDeps(claude, gemini, retriever, embed_model=settings.embed_model, embedder=embedder)


# --- small helpers -------------------------------------------------------------------


def _language(code: str) -> str:
    code = code.strip().lower().replace("_", "-")
    head = code.split("-")[0]
    if 2 <= len(head) <= 3 and head.isascii() and head.isalpha():
        return head
    return "en"


class _Node:
    """Sends `status start` now and `status end` (with provider, model, ms) on `end()`;
    `end()` returns the node's trace entry."""

    def __init__(self, writer: StreamWriter, name: str) -> None:
        self.writer = writer
        self.name = name
        self.started = time.perf_counter()
        writer({"event": "status", "data": {"node": name, "state": "start"}})

    def end(self, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        ms = round((time.perf_counter() - self.started) * 1000)
        data: dict[str, Any] = {"node": self.name, "state": "end", "ms": ms}
        if provider:
            data["provider"] = provider
            data["model"] = model
        self.writer({"event": "status", "data": data})
        return {"name": self.name, "provider": provider, "model": model, "ms": ms}


def _emit_answer(writer: StreamWriter, text: str, citations: list[dict[str, Any]]) -> None:
    writer({"event": "answer", "data": {"text": text, "citations": citations}})


def _emit_check(writer: StreamWriter, check: dict[str, Any]) -> None:
    writer({"event": "check", "data": check})


def _normalise(text: str) -> str:
    return text.strip().replace("’", "'").casefold().rstrip(".!?。！？ ")  # noqa: RUF001


_REFUSAL_KEYS = {_normalise(r) for r in _REFUSALS}


def _is_refusal(text: str) -> bool:
    return _normalise(text) in _REFUSAL_KEYS


def _retry_query(state: QAState) -> str:
    # Search again for what the failed sentences claim; without quotes, use the
    # visitor's own wording, which differs from the rewritten query.
    quotes = state.get("previous_unsupported") or []
    extra = " ".join(quotes) if quotes else state["question"]
    return f"{state['query']} {extra}"[:MAX_QUERY_CHARS]


# --- the graph -----------------------------------------------------------------------


def build_graph(deps: GraphDeps, checkpointer: Any = None) -> Any:
    """Compile the graph. Input: `{"question": str, "history": list[RecalledTurn]}`."""
    answer_order = (deps.claude, deps.gemini)
    check_order = (deps.gemini, deps.claude)
    timeout = deps.call_timeout

    async def begin(state: QAState) -> dict[str, Any]:
        return turn_defaults()

    async def recall(state: QAState) -> dict[str, Any]:
        """Memory: embed the question, pick earlier turns (memory.recall) for `history`."""
        if deps.embedder is None:
            return {}
        writer = get_stream_writer()
        node = _Node(writer, "recall")
        [vector] = await deps.embedder.embed([state["question"]])
        history = memory.recall(state.get("turns") or [], vector)  # type: ignore[arg-type]
        trace = node.end("openai", deps.embed_model)
        return {
            "history": history,
            "question_embedding": memory.pack_embedding(vector),
            "nodes": [*state["nodes"], trace],
        }

    async def rewrite(state: QAState) -> dict[str, Any]:
        writer = get_stream_writer()
        node = _Node(writer, "rewrite")
        question = state["question"]
        messages = prompts.rewrite_messages(question, state.get("history") or [])
        out, provider = await structured(
            "rewrite", answer_order, prompts.RewriteOut, messages, deadline_s=timeout
        )
        language = _language(out.language)
        private = out.private or looks_private(question)
        off_topic = out.off_topic and not private
        query = " ".join(out.query.split())[:MAX_QUERY_CHARS] or question
        trace = node.end(provider.name, provider.model)
        writer(
            {
                "event": "route",
                "data": {"private": private, "off_topic": off_topic, "language": language},
            }
        )
        return {
            "query": query,
            "language": language,
            "private": private,
            "off_topic": off_topic,
            "nodes": [*state["nodes"], trace],
        }

    async def retrieve(state: QAState) -> dict[str, Any]:
        writer = get_stream_writer()
        node = _Node(writer, "retrieve")
        query = state.get("retry_query") or state["query"]
        pages = ABOUT_PAGES if state["private"] else None
        found = list(await deps.retriever(query, pages=pages))[:MAX_SECTIONS]
        if not found and state.get("retried"):
            found = state["sections"]  # the retry search found nothing: keep the first sections
        trace = node.end("openai", deps.embed_model)
        return {"sections": found, "nodes": [*state["nodes"], trace]}

    async def answer(state: QAState) -> dict[str, Any]:
        """General and off-topic answers; the only node whose tokens are streamed."""
        writer = get_stream_writer()
        node = _Node(writer, "answer")
        question, language = state["question"], state["language"]
        sections: Sequence[Section] = state["sections"]
        if state["off_topic"]:
            messages = prompts.off_topic_messages(question, language)
        elif not sections:
            trace = node.end()
            _emit_answer(writer, NOT_FOUND, [])
            return {"answer_text": NOT_FOUND, "citations": [], "nodes": [*state["nodes"], trace]}
        else:
            messages = prompts.answer_messages(
                question,
                language,
                sections,
                state.get("history") or [],
                state.get("previous_unsupported") or [],
            )

        def reset_fallback() -> None:
            writer({"event": "reset", "data": {"reason": "fallback"}})

        raw, provider = await stream_text(
            "answer", answer_order, messages, deadline_s=timeout, on_switch=reset_fallback
        )
        text, citations = resolve_citations(raw, [] if state["off_topic"] else sections)
        trace = node.end(provider.name, provider.model)
        _emit_answer(writer, text, citations)
        return {"answer_text": text, "citations": citations, "nodes": [*state["nodes"], trace]}

    async def answer_private(state: QAState) -> dict[str, Any]:
        """Private answer: a draft only. Nothing is sent until `check_private` passes it."""
        writer = get_stream_writer()
        node = _Node(writer, "answer")
        sections: Sequence[Section] = state["sections"]
        if not sections:
            trace = node.end()
            check = {
                "passed": False,
                "unsupported": [],
                "retried": False,
                "reason": "not_in_about",
            }
            _emit_answer(writer, PRIVATE_FALLBACK, [ABOUT_CITATION])
            _emit_check(writer, check)
            return {
                "answer_text": PRIVATE_FALLBACK,
                "citations": [ABOUT_CITATION],
                "check": check,
                "nodes": [*state["nodes"], trace],
            }
        messages = prompts.private_answer_messages(state["question"], state["language"], sections)
        # nostream: LangGraph keeps this model out of the `messages` stream altogether, a
        # second guard besides the node-name filter in output.answer_token.
        raw, provider = await stream_text(
            PRIVATE_ANSWER_NODE, answer_order, messages, deadline_s=timeout, visible=False
        )
        text, citations = resolve_citations(raw, sections)
        trace = node.end(provider.name, provider.model)
        if _is_refusal(text):
            check = {"passed": False, "unsupported": [], "retried": False, "reason": "not_in_about"}
            _emit_answer(writer, PRIVATE_FALLBACK, [ABOUT_CITATION])
            _emit_check(writer, check)
            return {
                "answer_text": PRIVATE_FALLBACK,
                "citations": [ABOUT_CITATION],
                "check": check,
                "nodes": [*state["nodes"], trace],
            }
        return {"answer_text": text, "citations": citations, "nodes": [*state["nodes"], trace]}

    async def check(state: QAState) -> dict[str, Any]:
        writer = get_stream_writer()
        node = _Node(writer, "check")
        text = state["answer_text"]
        messages = prompts.check_messages(state["question"], text, state["sections"])
        out, provider = await structured(
            "check", check_order, prompts.CheckOut, messages, deadline_s=timeout
        )
        unsupported = verify_unsupported(out.unsupported, text)
        passed = out.passed and not unsupported
        result = {
            "passed": passed,
            "unsupported": unsupported,
            "retried": state["retried"],
            "reason": None if passed else "unsupported",
        }
        trace = node.end(provider.name, provider.model)
        if passed or state["retried"]:
            _emit_check(writer, result)  # final; a check that leads to a retry is not shown
        return {"check": result, "nodes": [*state["nodes"], trace]}

    async def prepare_retry(state: QAState) -> dict[str, Any]:
        get_stream_writer()({"event": "reset", "data": {"reason": "retry"}})
        previous = (state.get("check") or {}).get("unsupported") or []
        update: dict[str, Any] = {
            "retried": True,
            "previous_unsupported": previous,
            "answer_text": "",
            "citations": [],
            "check": None,
        }
        update["retry_query"] = _retry_query({**state, **update})  # type: ignore[typeddict-item]
        return update

    async def check_private(state: QAState) -> dict[str, Any]:
        writer = get_stream_writer()
        node = _Node(writer, "check")
        draft = state["answer_text"]
        messages = prompts.private_check_messages(state["question"], draft, state["sections"])
        out, provider = await structured(
            "check", check_order, prompts.PrivateCheckOut, messages, deadline_s=timeout
        )
        unsupported = verify_unsupported(out.unsupported, draft)
        passed = out.enough_info and out.passed and not unsupported
        trace = node.end(provider.name, provider.model)
        update: dict[str, Any] = {
            "private_check": {"enough_info": out.enough_info, "unsupported": unsupported},
            "nodes": [*state["nodes"], trace],
        }
        if passed:
            text, citations = draft, state["citations"]
        else:
            text, citations = PRIVATE_FALLBACK, [ABOUT_CITATION]
        result = {
            "passed": passed,
            "unsupported": [],
            "retried": False,
            "reason": None if passed else "not_in_about" if not out.enough_info else "inferred",
        }
        _emit_answer(writer, text, citations)
        _emit_check(writer, result)
        return {**update, "answer_text": text, "citations": citations, "check": result}

    async def finish(state: QAState) -> dict[str, Any]:
        trace = {
            "nodes": state["nodes"],
            "sections": [
                {"url": s.url, "heading": s.heading, "score": s.score} for s in state["sections"]
            ],
            "memory": [
                {"turn_id": t.turn_id, "question": t.question, "score": t.score}
                for t in state.get("history") or []
            ],
        }
        get_stream_writer()({"event": "trace", "data": trace})
        return {"trace": trace}

    def after_rewrite(state: QAState) -> Literal["answer", "retrieve"]:
        return "answer" if state["off_topic"] else "retrieve"

    def after_retrieve(state: QAState) -> Literal["answer", "answer_private"]:
        return "answer_private" if state["private"] else "answer"

    def after_answer(state: QAState) -> Literal["check", "finish"]:
        if state["off_topic"] or not state["sections"]:
            return "finish"
        return "check"

    def after_answer_private(state: QAState) -> Literal["check_private", "finish"]:
        return "finish" if state.get("check") else "check_private"

    def after_check(state: QAState) -> Literal["prepare_retry", "finish"]:
        check_result = state.get("check") or {}
        if check_result.get("passed") or state["retried"]:
            return "finish"
        return "prepare_retry"

    builder = StateGraph(QAState, input_schema=QAInput)
    builder.add_node("begin", begin)
    builder.add_node("recall", recall)
    builder.add_node("rewrite", rewrite)
    builder.add_node("retrieve", retrieve)
    builder.add_node(ANSWER_NODE, answer)
    builder.add_node(PRIVATE_ANSWER_NODE, answer_private)
    builder.add_node("check", check)
    builder.add_node("prepare_retry", prepare_retry)
    builder.add_node("check_private", check_private)
    builder.add_node("finish", finish)

    builder.add_edge(START, "begin")
    builder.add_edge("begin", "recall")
    builder.add_edge("recall", "rewrite")
    builder.add_conditional_edges("rewrite", after_rewrite)
    builder.add_conditional_edges("retrieve", after_retrieve)
    builder.add_conditional_edges(ANSWER_NODE, after_answer)
    builder.add_conditional_edges(PRIVATE_ANSWER_NODE, after_answer_private)
    builder.add_conditional_edges("check", after_check)
    builder.add_edge("prepare_retry", "retrieve")
    builder.add_edge("check_private", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)
