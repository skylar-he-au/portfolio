"""State of the Q&A graph.

`question` and `history` come in with each turn. Everything in `TURN_FIELDS` belongs to one
turn only: the first node resets it, so nothing leaks from the previous turn when a
checkpointer keeps the conversation's state (3b2).
"""

from typing import Any, TypedDict

from app.rag.types import RecalledTurn, Section


class QAInput(TypedDict):
    question: str
    history: list[RecalledTurn]


class QAState(TypedDict, total=False):
    question: str
    history: list[RecalledTurn]

    # Set by `rewrite`.
    query: str  # standalone English search query
    language: str  # language of the visitor's question, e.g. "en", "zh"
    private: bool  # about Skylar personally: answered from /about only, shown only if checked
    off_topic: bool  # neither the projects nor Skylar: polite decline, no retrieval, no check

    # Set by `retrieve`.
    sections: list[Section]

    # Set by `answer` / `answer_private` / the check nodes.
    answer_text: str  # final text; every [n] in it matches `citations`
    citations: list[dict[str, Any]]  # [{"n", "url", "heading"}]
    check: dict[str, Any] | None  # {"passed", "unsupported", "retried"}; None when not checked
    private_check: dict[str, Any] | None  # {"enough_info", "unsupported"} of the private draft

    # Retry after a failed check (general questions, at most once).
    retried: bool
    retry_query: str
    previous_unsupported: list[str]

    # Trace: one entry per node that ran ({"name", "provider", "model", "ms"}), then the
    # full `trace` event payload built by `finish`.
    nodes: list[dict[str, Any]]
    trace: dict[str, Any]


def turn_defaults() -> dict[str, Any]:
    """Fresh values for every per-turn field (new lists each call)."""
    return {
        "query": "",
        "language": "en",
        "private": False,
        "off_topic": False,
        "sections": [],
        "answer_text": "",
        "citations": [],
        "check": None,
        "private_check": None,
        "retried": False,
        "retry_query": "",
        "previous_unsupported": [],
        "nodes": [],
        "trace": {},
    }


TURN_FIELDS = frozenset(turn_defaults())
