"""State of the Q&A graph.

`question` and `history` come in with each turn. Everything in `TURN_FIELDS` belongs to one
turn only: the first node resets it, so nothing leaks from the previous turn when a
checkpointer keeps the conversation's state (3b2). `turns` is the conversation's memory: it
lives across turns, and only the API adds to it, after a turn reached `done`.
"""

from typing import Any, TypedDict

from app.rag.types import RecalledTurn, Section


class QAInput(TypedDict):
    question: str
    history: list[RecalledTurn]


class QAState(TypedDict, total=False):
    question: str
    history: list[RecalledTurn]  # earlier turns picked by `recall` (memory.recall)

    # Memory, kept across turns: finished turns, oldest first (memory.StoredTurn dicts).
    turns: list[dict[str, Any]]
    # Set by `recall`: this question's embedding (float32 bytes), stored with the turn.
    question_embedding: bytes

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
    # {"enough_info", "unsupported"} of the private draft; {"enough_info": None,
    # "unsupported": [], "guard": <rule>} when code replaced it before the check (10s).
    private_check: dict[str, Any] | None

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
        "question_embedding": b"",
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
