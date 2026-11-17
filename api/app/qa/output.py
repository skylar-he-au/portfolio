"""Turning model output and graph stream parts into what visitors see.

Everything here is plain code, not prompts: citation links are built from `Section.url`,
`unsupported` sentences are kept only if they really appear in the answer, and only the
general answer node's tokens are ever streamed.
"""

import re
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from app.rag.types import Section

# The only node whose model output may reach the visitor as `token` events. The private
# answer node has a different name (`answer_private`), so its unchecked text can never
# match this filter, even if the filter were written wrong.
ANSWER_NODE = "answer"

# "[1]", "[1][2]" or "[1, 2]", with the whitespace before it (so a dropped one leaves no gap).
_CITE_RE = re.compile(r"\s*\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_TRAILING_PUNCT = ".!?;:,。！？；：，"  # noqa: RUF001 - full-width marks end Chinese sentences


def resolve_citations(text: str, sections: Sequence[Section]) -> tuple[str, list[dict[str, Any]]]:
    """Keep `[n]` only where 1 <= n <= len(sections); return (text, citations used).

    Citations keep the section numbering the model saw, so `[3]` in the text is
    `{"n": 3, ...}` in the list. `[1, 2]` is rewritten as `[1][2]`.
    """
    used: set[int] = set()

    def keep_valid(match: re.Match[str]) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        valid = list(dict.fromkeys(n for n in numbers if 1 <= n <= len(sections)))
        if not valid:
            return ""
        used.update(valid)
        whole = match.group(0)
        leading = whole[: len(whole) - len(whole.lstrip())]
        return leading + "".join(f"[{n}]" for n in valid)

    cleaned = _CITE_RE.sub(keep_valid, text).strip()
    citations = [
        {"n": n, "url": sections[n - 1].url, "heading": sections[n - 1].heading}
        for n in sorted(used)
    ]
    return cleaned, citations


def verify_unsupported(candidates: Iterable[str], answer: str) -> list[str]:
    """Keep the check's `unsupported` quotes that are really substrings of `answer`.

    A quote that differs only by trailing punctuation is trimmed until it matches; anything
    else the model paraphrased or invented is dropped, and so are duplicates.
    """
    kept: list[str] = []
    for raw in candidates:
        quote = raw.strip()
        while quote and quote not in answer and quote[-1] in _TRAILING_PUNCT:
            quote = quote[:-1].rstrip()
        if quote and quote in answer and quote not in kept:
            kept.append(quote)
    # A quote inside a longer kept quote marks the same text twice: keep the longer one.
    return [q for q in kept if not any(q != other and q in other for other in kept)]


def message_text(message: Any) -> str:
    """Text of a message chunk; tool-call and other non-text blocks count as empty."""
    if isinstance(message, BaseMessage):
        return message.text
    return ""


def answer_token(part: dict[str, Any]) -> str | None:
    """The visitor-facing text in one `astream(..., version="v2")` part, if any.

    Only `messages` parts from the top-level graph's `answer` node count; `rewrite`,
    `check` and `answer_private` output is dropped.
    """
    if part.get("type") != "messages" or part.get("ns"):
        return None
    message, metadata = part["data"]
    if metadata.get("langgraph_node") != ANSWER_NODE:
        return None
    return message_text(message) or None


async def events(stream: AsyncIterator[dict[str, Any]]) -> AsyncIterator[tuple[str, dict]]:
    """(event name, data) pairs for SSE from `graph.astream(input, config,
    stream_mode=["messages", "custom"], version="v2")`.

    Custom parts are the graph's own events (`route`, `status`, `reset`, `answer`, `check`,
    `trace`); messages parts become `token` only through `answer_token`.
    """
    async for part in stream:
        kind = part.get("type")
        if kind == "custom" and not part.get("ns"):
            payload = part["data"]
            yield payload["event"], payload["data"]
        elif kind == "messages":
            text = answer_token(part)
            if text:
                yield "token", {"text": text}
