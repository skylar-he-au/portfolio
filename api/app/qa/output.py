"""Turning model output and graph stream parts into what visitors see.

Everything here is plain code, not prompts: citation links are built from `Section.url`,
`unsupported` sentences are kept only if they really appear in the answer, quotes the
model gives for a citation are kept only if they really appear in that section (v10), and
only the general answer node's tokens are ever streamed, with the quotes block cut out.
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

# --- quotes (v10) ----------------------------------------------------------------------
# The answer model ends its reply with a block the visitor never sees:
#     <quotes>
#     [2] A sentence copied from section 2.
#     </quotes>
# Code cuts the block out of the text (streamed and final), then keeps a quote only if, in
# the plain text below, it is a substring of one block of section n.
QUOTES_OPEN = "<quotes"
QUOTES_CLOSE = "</quotes"
_QUOTE_LINE = re.compile(r"^\s*((?:\[\d{1,3}(?:\s*,\s*\d{1,3})*\]\s*)+)[:\-]?\s*(.*?)\s*$")
MIN_QUOTE_CHARS = 12  # shorter "quotes" would mark common words all over the page
MAX_QUOTES_PER_CITATION = 5  # an answer has at most 5 sentences, each one claim

# Plain text, the same rules the browser applies to the page's `textContent` (step 10
# contract "出处的格式"): Markdown marks removed, typography folded to ASCII, whitespace
# collapsed. One line per block (paragraph, list item, heading, table cell, code line), so
# a quote can never run from one paragraph into the next.
_TYPOGRAPHY = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "--",
        "\u2013": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s+(.*?))?(?:\s+#+)?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,9}[.)])\s+(.*)$")
_RULE = re.compile(r"^\s{0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?")
_CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.DOTALL)
_INLINE_RULES = (
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),  # links and images keep their text
    (re.compile(r"<((?:https?|mailto):[^>\s]+)>"), r"\1"),  # autolinks
    (re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*"), r"\1"),
    (re.compile(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)"), r"\1"),
    (re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)"), r"\1"),  # not inside snake_case
)
_SPACE = re.compile(r"\s+")
_QUOTE_LIST_MARKER = re.compile(r"^(?:[-*+]|\d{1,9}[.)])\s+")  # "2. Code rejects ..." copied
# Characters a quote may differ by at its ends: sentence punctuation and quotation marks.
_EDGE = " .,;:!?\"'()"


def _inline(text: str) -> str:
    """Inline Markdown removed; code spans keep their content as written."""
    out: list[str] = []
    pos = 0
    for match in _CODE_SPAN.finditer(text):
        out.append(_marks(text[pos : match.start()]))
        out.append(match.group(2))
        pos = match.end()
    out.append(_marks(text[pos:]))
    return "".join(out)


_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>])")
_ESCAPED = "\\`*_{}[]()#+-.!|>"
_PRIVATE_USE = 0xE000  # escaped marks wait here, out of the rules' way


def _marks(text: str) -> str:
    text = _ESCAPE.sub(lambda m: chr(_PRIVATE_USE + _ESCAPED.index(m.group(1))), text)
    for pattern, replacement in _INLINE_RULES:
        text = pattern.sub(replacement, text)
    return "".join(
        _ESCAPED[ord(c) - _PRIVATE_USE]
        if _PRIVATE_USE <= ord(c) < _PRIVATE_USE + len(_ESCAPED)
        else c
        for c in text
    )


def fold(text: str) -> str:
    """Typography folded to ASCII and whitespace collapsed to one space (no Markdown)."""
    return _SPACE.sub(" ", text.translate(_TYPOGRAPHY)).strip()


def plain_text(markdown: str) -> str:
    """A section's Markdown as plain text: one line per block, see `_TYPOGRAPHY` above."""
    blocks: list[str] = []
    current: list[str] = []
    fence = ""

    def close() -> None:
        if current:
            blocks.append(_inline(" ".join(current)))
            current.clear()

    for line in markdown.splitlines():
        opened = _FENCE.match(line)
        if fence:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence):
                fence = ""
            else:
                blocks.append(line)  # code: as written, one block per line
            continue
        if opened:
            close()
            fence = opened.group(1)
            continue
        line = _BLOCKQUOTE.sub("", line)
        if not line.strip() or _RULE.match(line):
            close()
            continue
        heading = _HEADING.match(line)
        if heading:
            close()
            blocks.append(_inline(heading.group(1) or ""))
            continue
        if line.lstrip().startswith("|"):
            close()
            if not _TABLE_SEPARATOR.match(line):
                cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
                blocks.extend(_inline(cell) for cell in cells)
            continue
        item = _LIST_ITEM.match(line)
        if item:
            close()
            current.append(item.group(1))
            continue
        current.append(line)
    close()
    return "\n".join(b for b in (fold(block) for block in blocks) if b)


def _quote_blocks(raw: str, final: bool) -> tuple[str, list[str]]:
    """(text without `<quotes>` blocks, the blocks' contents).

    `final=False` is for a stream still being written: an unclosed block hides the rest,
    and a tail that could be the start of `<quotes` is held back. The visible text then only
    ever grows as more of the stream arrives, so it can be sent piece by piece.
    """
    lower = raw.lower()
    visible: list[str] = []
    inner: list[str] = []
    pos = 0
    while True:
        start = lower.find(QUOTES_OPEN, pos)
        if start < 0:
            tail = raw[pos:]
            if not final:
                tail = _hold_back(tail)
            visible.append(tail)
            break
        visible.append(raw[pos:start])
        body_start = start + len(QUOTES_OPEN)
        tag_end = raw.find(">", body_start, body_start + 20)
        if tag_end >= 0:
            body_start = tag_end + 1
        end = lower.find(QUOTES_CLOSE, body_start)
        if end < 0:  # unclosed: everything after the tag is quotes
            inner.append(raw[body_start:])
            break
        inner.append(raw[body_start:end])
        after = end + len(QUOTES_CLOSE)
        rest = raw[after:]
        closing = re.match(r"\s*>", rest)
        if closing:
            pos = after + closing.end()
        elif not final and not rest.strip():
            break  # the `>` may still come
        else:
            pos = after
    return "".join(visible), inner


def _hold_back(tail: str) -> str:
    """`tail` without an ending that could still grow into `<quotes`."""
    lower = tail.lower()
    for size in range(min(len(tail), len(QUOTES_OPEN) - 1), 0, -1):
        if QUOTES_OPEN.startswith(lower[-size:]):
            return tail[:-size]
    return tail


def visible_text(raw: str) -> str:
    """What a visitor may see of a stream so far: no quotes block, nor the start of one."""
    return _quote_blocks(raw, final=False)[0]


def split_quotes(raw: str) -> tuple[str, list[tuple[int, str]]]:
    """(answer text without its quotes block, the (n, quote) pairs in the block, deduplicated)."""
    text, inner = _quote_blocks(raw, final=True)
    pairs: list[tuple[int, str]] = []
    for line in "\n".join(inner).splitlines():
        match = _QUOTE_LINE.match(line)
        if not match or not match.group(2):
            continue
        for number in re.findall(r"\d{1,3}", match.group(1)):
            pair = (int(number), match.group(2))
            if pair not in pairs:
                pairs.append(pair)
    return text, pairs


def find_quote(quote: str, plain: str) -> str | None:
    """`quote` as it appears in one block of `plain` (see `plain_text`), or None.

    Word for word: only whitespace, the Markdown and typography that `plain_text` removes,
    and punctuation or quotation marks at either end may differ.
    """
    folded = fold(_inline(_QUOTE_LIST_MARKER.sub("", quote.strip())))
    blocks = plain.split("\n")
    candidates = (folded, folded.strip("\"'"), folded.strip(_EDGE))
    for candidate in dict.fromkeys(candidates):
        if len(candidate) >= MIN_QUOTE_CHARS and any(candidate in b for b in blocks):
            return candidate
    return None


def resolve_answer(raw: str, sections: Sequence[Section]) -> tuple[str, list[dict[str, Any]], int]:
    """General answer (v10): (text, citations with checked `quotes`, number of quotes given).

    The quotes block is cut out of the text; each citation keeps up to
    `MAX_QUOTES_PER_CITATION` of its quotes that `find_quote` finds in its own section.
    """
    body, pairs = split_quotes(raw)
    text, citations = resolve_citations(body, sections)
    plains: dict[int, str] = {}
    for citation in citations:
        n = citation["n"]
        kept: list[str] = citation["quotes"]
        for number, quote in pairs:
            if number != n or len(kept) >= MAX_QUOTES_PER_CITATION:
                continue
            if n not in plains:
                plains[n] = plain_text(sections[n - 1].text)
            found = find_quote(quote, plains[n])
            if found and found not in kept:
                kept.append(found)
    return text, citations, len(pairs)


def resolve_citations(text: str, sections: Sequence[Section]) -> tuple[str, list[dict[str, Any]]]:
    """Keep `[n]` only where 1 <= n <= len(sections); return (text, citations used).

    Citations keep the section numbering the model saw, so `[3]` in the text is
    `{"n": 3, ...}` in the list. `[1, 2]` is rewritten as `[1][2]`. Any `<quotes>` block is
    cut out, and every citation gets `quotes: []` (`resolve_answer` fills them).
    """
    text = _quote_blocks(text, final=True)[0]
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
        {"n": n, "url": sections[n - 1].url, "heading": sections[n - 1].heading, "quotes": []}
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
    `trace`); messages parts become `token` only through `answer_token`, and only the part
    `visible_text` allows: the quotes block (v10) is never sent. A `reset` (fallback model,
    retry) or the final `answer` starts the next stream afresh; what was held back is dropped,
    since the `answer` event carries the whole text anyway.
    """
    raw, sent = "", ""
    async for part in stream:
        kind = part.get("type")
        if kind == "custom" and not part.get("ns"):
            payload = part["data"]
            if payload["event"] in ("reset", "answer"):
                raw, sent = "", ""
            yield payload["event"], payload["data"]
        elif kind == "messages":
            text = answer_token(part)
            if not text:
                continue
            raw += text
            visible = visible_text(raw)
            if len(visible) > len(sent) and visible.startswith(sent):
                yield "token", {"text": visible[len(sent) :]}
                sent = visible
