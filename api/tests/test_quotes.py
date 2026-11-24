# ruff: noqa: RUF001 - typographic characters are what these tests are about
"""Quotes behind each citation (v10): plain text, checking quotes, and the stream filter.

The shared fixture `fixtures/quotes.json` is the step 10 contract with the web side (10c2).
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from app.qa.output import (
    events,
    find_quote,
    plain_text,
    resolve_answer,
    resolve_citations,
    split_quotes,
)
from app.rag.types import Section

# anyio's pytest plugin ships with anyio (a FastAPI dependency); asyncio only.
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "quotes.json").read_text())
SECTIONS = FIXTURE["sections"]


def _section(fixture: dict[str, Any]) -> Section:
    return Section(fixture["page"], fixture["section_id"], fixture["heading"], fixture["markdown"])


MATCHING = _section(SECTIONS[0])  # "How RoleRadar matches jobs"
TAILORING = _section(SECTIONS[1])


# --- plain text ---------------------------------------------------------------------


@pytest.mark.parametrize("fixture", SECTIONS, ids=lambda f: f["section_id"])
def test_fixture_sections_have_the_agreed_plain_text(fixture: dict[str, Any]):
    assert plain_text(fixture["markdown"]).split("\n") == fixture["plain"]


def test_plain_text_rules():
    markdown = "\n".join(
        [
            "### A *small* heading",
            "",
            "Some **bold**, _soft_ and `snake_case_name` with a [link](https://x.example/a_b)",
            "that wraps onto a second line.",
            "",
            "- First item -- with a dash",
            "  continued here.",
            "1. Numbered “quoted” item…",
            "",
            "| Name | What it does |",
            "| --- | :---: |",
            "| `ask` | Answers questions |",
            "",
            "```python",
            "x = a * b * c",
            "```",
            "",
            "> Quoted line with an en–dash and an em—dash.",
            "",
            "---",
            "Escaped \\*stars\\* stay. snake_case stays.",
        ]
    )
    assert plain_text(markdown).split("\n") == [
        "A small heading",
        "Some bold, soft and snake_case_name with a link that wraps onto a second line.",
        "First item -- with a dash continued here.",
        'Numbered "quoted" item...',
        "Name",
        "What it does",
        "ask",
        "Answers questions",
        "x = a * b * c",
        "Quoted line with an en-dash and an em--dash.",
        "Escaped *stars* stay. snake_case stays.",
    ]


# --- checking quotes ----------------------------------------------------------------


def test_quote_copied_with_its_list_number_still_matches():
    plain = "\n".join(SECTIONS[1]["plain"])
    quote = "2. Code rejects anything it can verify is new: numbers, source line references"
    assert find_quote(quote, plain) == quote[3:]
    assert find_quote("- " + quote[3:], plain) == quote[3:]


@pytest.mark.parametrize("fixture", SECTIONS, ids=lambda f: f["section_id"])
def test_fixture_quotes(fixture: dict[str, Any]):
    plain = "\n".join(fixture["plain"])
    for case in fixture["match"]:
        assert find_quote(case["quote"], plain) == case["expect"], case["why"]
    for case in fixture["no_match"]:
        assert find_quote(case["quote"], plain) is None, case["why"]


def _answer(quotes: list[str]) -> str:
    lines = "\n".join(f"[1] {q}" for q in quotes)
    answer = "RoleRadar reads each ad once and code decides the fit [1]."
    return f"{answer}\n\n<quotes>\n{lines}\n</quotes>"


def test_quotes_kept_and_dropped():
    one_word = "Each ad is read twice and the result is shared by all users."
    edge = "Because tiers come from code, every reason traces back to a line in the ad!"
    two_paragraphs = (
        "Because tiers come from code, every reason traces back to a line in the ad. "
        "On real data the rules alone over-rated matches."
    )
    apostrophe = "So I added a per-user resume check of the ad’s must-have requirements"
    exact = (
        "Every extracted fact has to include a verbatim quote from the ad, "
        "and code drops or downgrades any quote it cannot find there."
    )
    raw = _answer([exact, one_word, edge, two_paragraphs, apostrophe])
    text, citations, given = resolve_answer(raw, [MATCHING])
    assert text == "RoleRadar reads each ad once and code decides the fit [1]."
    assert given == 5
    # Kept, dropped, kept (trimmed), dropped, kept (straight apostrophe).
    assert citations == [
        {
            "n": 1,
            "url": MATCHING.url,
            "heading": MATCHING.heading,
            "quotes": [
                exact,
                "Because tiers come from code, every reason traces back to a line in the ad",
                "So I added a per-user resume check of the ad's must-have requirements",
            ],
        }
    ]


def test_quote_goes_only_to_its_own_section_and_used_citations():
    quote = "The original resume is never changed."
    raw = (
        "It never edits the original [2]. It reads ads once [1].\n<quotes>\n"
        f"[1] {quote}\n[2] {quote}\n[3] {quote}\n[2] {quote}\n</quotes>"
    )
    text, citations, given = resolve_answer(raw, [MATCHING, TAILORING])
    assert text == "It never edits the original [2]. It reads ads once [1]."
    assert given == 3  # the repeated [2] line counts once; [3] is out of range but was given
    assert [(c["n"], c["quotes"]) for c in citations] == [(1, []), (2, [quote])]


def test_quote_lines_accept_several_numbers_and_wrapping_quotes():
    text, pairs = split_quotes(
        'A [1][2].\n<QUOTES>\n[1][2]: "One line."\n[3, 4] Two.\nnoise\n</QUOTES>'
    )
    assert text == "A [1][2].\n"
    assert pairs == [(1, '"One line."'), (2, '"One line."'), (3, "Two."), (4, "Two.")]


def test_unclosed_block_and_block_before_the_answer():
    assert split_quotes("Answer [1].\n<quotes>\n[1] Something long enough")[0] == "Answer [1].\n"
    text, pairs = split_quotes("<quotes>\n[1] A quoted sentence.\n</quotes>\nAnswer [1].")
    assert text.strip() == "Answer [1]."
    assert pairs == [(1, "A quoted sentence.")]


def test_answer_without_quotes_block_still_has_empty_quotes():
    text, citations, given = resolve_answer("It reads ads once [1].", [MATCHING])
    assert text == "It reads ads once [1]."
    assert given == 0
    assert citations[0]["quotes"] == []


def test_resolve_citations_cuts_quotes_and_gives_empty_quotes():
    text, citations = resolve_citations(
        "Private answer [1].\n<quotes>\n[1] x\n</quotes>", [MATCHING]
    )
    assert text == "Private answer [1]."
    assert citations[0]["quotes"] == []


# --- the stream ---------------------------------------------------------------------


def _token(text: str) -> dict[str, Any]:
    chunk = AIMessageChunk(content=text)
    return {"type": "messages", "ns": (), "data": (chunk, {"langgraph_node": "answer"})}


def _custom(event: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"type": "custom", "ns": (), "data": {"event": event, "data": data}}


async def _collect(parts: list[dict[str, Any]]) -> list[tuple[str, dict]]:
    async def stream() -> AsyncIterator[dict[str, Any]]:
        for part in parts:
            yield part

    return [pair async for pair in events(stream())]


def _streamed(pairs: list[tuple[str, dict]]) -> str:
    return "".join(data["text"] for name, data in pairs if name == "token")


QUOTE = "Every extracted fact has to include a verbatim quote from the ad."


async def test_stream_never_sends_the_quotes_block_split_across_tokens():
    raw = f"It reads each ad once [1]. Code decides [1].\n\n<quotes>\n[1] {QUOTE}\n</quotes>"
    # Every split point, including inside "<quotes>" and "</quotes>".
    for size in (1, 2, 3, 5, 7):
        parts = [_token(raw[i : i + size]) for i in range(0, len(raw), size)]
        streamed = _streamed(await _collect(parts))
        assert "<" not in streamed and "quotes" not in streamed and QUOTE not in streamed
        assert streamed == "It reads each ad once [1]. Code decides [1].\n\n"


async def test_stream_releases_a_held_back_angle_bracket():
    parts = [_token("Latency stays <"), _token("200 ms [1].")]
    assert _streamed(await _collect(parts)) == "Latency stays <200 ms [1]."


async def test_stream_text_after_a_closed_block_is_sent():
    parts = [_token("<quo"), _token("tes>\n[1] "), _token(QUOTE), _token("\n</quo"), _token("tes>")]
    parts += [_token("\nAnswer [1].")]
    assert _streamed(await _collect(parts)) == "\nAnswer [1]."


async def test_stream_reset_while_a_marker_is_half_open():
    parts = [
        _token("First model answer [1]. <quo"),
        _custom("reset", {"reason": "fallback"}),
        _token("tes> leaked?"),  # the fallback model starts afresh: this is plain text
        _token(" Second answer [1].\n<quotes>\n"),
        _token(f"[1] {QUOTE}\n</quotes>"),
        _custom("answer", {"text": "Second", "citations": []}),
    ]
    pairs = await _collect(parts)
    tokens = [data["text"] for name, data in pairs if name == "token"]
    assert tokens[0] == "First model answer [1]. "
    after_reset = "".join(tokens[1:])
    assert after_reset == "tes> leaked? Second answer [1].\n"
    assert [name for name, _ in pairs if name != "token"] == ["reset", "answer"]


async def test_stream_reset_inside_a_block_does_not_hide_the_next_answer():
    parts = [
        _token("Answer [1].\n<quotes>\n[1] half a quo"),
        _custom("reset", {"reason": "retry"}),
        _token("New answer [1]."),
    ]
    assert _streamed(await _collect(parts)) == "Answer [1].\nNew answer [1]."
