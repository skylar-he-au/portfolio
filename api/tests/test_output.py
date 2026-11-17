# ruff: noqa: RUF001 - Chinese test strings use full-width punctuation
"""Citations, `unsupported` quotes and the token filter (pure code, no graph)."""

from langchain_core.messages import AIMessageChunk

from app.qa.output import answer_token, resolve_citations, verify_unsupported
from app.rag.types import Section

TWO = [
    Section("/projects/roleradar", "how-roleradar-matches-jobs", "How RoleRadar matches jobs", "a"),
    Section("/projects/roleradar", "how-roleradar-is-tested", "How RoleRadar is tested", "b"),
]


def test_out_of_range_citation_is_dropped():
    text, citations = resolve_citations("It ranks jobs [1]. It also flies [3].", TWO)
    assert text == "It ranks jobs [1]. It also flies."
    assert citations == [
        {
            "n": 1,
            "url": "/projects/roleradar#how-roleradar-matches-jobs",
            "heading": "How RoleRadar matches jobs",
        }
    ]


def test_valid_citations_become_the_right_urls():
    text, citations = resolve_citations("It ranks jobs and tests itself [1][2].", TWO)
    assert text == "It ranks jobs and tests itself [1][2]."
    assert [c["url"] for c in citations] == [
        "/projects/roleradar#how-roleradar-matches-jobs",
        "/projects/roleradar#how-roleradar-is-tested",
    ]


def test_comma_list_is_split_and_bad_numbers_removed():
    text, citations = resolve_citations("Both [2, 1, 7]. None [0].", TWO)
    assert text == "Both [2][1]. None."
    assert [c["n"] for c in citations] == [1, 2]


def test_no_sections_strips_every_citation():
    text, citations = resolve_citations("I can only talk about projects [1].", [])
    assert text == "I can only talk about projects."
    assert citations == []


def test_unsupported_keeps_only_real_substrings():
    answer = "RoleRadar scores jobs by skill overlap [1]. It was built in 2019."
    kept = verify_unsupported(
        [
            "It was built in 2019.",  # exact
            "It was built in 2019",  # same sentence without the full stop
            "RoleRadar uses GPT-9 to score jobs.",  # paraphrase/invention: not in the answer
            "  ",
        ],
        answer,
    )
    assert kept == ["It was built in 2019."]


def test_unsupported_trims_trailing_punctuation_to_match():
    answer = "我用技能重合度给职位打分[1]，它很快"
    assert verify_unsupported(["它很快。"], answer) == ["它很快"]


def _part(node: str, text: str, ns: tuple = ()) -> dict:
    return {
        "type": "messages",
        "ns": ns,
        "data": (AIMessageChunk(content=text), {"langgraph_node": node}),
    }


def test_answer_token_only_passes_the_answer_node():
    assert answer_token(_part("answer", "Hi")) == "Hi"
    for node in ("rewrite", "check", "answer_private", "check_private"):
        assert answer_token(_part(node, "secret")) is None
    assert answer_token(_part("answer", "")) is None
    assert answer_token(_part("answer", "Hi", ns=("sub:1",))) is None
    assert answer_token({"type": "custom", "ns": (), "data": {"event": "x", "data": {}}}) is None


def test_answer_token_ignores_tool_call_blocks():
    chunk = AIMessageChunk(content=[{"type": "tool_use", "id": "t", "name": "x", "input": {}}])
    part = {"type": "messages", "ns": (), "data": (chunk, {"langgraph_node": "answer"})}
    assert answer_token(part) is None
