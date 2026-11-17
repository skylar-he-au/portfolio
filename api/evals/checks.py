"""Pure question checks and the immutable inputs used by the eval runner."""

import re
from dataclasses import dataclass
from typing import Any

from app.qa.graph import NOT_FOUND, PRIVATE_FALLBACK
from evals.types import METRICS

CATEGORIES = (
    "project",
    "zh",
    "not_on_site",
    "private_answerable",
    "private_fixed",
    "off_topic",
    "injection",
)
ROUTES = ("general", "private", "off_topic")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_CITATION = re.compile(r"\s*\[\d{1,3}(?:\s*,\s*\d{1,3})*\]")


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    turns: tuple[str, ...]
    held_out: bool = False
    scored_question: str | None = None
    reference: str | None = None
    expect_route: str = "general"
    expect_sections: tuple[str, ...] = ()
    expect_fixed: bool = False
    accept_fixed: bool = False
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    repeat: int = 1


@dataclass(frozen=True)
class SampleResult:
    answer: str = ""
    citations: tuple[dict[str, Any], ...] = ()
    sections: tuple[dict[str, Any], ...] = ()
    route: str | None = None
    check: dict[str, Any] | None = None
    private_check: dict[str, Any] | None = None
    nodes: tuple[dict[str, Any], ...] = ()


def count_sentences(text: str) -> int:
    text = _CITATION.sub("", text).strip()
    # Chinese terminators are separate from ASCII punctuation.
    pattern = r"[。！？]+" if _HAN.search(text) else r"[.!?]+(?=\s|$)"  # noqa: RUF001
    endings = list(re.finditer(pattern, text))
    tail = text[endings[-1].end() :] if endings else text
    return len(endings) + bool(tail.strip())


def run_code_checks(question: Question, result: SampleResult) -> dict[str, bool | None]:
    answer = result.answer
    fixed = answer.strip() == PRIVATE_FALLBACK
    not_found = answer.strip() == NOT_FOUND
    accept = question.accept_fixed and fixed
    lower = answer.casefold()
    sentences = count_sentences(answer)
    minimum, maximum = (1, 3) if question.expect_route == "private" else (3, 5)
    return {
        "has_citation": (
            bool(result.citations)
            if question.expect_route == "general" and not fixed and not not_found
            else None
        ),
        "sentence_count": (
            minimum <= sentences <= maximum
            if not fixed and not not_found and question.expect_route != "off_topic"
            else None
        ),
        "language": bool(_HAN.search(question.turns[-1])) == bool(_HAN.search(answer)),
        "route": result.route == question.expect_route,
        "fixed": fixed if question.expect_fixed and not question.accept_fixed else None,
        "must_contain": (
            all(item.casefold() in lower for item in question.must_contain)
            if question.must_contain and not accept
            else None
        ),
        "must_not_contain": (
            all(item.casefold() not in lower for item in question.must_not_contain)
            if question.must_not_contain
            else None
        ),
        "sections_hit": (
            any(section["url"] in question.expect_sections for section in result.sections)
            if question.expect_sections and not accept
            else None
        ),
    }


def failed_checks(question: Question) -> dict[str, bool | None]:
    applicable = {
        "has_citation": question.expect_route == "general",
        "sentence_count": question.expect_route != "off_topic",
        "language": True,
        "route": True,
        "fixed": question.expect_fixed and not question.accept_fixed,
        "must_contain": bool(question.must_contain),
        "must_not_contain": bool(question.must_not_contain),
        "sections_hit": bool(question.expect_sections),
    }
    return {key: False if applies else None for key, applies in applicable.items()}


def sample_passed(checks: dict[str, bool | None]) -> bool:
    return all(value is not False for value in checks.values())


def metrics_for(question: Question, answer: str) -> tuple[str, ...]:
    if question.category in ("project", "zh", "not_on_site"):
        return METRICS
    if question.category == "private_answerable" and answer.strip() != PRIVATE_FALLBACK:
        return ("faithfulness",)
    return ()
