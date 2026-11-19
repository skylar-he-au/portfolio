"""The job-match summaries published in `content/evals/` carry scores and counts only.

The job ads, the requirement texts and the guides stay in the docs repo (SPEC "评测先于上线",
job-match items). Every string in a published match summary must sit at one of the few
fields that hold names (run id, date, prompt version, route, split, model ids), and those
must look like names. When the docs repo's eval set is next to this checkout (or
`MATCH_SET_PATH` points at it), no ad line, requirement text or note from it may appear in
any published summary either.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "content" / "evals"
MATCH_SET = Path(
    os.environ.get("MATCH_SET_PATH", ROOT.parent / "docs" / "evals" / "match-set.yaml")
)

TOP_LEVEL = {
    "kind", "run_id", "date", "prompt_version", "route", "split", "repeats", "questions",
    "samples", "errors", "fallback_samples", "models", "scores", "code_checks", "by_kind",
    "requirements", "cost_usd", "cost_pools_usd", "grader", "grading",
}  # fmt: skip
NAME_FIELDS = {"kind", "run_id", "date", "prompt_version", "route", "split"}
# "+" joins two prompt versions in one run id ("match-v5+check-v2", step 6 round 2).
NAME = re.compile(r"[A-Za-z0-9._:+-]{1,64}")
# Who graded a round-2 summary: the subscription grader (docs/evals/GRADING.md), at
# `grader` and `models.judge_*` only.
GRADER = re.compile(r"[a-z0-9.-]{1,40} \(subscription, GRADING\.md\)")
GRADER_PATHS = {("grader",), ("models", "judge_align"), ("models", "judge_faithfulness")}
COMPARE = ROOT / "content" / "evals" / "compare"
# A step's models in a comparison table: "openai:gpt-6-luna:none -> anthropic:claude-haiku-5-5:off
# (max 1024)", the read step adding ", up to 12 requirements".
_CHOICE = r"[a-z0-9.-]{1,40}:[a-z0-9.-]{1,40}:[a-z]{1,10}"
ORDER = re.compile(rf"{_CHOICE}( -> {_CHOICE})* \(max \d{{1,6}}\)(, up to \d{{1,2}} requirements)?")
COMPARE_NAME = re.compile(r"[A-Za-z0-9._:+-]{1,96}")  # replay run ids run past 64


def _match_summaries() -> list[tuple[Path, Any]]:
    files = sorted(EVALS.glob("*.json"))
    loaded = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in files]
    return [(path, data) for path, data in loaded if data.get("kind") == "match"]


def _strings(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        found = [((*path, "<key>"), key) for key in value]
        for key, inner in value.items():
            found += _strings(inner, (*path, key))
        return found
    if isinstance(value, list):
        return [item for inner in value for item in _strings(inner, (*path, "[]"))]
    return []


def _allowed(path: tuple[str, ...], text: str) -> bool:
    if path in GRADER_PATHS and GRADER.fullmatch(text):
        return True
    if path == ("grading", "status"):
        return text in {"complete", "incomplete"}
    if not NAME.fullmatch(text):
        return False
    if path[-1:] == ("<key>",):
        return True  # keys: metric, split, check and kind names, all name-shaped
    return (len(path) == 1 and path[0] in NAME_FIELDS) or (len(path) == 2 and path[0] == "models")


def test_published_match_summaries_hold_names_and_numbers_only() -> None:
    summaries = _match_summaries()
    if not summaries:
        pytest.skip("no job-match summary published yet")
    for path, data in summaries:
        assert set(data) <= TOP_LEVEL, path.name
        bad = [(p, s) for p, s in _strings(data) if not _allowed(p, s)]
        assert not bad, (path.name, bad[:3])


def test_the_name_check_rejects_text() -> None:
    sample = {"run_id": "x", "models": {"answer": "claude-haiku-5-5"}, "scores": {"tune": {}}}
    assert all(_allowed(p, s) for p, s in _strings(sample))
    for leak in (
        {"jd": "Senior engineer"},
        {"scores": {"tune": {"recall": {"note": "Python"}}}},
        {"route": "Python, TypeScript or JavaScript"},
        {"by_kind": {"real": ["3+ years of Python"]}},
    ):
        assert not all(_allowed(p, s) for p, s in _strings(leak)), leak


def _compare_tables() -> list[tuple[Path, Any]]:
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(COMPARE.glob("*.json"))]


def _compare_allowed(path: tuple[str, ...], text: str) -> bool:
    if path[-1:] == ("order",):
        return bool(ORDER.fullmatch(text))
    return bool(COMPARE_NAME.fullmatch(text))


def test_published_comparison_tables_hold_names_and_numbers_only() -> None:
    tables = _compare_tables()
    if not tables:
        pytest.skip("no comparison table published yet")
    for path, data in tables:
        assert set(data["steps"]) == {"read", "match", "check", "write"}, path.name
        bad = [(p, s) for p, s in _strings(data) if not _compare_allowed(p, s)]
        assert not bad, (path.name, bad[:3])


def test_the_comparison_check_rejects_text() -> None:
    order = ("steps", "read", "options", "[]", "order")
    assert _compare_allowed(order, "google:gemini-3.5-flash-lite:minimal (max 1024)")
    assert _compare_allowed(
        order, "anthropic:claude-haiku-5-5:off (max 1024), up to 12 requirements"
    )
    for path, leak in (
        (("steps", "read", "options", "[]", "option"), "TypeScript and React"),
        (("steps", "read", "options", "[]", "order"), "Senior engineer (max 3)"),
        (("note",), "3+ years of Python"),
    ):
        assert not _compare_allowed(path, leak), leak


def test_the_name_check_takes_round_two_names_and_the_grader() -> None:
    grader = "claude-sonnet-5-5 (subscription, GRADING.md)"
    assert _allowed(("prompt_version",), "match-v5+check-v2")
    assert _allowed(("grader",), grader)
    assert _allowed(("models", "judge_faithfulness"), grader)
    assert _allowed(("grading", "status"), "complete")
    assert not _allowed(("route",), grader)
    assert not _allowed(("grader",), "Senior engineer (subscription, GRADING.md)")
    assert not _allowed(("grading", "status"), "Python")


def _eval_set_texts() -> set[str]:
    if not MATCH_SET.exists():
        pytest.skip("docs repo eval set not next to this checkout (set MATCH_SET_PATH)")
    yaml = pytest.importorskip("yaml")
    jobs = yaml.safe_load(MATCH_SET.read_text(encoding="utf-8"))["jobs"]
    texts: set[str] = set()
    for job in jobs:
        texts.update(line.strip() for line in job["jd"].splitlines() if len(line.strip()) >= 20)
        texts.update(r["text"] for r in job["requirements"])
        texts.add(job["title"])
    return texts


def test_published_comparison_tables_contain_no_eval_set_text() -> None:
    texts = _eval_set_texts()
    tables = _compare_tables()
    if not tables:
        pytest.skip("no comparison table published yet")
    for path, _ in tables:
        raw = path.read_text(encoding="utf-8")
        assert not [t for t in texts if t in raw], path.name


def test_published_match_summaries_contain_no_eval_set_text() -> None:
    texts = _eval_set_texts()
    summaries = _match_summaries()
    if not summaries:
        pytest.skip("no job-match summary published yet")
    for path, _ in summaries:
        raw = path.read_text(encoding="utf-8")
        assert not [t for t in texts if t in raw], path.name
