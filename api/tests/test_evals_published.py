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
    "requirements", "cost_usd", "cost_pools_usd",
}  # fmt: skip
NAME_FIELDS = {"kind", "run_id", "date", "prompt_version", "route", "split"}
NAME = re.compile(r"[A-Za-z0-9._:-]{1,64}")


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


def test_published_match_summaries_contain_no_eval_set_text() -> None:
    if not MATCH_SET.exists():
        pytest.skip("docs repo eval set not next to this checkout (set MATCH_SET_PATH)")
    yaml = pytest.importorskip("yaml")
    jobs = yaml.safe_load(MATCH_SET.read_text(encoding="utf-8"))["jobs"]
    texts: set[str] = set()
    for job in jobs:
        texts.update(line.strip() for line in job["jd"].splitlines() if len(line.strip()) >= 20)
        texts.update(r["text"] for r in job["requirements"])
        texts.add(job["title"])
    summaries = _match_summaries()
    if not summaries:
        pytest.skip("no job-match summary published yet")
    for path, _ in summaries:
        raw = path.read_text(encoding="utf-8")
        assert not [t for t in texts if t in raw], path.name
