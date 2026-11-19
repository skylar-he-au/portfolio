"""The job-match eval set and the pure checks of one sample (no model calls here).

Rules (SPEC "评测先于上线", job-match items):
- Code checks, all must pass: the run gave a guide; the output fits the schema and the
  stream carries only the agreed event shapes; every link is a section retrieved for that
  requirement (on a project page); every link in the guide is one the match agent found;
  the guide is in the job ad's language; no stream event or panel field contains a system
  prompt sentence or (injection items) any of the item's `forbidden` strings.
- Scores by requirement id, after a judge maps each read requirement to gold ids
  (`alignment`; several read requirements may map to one gold id, one read requirement
  may map to several (the subscription grader, 6 r2), unmapped ones are "extra"):
  - section recall: of the gold requirements that have expected sections, the share with
    at least one guide link among them (through any read requirement mapped to it). An item
    with no such gold requirement has no recall (None), not 0 or 1.
  - precision: of the guide links of mapped requirements, the share that are among the
    expected sections of the gold requirement (a link for a gold "not covered" requirement
    counts as wrong). Links of unmapped requirements are left out. None when no link counts.
  - not-covered accuracy: for each mapped read requirement, right when it is in the "Not
    covered" line exactly when its gold requirement is `not_covered` (an unchecked one is
    not in the line). None when nothing is mapped.
  - unchecked: the guide's count, kept apart.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.match.prompts import SYSTEM_PROMPT_PROBES
from app.match.state import MAX_REQUIREMENTS, Guide

KINDS = ("real", "injection", "long", "zh")
CODE_CHECKS = (
    "completed",
    "schema",
    "links_retrieved",
    "links_from_findings",
    "language",
    "no_leak",
)
PROJECT_PREFIX = "/projects/"
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_HAN = re.compile(r"[一-鿿]")
_STAGES = {"fetching", "reading", "matching", "writing", "checking"}
_STATUSES = {"found", "not_found", "unchecked"}


@dataclass(frozen=True)
class GoldRequirement:
    id: str
    text: str
    sections: tuple[str, ...] = ()
    not_covered: bool = False


@dataclass(frozen=True)
class MatchJob:
    id: str
    kind: str
    holdout: bool
    title: str
    jd: str
    requirements: tuple[GoldRequirement, ...]
    forbidden: tuple[str, ...] = ()

    @property
    def language(self) -> str:
        return "zh" if self.kind == "zh" else "en"


@dataclass(frozen=True)
class MatchSample:
    """What one run of the graph left: the stream and the final notes, JSON-shaped."""

    events: tuple[dict[str, Any], ...] = ()
    requirements: tuple[dict[str, Any], ...] = ()  # {"id", "text", "query"}
    findings: tuple[dict[str, Any], ...] = ()  # as in the notes: requirement_id, status, ...
    retrieved: Mapping[str, list[str] | None] = field(default_factory=dict)  # id -> urls
    result: dict[str, Any] | None = None  # Guide.model_dump()
    # The findings after the check (the links it rejected removed); the same as `findings`
    # when the check did not run. Empty in rows written before 6 r2.
    checked_findings: tuple[dict[str, Any], ...] = ()
    # The check's word on each pair: {"requirement_id", "url", "shows": true|false|"missing"}.
    check_verdicts: tuple[dict[str, Any], ...] = ()


def load_match_set(path: Path | str) -> list[MatchJob]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read match set: {type(error).__name__}") from error
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("match set version must be 1")
    entries = data.get("jobs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("jobs must be a non-empty list")
    jobs: list[MatchJob] = []
    seen: set[str] = set()
    for index, item in enumerate(entries, 1):
        prefix = f"job {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be a mapping")
        ident = item.get("id")
        if not isinstance(ident, str) or not _ID.fullmatch(ident) or ident in seen:
            raise ValueError(f"{prefix}: id must be unique, lowercase letters, digits, hyphens")
        seen.add(ident)
        if item.get("kind") not in KINDS:
            raise ValueError(f"{ident}: invalid kind")
        if type(item.get("holdout")) is not bool:
            raise ValueError(f"{ident}: holdout must be a boolean")
        jd = item.get("jd")
        if not isinstance(jd, str) or not jd.strip():
            raise ValueError(f"{ident}: jd must be a non-empty string")
        forbidden = item.get("forbidden", [])
        if not isinstance(forbidden, list) or not all(
            isinstance(text, str) and text for text in forbidden
        ):
            raise ValueError(f"{ident}: forbidden must be a list of strings")
        raw_requirements = item.get("requirements")
        if not isinstance(raw_requirements, list) or not raw_requirements:
            raise ValueError(f"{ident}: requirements must be a non-empty list")
        requirements: list[GoldRequirement] = []
        for raw in raw_requirements:
            requirements.append(_gold_requirement(ident, raw, {r.id for r in requirements}))
        jobs.append(
            MatchJob(
                id=ident,
                kind=item["kind"],
                holdout=item["holdout"],
                title=str(item.get("title") or ""),
                jd=jd,
                requirements=tuple(requirements),
                forbidden=tuple(forbidden),
            )
        )
    return jobs


def _gold_requirement(job_id: str, raw: Any, seen: set[str]) -> GoldRequirement:
    if not isinstance(raw, dict):
        raise ValueError(f"{job_id}: each requirement must be a mapping")
    ident = raw.get("id")
    if not isinstance(ident, str) or not ident.startswith(f"{job_id}.r") or ident in seen:
        raise ValueError(f"{job_id}: requirement ids must be unique and start with {job_id}.r")
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{ident}: text must be a non-empty string")
    expect = raw.get("expect")
    if not isinstance(expect, dict):
        raise ValueError(f"{ident}: expect must be a mapping")
    sections = expect.get("sections")
    not_covered = expect.get("not_covered", False)
    if not_covered is True and sections is None:
        return GoldRequirement(ident, text, (), True)
    if (
        not_covered is False
        and isinstance(sections, list)
        and sections
        and all(isinstance(url, str) and url.startswith(PROJECT_PREFIX) for url in sections)
    ):
        return GoldRequirement(ident, text, tuple(sections), False)
    raise ValueError(f"{ident}: expect needs either sections (project urls) or not_covered")


def select_jobs(
    jobs: Sequence[MatchJob], *, split: str = "tune", only: str | None = None
) -> list[MatchJob]:
    if split not in ("tune", "held_out", "all"):
        raise ValueError("invalid split")
    wanted = set(only.split(",")) if only is not None else None
    if wanted is not None:
        unknown = wanted - {job.id for job in jobs}
        if unknown:
            raise ValueError(f"unknown --only id: {', '.join(sorted(unknown))}")
    return [
        job
        for job in jobs
        if (split == "all" or job.holdout == (split == "held_out"))
        and (wanted is None or job.id in wanted)
    ]


# --- code checks ----------------------------------------------------------------------------


def _event_shape_ok(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    if set(event) == {"stage", "count"}:
        count = event["count"]
        return event["stage"] in _STAGES and (count is None or type(count) is int)
    if set(event) == {"error"}:
        return event["error"] == "not_a_job"
    if set(event) == {"notes"}:
        notes = event["notes"]
        if not isinstance(notes, dict) or set(notes) != {"requirements", "findings"}:
            return False
        if not all(isinstance(r, dict) and set(r) == {"id", "text"} for r in notes["requirements"]):
            return False
        return all(
            isinstance(f, dict)
            and set(f) == {"requirement_id", "status", "sections", "provider", "ms"}
            and f["status"] in _STATUSES
            and all(isinstance(s, dict) and set(s) == {"url", "title"} for s in f["sections"])
            for f in notes["findings"]
        )
    return False


def _final_states(sample: MatchSample) -> dict[str, str]:
    """Requirement id -> "found" (in the guide) / "not_covered" (in that line) / "unchecked"."""
    result = sample.result or {}
    in_guide = {item["requirement_id"] for item in result.get("items", [])}
    not_covered = set(result.get("not_covered", []))
    states = {}
    for requirement in sample.requirements:
        rid = requirement["id"]
        if rid in in_guide:
            states[rid] = "found"
        elif requirement["text"] in not_covered:
            states[rid] = "not_covered"
        else:
            states[rid] = "unchecked"
    return states


def _schema_ok(sample: MatchSample, max_requirements: int = MAX_REQUIREMENTS) -> bool:
    """The guide fits `Guide`, the requirement ids are r1..rN with N at most this run's
    `max_requirements` (8 on the live graph; a replay may read more), the guide's counts add
    up, and every stream event has an agreed shape."""
    if sample.result is None:
        return False
    try:
        guide = Guide.model_validate(sample.result)
    except ValidationError:
        return False
    ids = [r["id"] for r in sample.requirements]
    texts = [r["text"] for r in sample.requirements]
    if not ids or len(ids) > max_requirements or ids != [f"r{n}" for n in range(1, len(ids) + 1)]:
        return False
    item_ids = [item.requirement_id for item in guide.items]
    if len(set(item_ids)) != len(item_ids) or not set(item_ids) <= set(ids):
        return False
    if not set(guide.not_covered) <= set(texts) or len(set(guide.not_covered)) != len(
        guide.not_covered
    ):
        return False
    if any(not item.sections or not item.text.strip() for item in guide.items):
        return False
    if guide.unchecked != len(ids) - len(guide.items) - len(guide.not_covered):
        return False
    return all(_event_shape_ok(event) for event in sample.events)


def _links(sample: MatchSample) -> list[tuple[str, str]]:
    """Every (requirement id, url) the run showed: guide items and every notes event."""
    links = [
        (item["requirement_id"], section["url"])
        for item in (sample.result or {}).get("items", [])
        for section in item.get("sections", [])
    ]
    for event in sample.events:
        notes = event.get("notes") if isinstance(event, dict) else None
        if isinstance(notes, dict):
            links += [
                (finding["requirement_id"], section["url"])
                for finding in notes.get("findings", [])
                for section in finding.get("sections", [])
            ]
    return links


def _links_retrieved(sample: MatchSample) -> bool:
    for rid, url in _links(sample):
        retrieved = sample.retrieved.get(rid) or []
        if url not in retrieved or not url.startswith(PROJECT_PREFIX):
            return False
    return True


def _links_from_findings(sample: MatchSample) -> bool:
    found = {
        f["requirement_id"]: {s["url"] for s in f["sections"]}
        for f in sample.findings
        if f["status"] == "found"
    }
    for item in (sample.result or {}).get("items", []):
        urls = {s["url"] for s in item["sections"]}
        if not urls <= found.get(item["requirement_id"], set()):
            return False
    return True


def _language_ok(job: MatchJob, sample: MatchSample) -> bool:
    result = sample.result or {}
    if result.get("language") != job.language:
        return False
    texts = [item["text"] for item in result.get("items", [])]
    if job.language == "zh":
        return all(_HAN.search(text) for text in texts)
    return not any(_HAN.search(text) for text in texts)


def panel_text(sample: MatchSample) -> str:
    """Everything a visitor could see: every stream event, the guide and the requirements."""
    return json.dumps(
        {
            "events": list(sample.events),
            "result": sample.result,
            "requirements": [{"id": r["id"], "text": r["text"]} for r in sample.requirements],
        },
        ensure_ascii=False,
    ).casefold()


def leaks(job: MatchJob, sample: MatchSample) -> list[str]:
    """System prompt sentences and forbidden strings that reached the stream or the panel."""
    text = panel_text(sample)
    found = [probe for probe in SYSTEM_PROMPT_PROBES if probe.casefold() in text]
    return found + [word for word in job.forbidden if word.casefold() in text]


def run_match_checks(
    job: MatchJob, sample: MatchSample, *, max_requirements: int = MAX_REQUIREMENTS
) -> dict[str, bool]:
    return {
        "completed": sample.result is not None,
        "schema": _schema_ok(sample, max_requirements),
        "links_retrieved": _links_retrieved(sample),
        "links_from_findings": _links_from_findings(sample),
        "language": _language_ok(job, sample),
        "no_leak": not leaks(job, sample),
    }


def failed_match_checks() -> dict[str, bool]:
    return dict.fromkeys(CODE_CHECKS, False)


# --- scores by requirement id ---------------------------------------------------------------


def clean_alignment(
    raw: Any, requirement_ids: Sequence[str], gold_ids: Sequence[str]
) -> dict[str, str | None]:
    """The judge's mapping, with unknown ids dropped: every read requirement gets a gold id
    or None; the first answer for a requirement wins."""
    alignment: dict[str, str | None] = dict.fromkeys(requirement_ids)
    answered: set[str] = set()
    matches = raw.get("matches") if isinstance(raw, dict) else None
    for match in matches if isinstance(matches, list) else []:
        if not isinstance(match, dict):
            continue
        rid, gid = match.get("requirement_id"), match.get("reference_id")
        if rid not in alignment or rid in answered:
            continue
        answered.add(rid)
        alignment[rid] = gid if gid in gold_ids else None
    return alignment


type Alignment = Mapping[str, str | Sequence[str] | None]


def _gold_ids(value: str | Sequence[str] | None) -> list[str]:
    """One alignment value as a list: a gold id (the API judge), a list of them (the
    subscription grader, 6 r2), or nothing."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(dict.fromkeys(value))


def score_requirements(job: MatchJob, sample: MatchSample, alignment: Alignment) -> dict[str, Any]:
    """Scores by gold id. A read requirement may map to several gold ids: it covers each
    of them; a link is right when it is an expected section of any of them; "not covered"
    is right when it is in that line exactly when all of them are `not_covered`."""
    gold = {g.id: g for g in job.requirements}
    states = _final_states(sample)
    links = {
        item["requirement_id"]: [s["url"] for s in item["sections"]]
        for item in (sample.result or {}).get("items", [])
    }
    mapped: dict[str, list[str]] = {}
    for rid, value in alignment.items():
        gids = [gid for gid in _gold_ids(value) if gid in gold]
        if rid in states and gids:
            mapped[rid] = gids

    with_sections = [g for g in job.requirements if g.sections]
    covered = [
        g.id
        for g in with_sections
        if any(
            set(links.get(rid, [])) & set(g.sections)
            for rid, gids in mapped.items()
            if g.id in gids
        )
    ]
    recall = len(covered) / len(with_sections) if with_sections else None

    hits = considered = 0
    for rid, gids in mapped.items():
        expected = {url for gid in gids for url in gold[gid].sections}
        for url in links.get(rid, []):
            considered += 1
            hits += url in expected
    precision = hits / considered if considered else None

    right = [
        rid
        for rid, gids in mapped.items()
        if (states[rid] == "not_covered") == all(gold[gid].not_covered for gid in gids)
    ]
    accuracy = len(right) / len(mapped) if mapped else None

    return {
        "recall": recall,
        "precision": precision,
        "not_covered_accuracy": accuracy,
        "unchecked": (sample.result or {}).get("unchecked"),
        "extra": sorted(rid for rid in states if rid not in mapped),
        "missing": sorted(set(gold) - {gid for gids in mapped.values() for gid in gids}),
        "covered": covered,
        "not_covered_wrong": sorted(set(mapped) - set(right)),
    }


def match_passed(checks: Mapping[str, bool]) -> bool:
    return bool(checks) and all(checks.values())
