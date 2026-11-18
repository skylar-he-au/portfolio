"""The shared notes of the job-match graph: the fields each agent reads and writes.

`MatchState` is the notes. Who may read and write which field is fixed in
`graph.NOTE_ACCESS` and enforced by code (a node that writes a field it does not own
raises `NoteAccessError`). `findings` is written by the parallel match agents at the same
time; `merge_findings` is its reducer.

Only validated fields leave the graph (`notes_snapshot`): requirement ids and texts,
statuses, `page#section_id` links, provider and ms. The models' reasoning never enters the
state; the job ad (`jd`) stays in memory for this one run.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel

from app.rag.types import Section

logger = logging.getLogger(__name__)

MAX_REQUIREMENTS = 8
MAX_REQUIREMENT_CHARS = 120
REQUIREMENT_ID = re.compile(r"r[1-8]")  # MAX_REQUIREMENTS

Stage = Literal["reading", "matching", "writing", "checking"]
Status = Literal["found", "not_found", "unchecked"]


class Requirement(BaseModel):
    id: str  # r1..r8, numbered by code
    text: str  # at most 120 characters, shown as plain text
    query: str  # English search phrase for retrieval


class SectionRef(BaseModel):
    url: str  # page#section_id, built by code from a retrieved section
    title: str  # the section heading


class Finding(BaseModel):
    requirement_id: str
    status: Status
    sections: list[SectionRef]
    provider: str | None
    ms: int


class GuideItem(BaseModel):
    requirement_id: str
    text: str
    sections: list[SectionRef]


class Guide(BaseModel):
    items: list[GuideItem]
    not_covered: list[str]  # requirement texts the pages don't show
    unchecked: int  # requirements that couldn't be checked
    top8: bool
    language: str


def requirement_number(requirement_id: str) -> int:
    return int(requirement_id[1:])


def merge_findings(left: Sequence[Finding], right: Sequence[Finding]) -> list[Finding]:
    """Reducer for `findings`: one finding per requirement id, sorted r1, r2, ...

    A finding whose id is not of the form `r<n>`, or whose id already has a finding (from
    an earlier write or earlier in the same write), is dropped and logged. Ids that are
    well formed but not in `requirements` cannot be seen here (a reducer only gets this
    channel); `graph` drops those before writing the guide (`known_findings`).
    """
    merged = {f.requirement_id: f for f in left}
    for finding in right:
        rid = finding.requirement_id
        if not REQUIREMENT_ID.fullmatch(rid):
            logger.warning("match finding rejected: malformed requirement id")
            continue
        if rid in merged:
            logger.warning("match finding rejected: duplicate for %s", rid)
            continue
        merged[rid] = finding
    return sorted(merged.values(), key=lambda f: requirement_number(f.requirement_id))


def known_findings(
    requirements: Sequence[Requirement], findings: Sequence[Finding]
) -> list[Finding]:
    """The findings whose requirement id is one of `requirements` (others are logged)."""
    ids = {r.id for r in requirements}
    kept = []
    for finding in findings:
        if finding.requirement_id in ids:
            kept.append(finding)
        else:
            logger.warning("match finding rejected: unknown requirement id")
    return kept


@dataclass
class NotesBoard:
    """What the panel sees while the parallel match agents run.

    LangGraph merges the parallel `findings` writes only when the whole step ends; the
    board lets each match agent send the whole notes (with the findings finished so far)
    the moment it finishes. It holds only validated models and merges with the same
    reducer, so it can never show more than the state will.
    """

    requirements: list[Requirement]
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings = merge_findings(self.findings, [finding])

    def snapshot(self) -> dict[str, Any]:
        return notes_snapshot(self.requirements, self.findings)


def notes_snapshot(
    requirements: Sequence[Requirement], findings: Sequence[Finding]
) -> dict[str, Any]:
    """The notes as the stream carries them: only fields checked by code."""
    return {
        "requirements": [{"id": r.id, "text": r.text} for r in requirements],
        "findings": [
            {
                "requirement_id": f.requirement_id,
                "status": f.status,
                "sections": [{"url": s.url, "title": s.title} for s in f.sections],
                "provider": f.provider,
                "ms": f.ms,
            }
            for f in findings
        ],
    }


class MatchInput(TypedDict):
    jd: str


class MatchState(TypedDict):
    jd: str  # in memory only
    requirements: list[Requirement]  # written by read
    findings: Annotated[list[Finding], merge_findings]  # written by match (parallel)
    result: Guide | None  # written by write, changed by check
    # Internal fields (not part of the notes the panel shows):
    language: NotRequired[str]  # written by read
    top8: NotRequired[bool]  # written by read
    board: NotRequired[NotesBoard]  # written by read
    # requirement id -> sections found for it; None when the search itself failed
    retrieved: NotRequired[dict[str, list[Section] | None]]  # written by search
    match_deadline: NotRequired[float]  # time.monotonic() deadline, written by search


class MatchTask(TypedDict):
    """What one match agent gets through `Send`: its own requirement and nothing else."""

    requirement: Requirement
    sections: list[Section] | None  # None: the search failed
    deadline: float
    board: NotesBoard
