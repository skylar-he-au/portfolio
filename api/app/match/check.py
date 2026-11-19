"""The `match-v1` check, the code half: which pairs to ask about and what the verdicts change.

The check runs after the match agents and before the guide is written, so the guide's
sentences can only mention sections the check kept. One model call (Gemini 3.5 Flash-Lite
first) checks every link of every `found` finding at once; each pair is one requirement and
the one section linked for it, nothing else. A link the check does not pass is removed; a
finding left with no link becomes "couldn't be checked" (`unchecked`), never "not covered":
the pages may well show it, we just could not confirm. No retry. A pair the check gives no
verdict for counts as not passed. The checked findings go to `checked_findings`; the graph
sends them to the panel once, and the write agent sees only them.
"""

from collections.abc import Mapping, Sequence

from app.match.prompts import Verdict
from app.match.state import Finding, Requirement
from app.rag.types import Section

type Pair = tuple[int, Requirement, Section]


def check_pairs(
    findings: Sequence[Finding],
    requirements: Sequence[Requirement],
    sections_by_url: Mapping[str, Section],
) -> list[Pair]:
    """Numbered (requirement, section) pairs, one per link of every `found` finding.

    A link whose section text is not in `sections_by_url` cannot be checked and gets no
    pair, so `apply_verdicts` removes it.
    """
    by_id = {r.id: r for r in requirements}
    pairs: list[Pair] = []
    for finding in findings:
        requirement = by_id.get(finding.requirement_id)
        if finding.status != "found" or requirement is None:
            continue
        for ref in finding.sections:
            section = sections_by_url.get(ref.url)
            if section is not None:
                pairs.append((len(pairs) + 1, requirement, section))
    return pairs


def apply_verdicts(
    findings: Sequence[Finding],
    pairs: Sequence[Pair],
    verdicts: Sequence[Verdict],
) -> list[Finding]:
    """The findings with every link the check did not pass removed: a `found` finding keeps
    the links that passed, one left with none becomes `unchecked` with no links. Findings
    that were not `found` are returned unchanged."""
    shown = {v.n for v in verdicts if v.shows}
    passed = {(req.id, sec.url) for n, req, sec in pairs if n in shown}
    out: list[Finding] = []
    for finding in findings:
        if finding.status != "found":
            out.append(finding)
            continue
        kept = [ref for ref in finding.sections if (finding.requirement_id, ref.url) in passed]
        if kept == finding.sections:
            out.append(finding)
        elif kept:
            out.append(finding.model_copy(update={"sections": kept}))
        else:
            out.append(finding.model_copy(update={"status": "unchecked", "sections": []}))
    return out


def fail_all(findings: Sequence[Finding]) -> list[Finding]:
    """The check itself could not run (both providers failed or ran out of time): no link is
    confirmed, so every `found` finding becomes "couldn't be checked"."""
    return [
        f.model_copy(update={"status": "unchecked", "sections": []}) if f.status == "found" else f
        for f in findings
    ]
