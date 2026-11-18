"""The `match-v1` check, the code half: which pairs to ask about and what the verdicts change.

One model call (Gemini 3.5 Flash-Lite first) checks every link of the guide at once; each
pair is one requirement and the one section linked for it, nothing else. A link the check
does not pass is removed; a guide item left with no link becomes "couldn't be checked"
(`unchecked`), never "not covered": the pages may well show it, we just could not confirm.
No retry. A pair the check gives no verdict for counts as not passed.
"""

from collections.abc import Mapping, Sequence

from app.match.prompts import Verdict
from app.match.state import Guide, Requirement
from app.rag.types import Section


def check_pairs(
    guide: Guide,
    requirements: Sequence[Requirement],
    sections_by_url: Mapping[str, Section],
) -> list[tuple[int, Requirement, Section]]:
    """Numbered (requirement, section) pairs, one per link in the guide.

    A link whose section text is not in `sections_by_url` cannot be checked and gets no
    pair, so `apply_verdicts` removes it.
    """
    by_id = {r.id: r for r in requirements}
    pairs: list[tuple[int, Requirement, Section]] = []
    for item in guide.items:
        requirement = by_id.get(item.requirement_id)
        if requirement is None:
            continue
        for ref in item.sections:
            section = sections_by_url.get(ref.url)
            if section is not None:
                pairs.append((len(pairs) + 1, requirement, section))
    return pairs


def apply_verdicts(
    guide: Guide,
    pairs: Sequence[tuple[int, Requirement, Section]],
    verdicts: Sequence[Verdict],
) -> Guide:
    """The guide with every link the check did not pass removed; items left without links
    are dropped and counted in `unchecked`."""
    shown = {v.n for v in verdicts if v.shows}
    passed = {(req.id, sec.url) for n, req, sec in pairs if n in shown}
    items = []
    failed = 0
    for item in guide.items:
        kept = [ref for ref in item.sections if (item.requirement_id, ref.url) in passed]
        if kept:
            items.append(item.model_copy(update={"sections": kept}))
        else:
            failed += 1
    return guide.model_copy(update={"items": items, "unchecked": guide.unchecked + failed})


def fail_all(guide: Guide) -> Guide:
    """The check itself could not run (both providers failed or ran out of time): no link is
    confirmed, so every item becomes "couldn't be checked"."""
    return guide.model_copy(update={"items": [], "unchecked": guide.unchecked + len(guide.items)})
