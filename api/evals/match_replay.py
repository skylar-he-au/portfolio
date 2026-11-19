"""Replay one job-match step on a stored run, with other models (6 r2, contract 7).

The upstream comes from one pass of a stored run (default: the v4 final run, pass 1): the
read requirements, the retrieved sections, the findings before the check and the findings
after it. Only the chosen step (`--step read|match|check|write`) calls a model, through the
same functions the graph uses (`app.match.graph.read_step`, `match_one`, `check_step`,
`write_step`); nothing upstream runs again, nothing downstream runs at all. A row keeps
only this step's products, and only the scores this step's products allow are computed
later (`evals.step_scores`).

- read: the job ad again (`--max-requirements` 8 or 12); `align` tasks for the graders.
- match: one match agent per stored requirement on its stored sections, in parallel.
- check: one check call over the stored findings before the check; its verdict per pair.
  The pairs are the stored run's pairs, so its `pair` grades are reused.
- write: the guide from the stored findings after the check (rebuilt from the stored guide
  for runs before 6 r2, which did not store them); `faith` tasks for the graders.

Before any call (contract 7): no commit to `content/projects`, `api/app/ingest.py` or
`api/app/rag` since the stored run started (`git log`), every section the replay needs is in
the local database (not the version number: every ingest makes a new one), and the
content hash goes into the run's meta. Every call goes through the eval meter: its cost
is in the ledger (docs/evals/spend-step6-r2.csv) and each item starts only if both pools
have the step's worst case left (`MatchMeter.check_room`).

`--dry-run` makes no call and needs no database: it prints each option's models in order
and its worst case, and checks contract 12 (spent + this batch's worst + the passes still
planned, against the total caps). `--batch FILE` gives several options at once (one line
of flags each) for the dry run.
"""

import argparse
import asyncio
import csv
import json
import shlex
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from app.match import models as match_models
from app.match.prompts import PROMPT_VERSION
from app.match.state import MAX_REQUIREMENTS
from evals.match_checks import MatchJob, MatchSample, load_match_set, run_match_checks, select_jobs
from evals.match_run import (
    STEPS,
    CallLog,
    MatchMeter,
    Pools,
    RoomMeter,
    _add,
    _pools,
    fallback_nodes,
    graph_worst,
    judge_context,
)
from evals.pricing import POOLS
from evals.types import CapReached, UnknownModel

V4_RUN = "20261119T030433-match-v4-default"
REPLAY_MAX_REQUIREMENTS = (MAX_REQUIREMENTS, 12)
# Grading kinds a replay of each step needs from the subscription graders; the others are
# the stored run's own grades (r2e): align for match/check/write, pair for check.
STEP_KINDS: Mapping[str, tuple[str, ...]] = {
    "read": ("align",),
    "match": (),
    "check": (),
    "write": ("faith",),
}
FUTURE_PASSES = 4  # contract 12: the combination once, then the final run three times
ITEM_TIMEOUT_S = 150.0

type Row = dict[str, Any]


# --- options --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Option:
    """One thing to try: a step, its models in order, and (read) how many requirements."""

    step: match_models.Step
    models: match_models.StepModels
    max_requirements: int = MAX_REQUIREMENTS

    @property
    def slug(self) -> str:
        """`haiku-5-5-low`, `luna-none`, `haiku-5-5-off-max12`: the first model, short."""
        first = self.models.first
        name = first.model.removeprefix("claude-").removeprefix("gemini-")
        name = name.removeprefix("gpt-6-").replace(".", "")
        slug = f"{name}-{first.effort}"
        return slug if self.max_requirements == MAX_REQUIREMENTS else f"{slug}-max12"

    def describe(self) -> str:
        line = f"{' -> '.join(str(c) for c in self.models.choices)}"
        line += f" (max {self.models.max_output_tokens})"
        if self.step == "read":
            line += f", up to {self.max_requirements} requirements"
        return line

    @property
    def providers(self) -> list[str]:
        return [c.provider for c in self.models.choices]


def make_option(
    step: str,
    base: match_models.MatchModels,
    models: Sequence[str] = (),
    fallbacks: Sequence[str] = (),
    max_output: Sequence[str] = (),
    max_requirements: int = MAX_REQUIREMENTS,
) -> Option:
    """`--step` with `--model` / `--fallback` / `--max-output` (`apply_overrides`); a flag
    for another step is an error: only this step runs, so it would do nothing."""
    if step not in STEPS:
        raise ValueError(f"step must be one of {', '.join(STEPS)}")
    for text in (*models, *fallbacks, *max_output):
        named = text.partition("=")[0].strip()
        if named != step:
            raise ValueError(f"{text!r}: this replay runs only the {step} step")
    if max_requirements not in REPLAY_MAX_REQUIREMENTS:
        raise ValueError(f"max requirements must be one of {REPLAY_MAX_REQUIREMENTS}")
    if max_requirements != MAX_REQUIREMENTS and step != "read":
        raise ValueError("--max-requirements is for the read step only")
    name = cast(match_models.Step, step)
    out = match_models.apply_overrides(base, models, fallbacks, max_output)
    return Option(name, out[name], max_requirements)


def replay_run_id(started: datetime, option: Option) -> str:
    return f"{started:%Y%m%dT%H%M%S}-{PROMPT_VERSION}-replay-{option.step}-{option.slug}"


# --- worst case -----------------------------------------------------------------------------


def _step_calls(step: str) -> tuple[int, int]:
    """(calls, input tokens per call) of one step at its largest, the inputs of
    `app.api.match.match_worst_steps` (a test keeps the two equal)."""
    from app.api.match import MAX_JD_CHARS, SECTION_CHARS_ESTIMATE
    from app.guard.usage import estimate_input_tokens
    from app.match import prompts
    from app.match.graph import MAX_LINKS, MAX_QUERY_CHARS, MAX_SECTIONS
    from app.match.state import MAX_REQUIREMENT_CHARS, Requirement
    from app.rag.types import Section

    marker = "0" * 16
    requirement = Requirement(
        id="r1", text="x" * MAX_REQUIREMENT_CHARS, query="x" * MAX_QUERY_CHARS
    )
    section = Section(
        "/projects/" + "x" * 30, "x" * 40, "x" * 80, "x" * SECTION_CHARS_ESTIMATE, 0.0
    )
    if step == "read":
        return 1, estimate_input_tokens(prompts.read_messages("x" * MAX_JD_CHARS, marker))
    if step == "match":
        messages = prompts.match_messages(requirement, [section] * MAX_SECTIONS, marker)
        return MAX_REQUIREMENTS, estimate_input_tokens(messages)
    if step == "check":
        pairs = [(n, requirement, section) for n in range(1, MAX_REQUIREMENTS * MAX_LINKS + 1)]
        return 1, estimate_input_tokens(prompts.check_messages(pairs, marker))
    numbered = [(n, requirement, [section] * MAX_LINKS) for n in range(1, MAX_REQUIREMENTS + 1)]
    return 1, estimate_input_tokens(prompts.write_messages(numbered, "en", marker))


def step_worst(option: Option, *, first_only: bool = False) -> Pools:
    """One item's worst case for this step, in eval pools: every call at the largest input
    and the step's output cap, at the price of every model in the order (a fallback may
    follow a first call that already cost something), or the first only (`first_only`,
    the app's reservation)."""
    from evals.pricing import cost_usd

    calls, input_tokens = _step_calls(option.step)
    choices = option.models.choices[:1] if first_only else option.models.choices
    parts = [
        _pools(
            c.model,
            calls * cost_usd(c.model, input_tokens, option.models.max_output_tokens),
        )
        for c in choices
    ]
    return _add(*parts)


def ledger_totals(path: Path) -> Pools:
    totals = dict.fromkeys(POOLS, 0.0)
    try:
        with path.open(newline="", encoding="utf-8") as ledger:
            for row in csv.DictReader(ledger):
                with suppress(TypeError, ValueError):
                    if row.get("pool") in totals:
                        totals[row["pool"]] += float(row.get("cost_usd") or "")
    except FileNotFoundError:
        pass
    return totals


def budget_check(
    options: Sequence[Option],
    jobs: int,
    *,
    spent: Pools,
    caps: Mapping[str, tuple[float, float]],
    future: Pools,
) -> dict[str, Any]:
    """Contract 12 for a batch of replays, each one pass over `jobs` items: spent + the
    batch's worst + `future` (an estimate of the passes still planned) fits each pool's
    total cap. A pass whose worst case is over the pass cap is only a warning: the meter
    checks the pass cap item by item against what was really spent (`check_room`), so
    such a pass runs unless the real spend gets close; one item's worst case over the pass
    cap is a problem (no item could start)."""
    lines = []
    problems: list[str] = []
    warnings: list[str] = []
    batch = dict.fromkeys(POOLS, 0.0)
    for option in options:
        per_item = step_worst(option)
        per_pass = {pool: per_item[pool] * jobs for pool in POOLS}
        batch = _add(batch, per_pass)
        for pool in POOLS:
            name = f"{option.step} {option.slug}: {pool}"
            if per_item[pool] > caps[pool][0]:
                problems.append(
                    f"{name} worst item {per_item[pool]:.6f} > pass cap {caps[pool][0]}"
                )
            elif per_pass[pool] > caps[pool][0]:
                warnings.append(
                    f"{name} worst pass {per_pass[pool]:.6f} > pass cap {caps[pool][0]}"
                )
        lines.append(
            {
                "step": option.step,
                "option": option.slug,
                "order": option.describe(),
                "uses_google": "google" in option.providers,
                "worst_per_item": _round(per_item),
                "worst_per_pass": _round(per_pass),
            }
        )
    need = _add(spent, batch, future)
    for pool in POOLS:
        if need[pool] > caps[pool][1]:
            problems.append(
                f"{pool}: spent {spent[pool]:.6f} + batch {batch[pool]:.6f} + later"
                f" {future[pool]:.6f} = {need[pool]:.6f} > total cap {caps[pool][1]}"
            )
    return {
        "options": lines,
        "jobs": jobs,
        "batch_worst": _round(batch),
        "spent": _round(spent),
        "later_estimate": _round(future),
        "total_needed": _round(need),
        "caps": {pool: {"pass": caps[pool][0], "total": caps[pool][1]} for pool in POOLS},
        "fits": not problems,
        "problems": problems,
        "warnings": warnings,
    }


def later_per_item(
    given: str | None, source_meta: Path, worst: Callable[[], Pools]
) -> tuple[Pools, str]:
    """The estimate per item of the full passes still planned (contract 12: an estimate,
    not the worst case): `real=X,credit=Y` as given; else the stored run's measured cost
    per item (its meta `run_cost` / `samples`; for v4 the credit part includes its API
    judge, so it is on the high side); else the default models' worst case."""
    if given:
        out = dict.fromkeys(POOLS, 0.0)
        for part in given.split(","):
            pool, sep, value = part.partition("=")
            if not sep or pool.strip() not in POOLS:
                raise ValueError(f"{part!r}: expected real=X or credit=Y")
            out[pool.strip()] = float(value)
        return out, "the given estimate per item"
    try:
        meta = json.loads(source_meta.read_text(encoding="utf-8"))
        samples = int(meta["samples"])
        cost = {pool: float(meta["run_cost"][pool]) / samples for pool in POOLS}
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        return worst(), "the default models' worst case (no measured cost)"
    return cost, f"the measured cost per item of {meta.get('run_id', source_meta.stem)}"


def _round(pools: Mapping[str, float]) -> dict[str, float]:
    return {pool: round(pools.get(pool, 0.0), 6) for pool in POOLS}


# --- the stored run -------------------------------------------------------------------------


class SourceError(Exception):
    """The stored run cannot feed this replay."""


def read_rows(path: Path) -> list[Row]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def source_rows(rows: Sequence[Row], jobs: Sequence[MatchJob], repeat: int) -> dict[str, Row]:
    """Each job's row of pass `repeat`; every job must have an ok one."""
    by_id = {str(r["id"]): r for r in rows if int(r["repeat"]) == repeat}
    missing = [j.id for j in jobs if by_id.get(j.id, {}).get("status") != "ok"]
    if missing:
        raise SourceError(f"no ok row in pass {repeat} for: {', '.join(missing)}")
    return {j.id: by_id[j.id] for j in jobs}


def checked_findings_of(row: Row) -> list[dict[str, Any]]:
    """The findings after the check. Rows before 6 r2 did not store them; the guide shows
    them: a requirement in the guide was `found` with exactly the guide's links (the
    writer never changes links), one in the "not covered" line `not_found`, any other
    `unchecked` (the check removed every link, or the match agent could not decide)."""
    stored = row.get("checked_findings")
    if stored:
        return list(stored)
    result = row.get("result") or {}
    in_guide = {item["requirement_id"]: item["sections"] for item in result.get("items") or []}
    not_covered = set(result.get("not_covered") or [])
    before = {f["requirement_id"]: f for f in row.get("findings") or []}
    out = []
    for requirement in row.get("requirements") or []:
        rid = requirement["id"]
        old = before.get(rid, {})
        if rid in in_guide:
            status, sections = "found", list(in_guide[rid])
        elif requirement["text"] in not_covered:
            status, sections = "not_found", []
        else:
            status, sections = "unchecked", []
        out.append(
            {
                "requirement_id": rid,
                "status": status,
                "sections": sections,
                "provider": old.get("provider"),
                "ms": int(old.get("ms") or 0),
            }
        )
    return out


def needed_urls(rows: Mapping[str, Row], step: str) -> set[str]:
    """The sections a replay of `step` shows a model: every retrieved one (match, check:
    the check looks a link's section up among the retrieved ones; write: the same)."""
    if step == "read":
        return set()
    return {
        url
        for row in rows.values()
        for urls in (row.get("retrieved") or {}).values()
        for url in urls or []
    }


type SectionFetch = Callable[[set[str]], Awaitable[dict[str, Any]]]


async def fetch_sections(database_url: str, urls: set[str]) -> dict[str, Any]:
    """url -> `Section` (score 0: no prompt shows it) of the current content version."""
    from app.db import Database, make_pool
    from app.rag.types import Section

    sql = """
        SELECT s.page, s.section_id, s.heading, s.text
        FROM sections s
        JOIN content_state c ON c.current_version = s.version
        WHERE c.id AND s.page || '#' || s.section_id = ANY(%(urls)s::text[])
    """
    db = Database(make_pool(database_url))
    try:
        async with db.connection() as conn:
            cursor = await conn.execute(sql, {"urls": sorted(urls)})
            found = await cursor.fetchall()
    finally:
        await db.close()
    sections = [Section(page, sid, heading, text) for page, sid, heading, text in found]
    return {s.url: s for s in sections}


async def preflight(
    source_run_id: str,
    urls: set[str],
    fetch: SectionFetch,
    *,
    changes: Callable[[datetime], list[str]] | None = None,
    content: Callable[[], str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Contract 7, before any call: (sections by url, what was checked for the meta)."""
    from evals import grading

    changed = (changes or grading.changed_since)(grading.run_started(source_run_id))
    if changed:
        raise SourceError("section sources changed since the stored run: " + "; ".join(changed))
    sections = await fetch(urls) if urls else {}
    missing = sorted(urls - sections.keys())
    if missing:
        raise SourceError(f"{len(missing)} sections not in the database: {', '.join(missing)}")
    return sections, {
        "sources_changed": [],
        "sections_checked": len(urls),
        "content_hash": (content or grading.content_hash)(),
    }


# --- one item -------------------------------------------------------------------------------


def _models(items: Sequence[dict[str, Any]], model: Any) -> list[Any]:
    return [model.model_validate(item) for item in items]


async def _replay_one(
    option: Option,
    job: MatchJob,
    source: Row,
    deps: Any,
    sections: Mapping[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """This step on one stored item: the row fields it produces."""
    from app.match import graph
    from app.match.state import Finding, Requirement

    if option.step == "read":  # before `retrieved`: a read replay fetches no sections
        update = await graph.read_step(
            job.jd, deps, max_requirements=option.max_requirements, config=config
        )
        return {
            "requirements": [r.model_dump() for r in update["requirements"]],
            "language": update["language"],
            "top8": update["top8"],
        }
    requirements = _models(source.get("requirements") or [], Requirement)
    retrieved = {
        rid: None if urls is None else [sections[u] for u in urls]
        for rid, urls in (source.get("retrieved") or {}).items()
    }
    if option.step == "match":
        deadline = time.monotonic() + deps.stage_timeouts["matching"]
        # no `board`: it keeps r1..r8 only (`state.REQUIREMENT_ID`), and nothing streams here
        findings = await asyncio.gather(
            *(
                graph.match_one(r, retrieved.get(r.id), deps, deadline=deadline, config=config)
                for r in requirements
            )
        )
        return {"findings": [f.model_dump() for f in findings]}
    if option.step == "check":
        result = await graph.check_step(
            requirements,
            _models(source.get("findings") or [], Finding),
            retrieved,
            deps,
            config=config,
        )
        return {
            "findings": list(source.get("findings") or []),
            "checked_findings": [f.model_dump() for f in result.findings],
            "check_verdicts": [v.model_dump() for v in result.verdicts],
        }
    stored = source.get("result") or {}
    checked = checked_findings_of(source)
    guide = await graph.write_step(
        requirements,
        _models(checked, Finding),
        retrieved,
        str(stored.get("language") or "en"),
        bool(stored.get("top8")),
        deps,
        config=config,
    )
    result = guide.model_dump()
    cited = {s["url"] for item in result["items"] for s in item["sections"]}
    return {
        "findings": list(source.get("findings") or []),
        "checked_findings": checked,
        "result": result,
        "section_texts": {url: judge_context(sections[url]) for url in sorted(cited)},
    }


def _upstream(source: Row, step: str) -> dict[str, Any]:
    """What the row keeps of the stored run: what this step was given."""
    if step == "read":
        return {}
    return {
        "requirements": list(source.get("requirements") or []),
        "retrieved": dict(source.get("retrieved") or {}),
    }


async def replay_item(
    option: Option,
    job: MatchJob,
    source: Row,
    *,
    deps: Any,
    sections: Mapping[str, Any],
    meter: RoomMeter,
    worst: Pools,
    run_id: str,
    candidate: str,
    expected: Mapping[str, str],
) -> Row:
    """One item once, retried once on an error. CapReached / UnknownModel propagate (after
    the spend is recorded)."""
    recorded = False
    started = time.monotonic()
    try:
        meter.check_room(worst)
        produced: dict[str, Any] | None = None
        log = CallLog()
        error_name = None
        for attempt in range(2):
            if attempt:
                meter.check_room(worst)
            log = CallLog()
            config = {"callbacks": [*meter.callbacks(), log]}
            try:
                async with asyncio.timeout(ITEM_TIMEOUT_S):
                    produced = await _replay_one(option, job, source, deps, sections, config)
                error_name = None
                break
            except (CapReached, UnknownModel):
                raise
            except Exception as error:  # noqa: BLE001 - retry the item once
                error_name = type(error).__name__
        cost = meter.record(job.id, 1)
        recorded = True
        row: Row = {
            "run_id": run_id,
            "id": job.id,
            "kind": job.kind,
            "held_out": job.holdout,
            "repeat": 1,
            "step": option.step,
            "candidate": candidate,
            "source_repeat": source["repeat"],
            "status": "error" if error_name else "ok",
            "error": error_name,
            "fallback": fallback_nodes(log.calls, expected),
            **_upstream(source, option.step),
            **(produced or {}),
            "calls": log.calls,
            "ms": round((time.monotonic() - started) * 1000),
            "cost_usd": cost,
        }
        if option.step == "write" and produced is not None:
            sample = MatchSample(
                requirements=tuple(row["requirements"]),
                findings=tuple(row["findings"]),
                retrieved=row["retrieved"],
                result=row["result"],
            )
            row["code_checks"] = run_match_checks(job, sample)
        return row
    finally:
        if not recorded:
            with suppress(CapReached, UnknownModel):
                meter.record(job.id, 1)


# --- the run --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayOutcome:
    rows: list[Row]
    status: str
    reason: str | None

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "complete" else 1


async def replay(
    option: Option,
    jobs: Sequence[MatchJob],
    sources: Mapping[str, Row],
    *,
    deps: Any,
    sections: Mapping[str, Any],
    meter: RoomMeter,
    run_id: str,
    out_dir: Path,
    tasks_out: Path,
    meta: Mapping[str, Any],
) -> ReplayOutcome:
    """One pass of `option` over `jobs`. Writes `<out_dir>/<run_id>.jsonl` and `.meta.json`
    and, when the step needs graders, `<tasks_out>/<run_id>/tasks.jsonl`."""
    from evals.grading import export_tasks, write_tasks

    worst = step_worst(option)
    expected = {option.step: option.models.first.provider}
    rows: list[Row] = []
    status, reason = "complete", None
    error_streak = fallback_streak = 0
    out_path = out_dir / f"{run_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meter.start_pass()
    try:
        with out_path.open("a", encoding="utf-8") as output:
            for job in jobs:
                row = await replay_item(
                    option, job, sources[job.id], deps=deps, sections=sections, meter=meter,
                    worst=worst, run_id=run_id, candidate=option.slug, expected=expected,
                )  # fmt: skip
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                rows.append(row)
                print(f"{job.id} {row['status']} {row['cost_usd']:.6f}", flush=True)
                error_streak = error_streak + 1 if row["status"] == "error" else 0
                fallback_streak = fallback_streak + 1 if row["fallback"] else 0
                if error_streak >= 3:
                    status, reason = "incomplete", f"three consecutive errors; last: {row['error']}"
                    break
                if fallback_streak >= 3:
                    # the scores would be the fallback's, not this option's
                    status, reason = "incomplete", "three consecutive fallbacks"
                    break
    except (CapReached, UnknownModel) as error:
        status, reason = "incomplete", f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001 - keep the finished rows after any failure
        status, reason = "incomplete", type(error).__name__
    if status == "incomplete":
        print(f"replay incomplete: {reason}", file=sys.stderr)
    full_meta: dict[str, Any] = {
        **meta,
        "run_id": run_id,
        "kind": "replay",
        "step": option.step,
        "option": option.slug,
        "order": option.describe(),
        "models": [str(c) for c in option.models.choices],
        "max_output_tokens": option.models.max_output_tokens,
        "max_requirements": option.max_requirements,
        "prompt_version": PROMPT_VERSION,
        "status": status,
        "reason": reason,
        "jobs": len(jobs),
        "samples": len(rows),
        "worst_per_item": _round(worst),
        "run_cost": meter.run_cost,
        "expected_providers": expected,
        "grader": "subscription" if STEP_KINDS[option.step] else None,
    }
    try:
        kinds = STEP_KINDS[option.step]
        if kinds and rows:
            tasks = [
                t
                # pairs never: a check replay judges the stored run's pairs, graded there
                for t in export_tasks(
                    rows,
                    jobs,
                    section_texts=lambda row: row.get("section_texts") or {},
                    pair_repeats=0,
                )
                if t["kind"] in kinds
            ]
            full_meta["tasks"] = str(
                write_tasks(
                    tasks,
                    tasks_out / run_id,
                    {
                        "run_id": run_id,
                        "content_hash": full_meta.get("content_hash"),
                        "section_texts": "run rows",
                        "kinds": list(kinds),
                    },
                )
            )
    except Exception as error:  # noqa: BLE001 - an export failure makes the run incomplete
        status, reason = "incomplete", type(error).__name__
        full_meta.update(status=status, reason=reason)
        print(f"replay incomplete: {reason}", file=sys.stderr)
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(full_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ReplayOutcome(rows, status, reason)


# --- command line ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    from evals.match_run import _positive_float

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--set", type=Path, required=True, dest="match_set")
    parser.add_argument("--out", type=Path, required=True, help="runs dir (docs/evals/runs)")
    parser.add_argument("--ledger", type=Path, required=True)
    for flag in ("pass-cap", "total-cap", "credit-pass-cap", "credit-total-cap"):
        parser.add_argument(f"--{flag}", type=_positive_float, required=True)
    parser.add_argument("--split", choices=("tune", "held_out", "all"), default="all")
    parser.add_argument("--only")
    parser.add_argument(
        "--from-run", default=V4_RUN, help="a run id in --out, or a .jsonl path (default: v4)"
    )
    parser.add_argument("--from-pass", type=int, default=1)
    parser.add_argument("--step", choices=STEPS)
    for flag, example in (
        ("model", "check=openai:gpt-6-luna:none"),
        ("fallback", "check= (no fallback) or read=anthropic:claude-haiku-5-5:off"),
        ("max-output", "read=4096"),
    ):
        parser.add_argument(
            f"--{flag}", action="append", default=[], help=f"this step only, e.g. {example}"
        )
    parser.add_argument(
        "--max-requirements", type=int, default=MAX_REQUIREMENTS, choices=REPLAY_MAX_REQUIREMENTS
    )
    parser.add_argument(
        "--tasks-out", type=Path, help="grading dir, tasks go to <dir>/<run_id>/ (default:"
        " ../grading next to --out)"
    )  # fmt: skip
    parser.add_argument("--dry-run", action="store_true", help="no calls: order, worst case")
    parser.add_argument(
        "--batch", type=Path, help="with --dry-run: one option per line (--step ... --model ...)"
    )
    parser.add_argument(
        "--future-passes",
        type=int,
        default=FUTURE_PASSES,
        help="full passes still planned after this batch (default 4: combination 1 + final 3)",
    )
    parser.add_argument(
        "--later-per-item",
        help="estimated cost of one item of those passes, real=X,credit=Y (default: the"
        " stored run's measured cost per item)",
    )
    return parser


def _option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="batch line", add_help=False, exit_on_error=False)
    parser.add_argument("--step", choices=STEPS, required=True)
    for flag in ("model", "fallback", "max-output"):
        parser.add_argument(f"--{flag}", action="append", default=[])
    parser.add_argument("--max-requirements", type=int, default=MAX_REQUIREMENTS)
    return parser


def batch_options(text: str, base: match_models.MatchModels) -> list[Option]:
    options = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            args = _option_parser().parse_args(shlex.split(line))
            option = make_option(
                args.step,
                base,
                args.model,
                args.fallback,
                args.max_output,
                args.max_requirements,
            )
            options.append(option)
        except (argparse.ArgumentError, ValueError) as error:
            raise ValueError(f"batch line {number}: {error}") from None
    if not options:
        raise ValueError("the batch file has no options")
    return options


def _source_path(from_run: str, out: Path) -> Path:
    path = Path(from_run)
    return path if path.suffix == ".jsonl" else out / f"{from_run}.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        jobs = select_jobs(load_match_set(args.match_set), split=args.split, only=args.only)
    except ValueError as error:
        print(f"invalid match set: {error}", file=sys.stderr)
        return 2
    from app.config import get_settings

    settings = get_settings()
    base = match_models.models_for(settings)
    try:
        if args.batch is not None:
            if not args.dry_run:
                raise ValueError("--batch is for --dry-run only: run the options one by one")
            options = batch_options(args.batch.read_text(encoding="utf-8"), base)
        elif args.step is None:
            raise ValueError("--step is required")
        else:
            options = [
                make_option(
                    args.step,
                    base,
                    args.model,
                    args.fallback,
                    args.max_output,
                    args.max_requirements,
                )
            ]
    except (OSError, ValueError) as error:
        print(f"invalid options: {error}", file=sys.stderr)
        return 2
    caps = {
        "real": (args.pass_cap, args.total_cap),
        "credit": (args.credit_pass_cap, args.credit_total_cap),
    }
    if args.dry_run:
        all_jobs = len(load_match_set(args.match_set))
        try:
            per_item, basis = later_per_item(
                args.later_per_item,
                _source_path(args.from_run, args.out).with_suffix(".meta.json"),
                lambda: graph_worst(
                    settings.answer_model, settings.check_model, settings.embed_model, "default"
                ),
            )
        except ValueError as error:
            print(f"invalid --later-per-item: {error}", file=sys.stderr)
            return 2
        future = {pool: per_item[pool] * all_jobs * args.future_passes for pool in POOLS}
        report = budget_check(
            options, len(jobs), spent=ledger_totals(args.ledger), caps=caps, future=future
        )
        report["later"] = f"{args.future_passes} passes x {all_jobs} items at {basis}"
        report["later_per_item"] = _round(per_item)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["fits"] else 1
    option = options[0]
    source_path = _source_path(args.from_run, args.out)
    try:
        stored = read_rows(source_path)
        source_run_id = str(stored[0]["run_id"])
        sources = source_rows(stored, jobs, args.from_pass)
    except (OSError, IndexError, KeyError, ValueError, SourceError) as error:
        print(f"cannot use the stored run: {error}", file=sys.stderr)
        return 2
    database_url = settings.database_url.get_secret_value()
    started = datetime.now(UTC)
    run_id = replay_run_id(started, option)

    async def run() -> int:
        from app.match.graph import build_match_deps

        try:
            sections, checked = await preflight(
                source_run_id,
                needed_urls(sources, option.step),
                lambda urls: fetch_sections(database_url, urls),
            )
        except SourceError as error:
            print(f"preflight failed, no call made: {error}", file=sys.stderr)
            return 1
        models = {**base, option.step: option.models}
        # no search, no embedding: the retrieved sections are the stored run's
        deps = build_match_deps(settings, cast(Any, None), cast(Any, None), models=models)
        meter = MatchMeter(caps, args.ledger, run_id)
        outcome = await replay(
            option,
            jobs,
            sources,
            deps=deps,
            sections=sections,
            meter=meter,
            run_id=run_id,
            out_dir=args.out,
            tasks_out=args.tasks_out or args.out.parent / "grading",
            meta={
                "source_run": source_run_id,
                "source_pass": args.from_pass,
                "split": args.split,
                **checked,
            },
        )
        return outcome.exit_code

    try:
        return asyncio.run(run())
    except Exception as error:  # noqa: BLE001 - setup failures must not expose config or prompts
        print(f"replay setup failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
