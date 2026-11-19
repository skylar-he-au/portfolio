"""Subscription grading of job-match runs: export the grading tasks, read the grades back,
score from them (docs/evals/README.md, "订阅判分").

Judging moved from the API (Sonnet through `ClaudeAligner` and RAGAS) to subagents on the
subscription that follow docs/evals/GRADING.md and answer in JSON; code does the scoring.
Nothing here calls a model.

Files, one directory per run (`<grading dir>/<run_id>/`):
- `tasks.jsonl`, one task per line:
  `{"task_id": "<run_id>/<job>/<repeat>/<kind>[/<n>]", "kind", "sample": {"id", "repeat",
  "candidate"}, "payload"}`. Kinds:
  - `faith`: the whole guide (`response`) and every section it links to (`contexts`, as
    the writer saw them: page and heading first, `judge_context`), plus the question RAGAS
    got (`question`). Faithfulness = supported statements / statements; no statements is
    None (RAGAS's nan), not 0.
  - `align`: the read requirements and the item's gold requirements; one read requirement
    may map to several gold ids.
  - `pair`: one (requirement, section) pair the check judged (every link of a `found`
    finding before the check), numbered `n` from 1 in requirement then link order.
- `tasks.meta.json`: how the export was made (content hash, where section texts came from).
- `grades.jsonl`, written by the graders: `{"task_id", "grader", ...}` with `statements`
  (`faith`), `map` (`align`) or `shows` (`pair`).

Reading back: the last line for a task id wins (a warning per repeat); a task id not in
`tasks.jsonl` is dropped (a warning); a line that is not JSON, has the wrong shape, or a
task nobody graded is a grading error, never a 0. One error on a required kind (`faith`,
`align`) marks the whole run "incomplete": it cannot be declared over the line (A6).
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.match_checks import MatchJob, MatchSample, load_match_set, score_requirements

KINDS = ("faith", "align", "pair")
REQUIRED = ("faith", "align")
SITE_ROOT = Path(__file__).resolve().parents[2]
CONTENT_DIR = SITE_ROOT / "content" / "projects"
# What an old run's section texts depend on: the pages, how they are cut, how they are read.
SOURCE_PATHS = ("content/projects", "api/app/ingest.py", "api/app/rag")

type Row = Mapping[str, Any]
type Task = dict[str, Any]


# --- content --------------------------------------------------------------------------------


def content_hash(content_dir: Path = CONTENT_DIR) -> str:
    """sha256 over every file under `content_dir`: relative path and bytes, in path order."""
    digest = hashlib.sha256()
    for path in sorted(p for p in content_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(content_dir).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(len(data).to_bytes(8, "big") + data)
    return f"sha256:{digest.hexdigest()}"


def run_started(run_id: str) -> datetime:
    """The start of a run from its id (`%Y%m%dT%H%M%S-...`, UTC)."""
    return datetime.strptime(run_id[:15], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def changed_since(since: datetime, root: Path = SITE_ROOT) -> list[str]:
    """Commits that touched the section sources after `since` (`git log`), one line each."""
    command = ["git", "log", f"--since={since.isoformat()}", "--format=%h %cI %s", "--"]
    # fixed arguments, no shell; git from PATH like every other dev command
    out = subprocess.run(  # noqa: S603
        [*command, *SOURCE_PATHS],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


# --- tasks ----------------------------------------------------------------------------------


def task_id(run_id: str, job_id: str, repeat: int, kind: str, n: int | None = None) -> str:
    base = f"{run_id}/{job_id}/{repeat}/{kind}"
    return base if n is None else f"{base}/{n}"


def _sample_of(row: Row) -> MatchSample:
    return MatchSample(
        events=tuple(row.get("events") or ()),
        requirements=tuple(row.get("requirements") or ()),
        findings=tuple(row.get("findings") or ()),
        retrieved=dict(row.get("retrieved") or {}),
        result=row.get("result"),
        checked_findings=tuple(row.get("checked_findings") or row.get("findings") or ()),
        check_verdicts=tuple(row.get("check_verdicts") or ()),
    )


def check_pairs(row: Row) -> list[tuple[str, str]]:
    """Every (requirement id, url) the check judged: each link of a `found` finding before
    the check, in requirement then link order (the graph's `check_pairs`)."""
    return [
        (finding["requirement_id"], section["url"])
        for finding in row.get("findings") or []
        if finding.get("status") == "found"
        for section in finding.get("sections") or []
    ]


def cited_urls(row: Row) -> list[str]:
    items = (row.get("result") or {}).get("items") or []
    return list(dict.fromkeys(s["url"] for item in items for s in item["sections"]))


def needed_urls(rows: Iterable[Row], *, pair_repeats: int | None = None) -> set[str]:
    """Every section an export of these rows shows a grader."""
    urls: set[str] = set()
    for row in rows:
        if row.get("status") != "ok":
            continue
        urls.update(cited_urls(row))
        if pair_repeats is None or row["repeat"] <= pair_repeats:
            urls.update(url for _, url in check_pairs(row))
    return urls


def row_tasks(
    row: Row,
    job: MatchJob,
    section_texts: Mapping[str, str],
    *,
    candidate: str,
    pairs: bool = True,
) -> list[Task]:
    """The grading tasks of one row (none for an error row). `section_texts` must hold
    every section the tasks show (KeyError otherwise: never a task with a hole)."""
    from evals.match_run import faith_input

    if row.get("status") != "ok":
        return []
    run_id, repeat = str(row["run_id"]), int(row["repeat"])
    sample = {"id": job.id, "repeat": repeat, "candidate": candidate}
    requirements = list(row.get("requirements") or [])
    tasks: list[Task] = []

    value = faith_input(_sample_of(row), section_texts)
    if value is not None:
        contexts = [section_texts[url] for url in cited_urls(row)]
        tasks.append(
            {
                "task_id": task_id(run_id, job.id, repeat, "faith"),
                "kind": "faith",
                "sample": sample,
                "payload": {
                    "question": value.user_input,
                    "response": value.response,
                    "contexts": contexts,
                },
            }
        )
    if requirements:
        tasks.append(
            {
                "task_id": task_id(run_id, job.id, repeat, "align"),
                "kind": "align",
                "sample": sample,
                "payload": {
                    "read": [{"id": r["id"], "text": r["text"]} for r in requirements],
                    "reference": [{"id": g.id, "text": g.text} for g in job.requirements],
                },
            }
        )
    if pairs:
        text_of = {r["id"]: r["text"] for r in requirements}
        for n, (rid, url) in enumerate(check_pairs(row), start=1):
            tasks.append(
                {
                    "task_id": task_id(run_id, job.id, repeat, "pair", n),
                    "kind": "pair",
                    "sample": sample,
                    "payload": {
                        "requirement_id": rid,
                        "requirement": text_of.get(rid, ""),
                        "url": url,
                        "section": section_texts[url],
                    },
                }
            )
    return tasks


def candidate_of(row: Row) -> str:
    return str(row.get("candidate") or row.get("route") or "default")


def export_tasks(
    rows: Sequence[Row],
    jobs: Sequence[MatchJob],
    *,
    section_texts: Callable[[Row], Mapping[str, str]],
    pair_repeats: int | None = None,
) -> list[Task]:
    """Every row's tasks, in row order. `pair_repeats` limits `pair` tasks to the first
    passes (None: all)."""
    by_id = {job.id: job for job in jobs}
    tasks: list[Task] = []
    for row in rows:
        job = by_id.get(str(row["id"]))
        if job is None:
            raise ValueError(f"row {row['id']} is not in the match set")
        tasks += row_tasks(
            row,
            job,
            section_texts(row),
            candidate=candidate_of(row),
            pairs=pair_repeats is None or int(row["repeat"]) <= pair_repeats,
        )
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task ids: the rows repeat an item and pass")
    return tasks


def write_tasks(tasks: Sequence[Task], directory: Path, meta: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tasks.jsonl"
    path.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8"
    )
    counts = {kind: sum(t["kind"] == kind for t in tasks) for kind in KINDS}
    (directory / "tasks.meta.json").write_text(
        json.dumps({**meta, "tasks": counts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# --- grades ---------------------------------------------------------------------------------


@dataclass
class Grades:
    """The grades of one export, checked against its tasks."""

    tasks: dict[str, Task]
    grades: dict[str, dict[str, Any]] = field(default_factory=dict)  # task id -> valid grade
    errors: dict[str, str] = field(default_factory=dict)  # task id -> why it has no grade
    warnings: list[str] = field(default_factory=list)

    @property
    def graders(self) -> list[str]:
        return sorted({str(g["grader"]) for g in self.grades.values()})

    def required_errors(self) -> dict[str, str]:
        return {tid: why for tid, why in self.errors.items() if self.tasks[tid]["kind"] in REQUIRED}

    @property
    def status(self) -> str:
        return "incomplete" if self.required_errors() else "complete"

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            kind: {
                "tasks": sum(t["kind"] == kind for t in self.tasks.values()),
                "graded": sum(self.tasks[tid]["kind"] == kind for tid in self.grades),
                "errors": sum(self.tasks[tid]["kind"] == kind for tid in self.errors),
            }
            for kind in KINDS
        }


def _faith_problem(grade: Mapping[str, Any], task: Task) -> str | None:
    statements = grade.get("statements")
    if not isinstance(statements, list):
        return "statements must be a list"
    for statement in statements:
        if (
            not isinstance(statement, dict)
            or not isinstance(statement.get("text"), str)
            or type(statement.get("supported")) is not bool
        ):
            return "each statement needs text (string) and supported (true/false)"
    return None


def _align_problem(grade: Mapping[str, Any], task: Task) -> str | None:
    mapping = grade.get("map")
    if not isinstance(mapping, dict):
        return "map must be an object"
    read = {r["id"] for r in task["payload"]["read"]}
    gold = {g["id"] for g in task["payload"]["reference"]}
    if set(mapping) != read:
        return "map must have exactly the read requirement ids"
    for value in mapping.values():
        if not isinstance(value, list) or not all(v in gold for v in value):
            return "each map value must be a list of reference ids"
    return None


def _pair_problem(grade: Mapping[str, Any], task: Task) -> str | None:
    return None if type(grade.get("shows")) is bool else "shows must be true or false"


_PROBLEMS = {"faith": _faith_problem, "align": _align_problem, "pair": _pair_problem}


def read_grades(tasks: Sequence[Task], lines: Iterable[str]) -> Grades:
    """Check `grades.jsonl` lines against the tasks."""
    out = Grades({t["task_id"]: t for t in tasks})
    latest: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            grade = json.loads(line)
        except json.JSONDecodeError:
            out.warnings.append(f"line {number}: not JSON, skipped")
            continue
        tid = grade.get("task_id") if isinstance(grade, dict) else None
        if not isinstance(tid, str):
            out.warnings.append(f"line {number}: no task_id, skipped")
            continue
        if tid not in out.tasks:
            out.warnings.append(f"line {number}: {tid} is not in tasks.jsonl, dropped")
            continue
        if tid in latest:
            out.warnings.append(f"line {number}: {tid} graded again, the last one counts")
        latest[tid] = grade
    for tid, task in out.tasks.items():
        grade = latest.get(tid)
        if grade is None:
            out.errors[tid] = "not graded"
            continue
        if not isinstance(grade.get("grader"), str) or not grade["grader"]:
            out.errors[tid] = "grader must be a non-empty string"
            continue
        problem = _PROBLEMS[task["kind"]](grade, task)
        if problem is not None:
            out.errors[tid] = problem
            continue
        out.grades[tid] = grade
    return out


def load_grades(directory: Path) -> Grades:
    tasks = read_jsonl(directory / "tasks.jsonl")
    path = directory / "grades.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return read_grades(tasks, lines)


# --- scores ---------------------------------------------------------------------------------


def faithfulness(grade: Mapping[str, Any]) -> float | None:
    statements = grade["statements"]
    if not statements:
        return None
    return sum(s["supported"] for s in statements) / len(statements)


def check_kept(row: Row) -> dict[tuple[str, str], bool | None]:
    """Whether the check passed each pair: from its verdicts when the row has them, else
    from the findings after the check, else (rows before 6 r2) from the guide's links.
    None when none of these is known (no guide)."""
    pairs = check_pairs(row)
    verdicts = row.get("check_verdicts")
    if verdicts:
        shown = {(v["requirement_id"], v["url"]): v.get("shows") is True for v in verdicts}
        return {pair: shown.get(pair, False) for pair in pairs}
    checked = row.get("checked_findings")
    if checked:
        kept = {
            (f["requirement_id"], s["url"])
            for f in checked
            if f.get("status") == "found"
            for s in f.get("sections") or []
        }
        return {pair: pair in kept for pair in pairs}
    result = row.get("result")
    if result is None:
        return dict.fromkeys(pairs)
    linked = {
        (item["requirement_id"], s["url"])
        for item in result.get("items") or []
        for s in item["sections"]
    }
    return {pair: pair in linked for pair in pairs}


def pair_counts(pairs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Judged pairs against the grader: right, wrongly removed (the section shows the
    requirement, the check dropped it), wrongly kept (it does not, the check passed it)."""
    judged = [p for p in pairs if p["shows"] is not None and p["kept"] is not None]
    return {
        "judged": len(judged),
        "right": sum(p["shows"] == p["kept"] for p in judged),
        "wrongly_removed": sum(p["shows"] and not p["kept"] for p in judged),
        "wrongly_kept": sum(not p["shows"] and p["kept"] for p in judged),
    }


def score_row(row: Row, job: MatchJob, grades: Grades) -> dict[str, Any]:
    """The row with alignment, scores, Faithfulness and pairs from the grades."""
    run_id, repeat = str(row["run_id"]), int(row["repeat"])
    graded = dict(row)
    errors: list[str] = []

    def grade_of(kind: str, n: int | None = None) -> dict[str, Any] | None:
        tid = task_id(run_id, job.id, repeat, kind, n)
        if tid not in grades.tasks:
            return None
        if tid in grades.errors:
            errors.append(f"{kind}{'' if n is None else f'/{n}'}: {grades.errors[tid]}")
        return grades.grades.get(tid)

    align = grade_of("align")
    alignment = None if align is None else {rid: list(gids) for rid, gids in align["map"].items()}
    faith = grade_of("faith")
    kept = check_kept(row)
    pairs = []
    for n, (rid, url) in enumerate(check_pairs(row), start=1):
        if task_id(run_id, job.id, repeat, "pair", n) not in grades.tasks:
            continue
        grade = grade_of("pair", n)
        pairs.append(
            {
                "requirement_id": rid,
                "url": url,
                "shows": None if grade is None else grade["shows"],
                "kept": kept[(rid, url)],
            }
        )
    graded.update(
        alignment=alignment,
        scores=None if alignment is None else score_requirements(job, _sample_of(row), alignment),
        faithfulness=None if faith is None else faithfulness(faith),
        faith_scored=task_id(run_id, job.id, repeat, "faith") in grades.tasks,
        pairs=pairs,
        pair_counts=pair_counts(pairs),
        judge_errors=[],
        grading_errors=errors,
    )
    return graded


def score_rows(
    rows: Sequence[Row], jobs: Sequence[MatchJob], grades: Grades
) -> list[dict[str, Any]]:
    by_id = {job.id: job for job in jobs}
    return [score_row(row, by_id[str(row["id"])], grades) for row in rows]


def grading_block(grades: Grades) -> dict[str, Any]:
    """What a summary says about the grading: names and numbers only."""
    return {
        "status": grades.status,
        "kinds": grades.counts(),
        "warnings": len(grades.warnings),
    }


# --- command line ---------------------------------------------------------------------------


async def _db_section_texts(database_url: str, urls: set[str]) -> dict[str, str]:
    from app.db import Database, make_pool
    from app.rag.types import Section
    from evals.match_run import judge_context

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
    return {
        (section := Section(page, sid, heading, text)).url: judge_context(section)
        for page, sid, heading, text in found
    }


class SourcesChanged(Exception):
    """The local database may no longer hold the sections an old run saw."""


def old_run_texts(
    run_id: str,
    urls: set[str],
    fetch: Callable[[set[str]], dict[str, str]],
    *,
    changes: Callable[[datetime], list[str]] = changed_since,
) -> dict[str, str]:
    """Section texts of a run that did not store them, from the local database, after the
    checks of contract 7: nothing that shapes the sections changed since the run, and every
    section the run used is in the database (the version number is not compared: every
    ingest makes a new one)."""
    commits = changes(run_started(run_id))
    if commits:
        raise SourcesChanged("section sources changed since the run: " + "; ".join(commits))
    texts = fetch(urls)
    missing = sorted(urls - texts.keys())
    if missing:
        raise SourcesChanged(f"{len(missing)} sections not in the database: {', '.join(missing)}")
    return texts


def _export(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.run)
    if not rows:
        print("no rows in the run", file=sys.stderr)
        return 2
    run_id = str(rows[0]["run_id"])
    jobs = load_match_set(args.match_set)
    stored = all("section_texts" in row for row in rows if row.get("status") == "ok")
    meta: dict[str, Any] = {
        "run_id": run_id,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pair_repeats": args.pair_repeats,
        "content_hash": content_hash(),
    }
    if stored:
        meta["section_texts"] = "run rows"
        texts_of: Callable[[Row], Mapping[str, str]] = lambda row: row.get("section_texts") or {}  # noqa: E731
    else:
        from app.config import get_settings

        database_url = get_settings().database_url.get_secret_value()
        urls = needed_urls(rows, pair_repeats=args.pair_repeats)
        try:
            texts = old_run_texts(
                run_id, urls, lambda u: asyncio.run(_db_section_texts(database_url, u))
            )
        except SourcesChanged as error:
            print(f"cannot export from the database: {error}", file=sys.stderr)
            return 1
        meta.update(section_texts="local database", sections=len(texts))
        texts_of = lambda row: texts  # noqa: E731
    tasks = export_tasks(rows, jobs, section_texts=texts_of, pair_repeats=args.pair_repeats)
    path = write_tasks(tasks, args.out / run_id, meta)
    print(json.dumps({"tasks": str(path), **{k: v for k, v in meta.items() if k != "run_id"}}))
    return 0


def _run_date(run_id: str) -> str:
    try:
        return run_started(run_id).date().isoformat()
    except ValueError:  # a run id without the start time (tests, hand-made runs)
        return datetime.now(UTC).date().isoformat()


def prompt_of(run_id: str, route: str) -> str:
    """The prompt version in a run id `<start>-<prompt version>-<route>`."""
    return run_id[16:].removesuffix(f"-{route}")


def _score(args: argparse.Namespace) -> int:
    from evals.summary import build_match_summary, write_summary

    rows = read_jsonl(args.run)
    run_id = str(rows[0]["run_id"])
    meta_path = args.run.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    jobs = load_match_set(args.match_set)
    route = str(meta.get("route") or rows[0].get("route") or "default")
    # Rows before 6 r2 kept no models in meta: take them from the API-judged summary.
    old_summary = args.run.with_suffix(".json")
    if "models" not in meta and old_summary.exists():
        old = json.loads(old_summary.read_text(encoding="utf-8"))
        meta["models"] = {
            k: v for k, v in (old.get("models") or {}).items() if not k.startswith("judge")
        }
    directory = args.grading_dir or (args.out / run_id)
    grades = load_grades(directory)
    for warning in grades.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    scored = score_rows(rows, jobs, grades)
    (directory / "scored.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8"
    )
    grader = ",".join(grades.graders) or None
    summary = build_match_summary(
        scored,
        run_id=run_id,
        date=_run_date(run_id),
        prompt_version=args.prompt_version
        or meta.get("prompt_version")
        or prompt_of(run_id, route),
        route=route,
        split=str(meta.get("split", "all")),
        repeats=int(meta.get("repeats", max(int(r["repeat"]) for r in rows))),
        questions=int(meta.get("jobs", len({r["id"] for r in rows}))),
        models={**(meta.get("models") or {}), "judge_align": grader, "judge_faithfulness": grader},
        cost_pools=meta.get("run_cost"),
        grader=grader,
        grading=grading_block(grades),
    )
    out = write_summary(summary, args.summary_out or directory)
    print(json.dumps({"status": grades.status, "summary": str(out), **grades.counts()}))
    if grades.status != "complete":
        for tid, why in sorted(grades.required_errors().items()):
            print(f"grading error: {tid}: {why}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write tasks.jsonl for a run (no model calls)")
    export.add_argument("--run", type=Path, required=True, help="the run's .jsonl")
    export.add_argument("--set", type=Path, required=True, dest="match_set")
    export.add_argument("--out", type=Path, required=True, help="grading dir (<out>/<run_id>/)")
    export.add_argument("--pair-repeats", type=int, help="pair tasks for passes 1..N only")
    score = commands.add_parser("score", help="read grades.jsonl back and score the run")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--set", type=Path, required=True, dest="match_set")
    score.add_argument("--out", type=Path, help="grading dir (<out>/<run_id>/)")
    score.add_argument("--grading-dir", type=Path, help="the run's own grading directory")
    score.add_argument("--summary-out", type=Path, help="default: the grading directory")
    score.add_argument("--prompt-version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export":
        return _export(args)
    if args.grading_dir is None and args.out is None:
        print("score needs --grading-dir or --out", file=sys.stderr)
        return 2
    return _score(args)


if __name__ == "__main__":
    raise SystemExit(main())
