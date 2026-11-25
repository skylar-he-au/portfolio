"""Subscription grading of Q&A runs: export the grading tasks, read the grades back, score
from them (docs/evals/README.md, "订阅判分", Q&A).

From step 10 the Q&A eval judges two metrics, the ones with a pass line (SPEC "第 10 步评测"):
Faithfulness (is every statement of the answer supported by the retrieved sections) and
ContextRecall (is every statement of the reference answer found in them). Subagents on the
subscription judge them following docs/evals/GRADING.md; code does the scoring. Nothing
here calls a model, reads the database or looks at git: every text comes from the run rows
(`sections[].text`, what the answer saw) and the question set (the reference answers).

Files, one directory per run (`<grading dir>/<run_id>/`), as for job match
(`evals.grading`):
- `tasks.jsonl`, one task per line:
  `{"task_id": "<run_id>/<question>/<repeat>/<kind>", "kind", "sample": {"id", "repeat",
  "category"}, "payload"}`. One task per answer and metric, never per sentence: the
  grader breaks the text into statements itself (as RAGAS does), with the whole answer in
  view so that "it" and "the project" are resolved. Kinds:
  - `qa_faith`: `question`, `response` (the answer without its [n] markers), `contexts`
    (every retrieved section, page and heading first).
  - `qa_recall`: `question`, `reference` (the question set's reference answer), `contexts`.
- `tasks.meta.json`: how the export was made.
- `grades.jsonl`: `{"task_id", "grader", "statements": [{"text", "supported", "reason"}]}`
  for both kinds. Score = supported / statements; no statements is None (RAGAS's nan).

Which rows get which tasks follows `metrics_for` (evals/checks.py) with `METRICS` as it is
now: an old run that asked RAGAS for four metrics is graded on the two.

Reading back is `evals.grading.read_grades`: the last line for a task wins, a task id not
in `tasks.jsonl` is dropped, a line that is not JSON, has the wrong shape or a task nobody
graded is a grading error, never a 0; one error marks the run "incomplete".
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.qa.graph import PRIVATE_FALLBACK
from evals.checks import Question, metrics_for, sample_passed
from evals.grading import (
    Grades,
    Problem,
    Task,
    grading_block,
    load_grades,
    read_jsonl,
    run_started,
    write_tasks,
)
from evals.pricing import POOLS
from evals.types import METRICS

QA_KINDS = ("qa_faith", "qa_recall")
METRIC_OF = {"qa_faith": "faithfulness", "qa_recall": "context_recall"}
KIND_OF = {metric: kind for kind, metric in METRIC_OF.items()}
_CITATION_MARKS = re.compile(r"\s*\[\d{1,3}(?:\s*,\s*\d{1,3})*\]")

type Row = Mapping[str, Any]


def task_id(run_id: str, question_id: str, repeat: int, kind: str) -> str:
    return f"{run_id}/{question_id}/{repeat}/{kind}"


def qa_context(section: Mapping[str, Any]) -> str:
    """A retrieved section as the grader sees it: the page and heading the answer model saw
    (app/qa/prompts.py `_sections_block`), then the text."""
    page = str(section["url"]).split("#", 1)[0]
    return f'Skylar\'s page {page}, section "{section["heading"]}":\n{section["text"]}'


def strip_citations(answer: str) -> str:
    """The answer without its [n] markers: the grader judges against all the sections
    together, as RAGAS did."""
    return _CITATION_MARKS.sub("", answer).strip()


def requested(row: Row, question: Question) -> tuple[str, ...]:
    """The metrics this row is judged on (none for an error row)."""
    if row.get("status") != "ok":
        return ()
    return tuple(m for m in metrics_for(question, str(row.get("answer") or "")) if m in METRICS)


def _contexts(row: Row) -> list[str]:
    sections = row.get("sections") or []
    missing = [s.get("url") for s in sections if not isinstance(s.get("text"), str)]
    if missing:
        raise ValueError(
            f"row {row['id']}/{row['repeat']} has sections without text: {', '.join(missing)}"
        )
    return [qa_context(section) for section in sections]


def row_tasks(row: Row, question: Question) -> list[Task]:
    """The grading tasks of one row. A row whose sections lack their text raises: never a
    task with a hole."""
    if list(row.get("turns") or []) != list(question.turns):
        raise ValueError(f"row {row['id']} asks other turns than the question set")
    metrics = requested(row, question)
    if not metrics:
        return []
    run_id, repeat = str(row["run_id"]), int(row["repeat"])
    sample = {"id": question.id, "repeat": repeat, "category": question.category}
    asked = str(row.get("scored_question") or question.scored_question or question.turns[-1])
    contexts = _contexts(row)
    tasks: list[Task] = []
    if "faithfulness" in metrics:
        tasks.append(
            {
                "task_id": task_id(run_id, question.id, repeat, "qa_faith"),
                "kind": "qa_faith",
                "sample": sample,
                "payload": {
                    "question": asked,
                    "response": strip_citations(str(row["answer"])),
                    "contexts": contexts,
                },
            }
        )
    if "context_recall" in metrics and question.reference:
        tasks.append(
            {
                "task_id": task_id(run_id, question.id, repeat, "qa_recall"),
                "kind": "qa_recall",
                "sample": sample,
                "payload": {
                    "question": asked,
                    "reference": question.reference,
                    "contexts": contexts,
                },
            }
        )
    return tasks


def _by_id(rows: Sequence[Row], questions: Sequence[Question]) -> dict[str, Question]:
    by_id = {q.id: q for q in questions}
    unknown = sorted({str(r["id"]) for r in rows} - by_id.keys())
    if unknown:
        raise ValueError(f"rows not in the question set: {', '.join(unknown)}")
    return by_id


def export_tasks(rows: Sequence[Row], questions: Sequence[Question]) -> list[Task]:
    by_id = _by_id(rows, questions)
    tasks = [task for row in rows for task in row_tasks(row, by_id[str(row["id"])])]
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task ids: the rows repeat a question and pass")
    return tasks


# --- grades ---------------------------------------------------------------------------------


def _statements_problem(grade: Mapping[str, Any], task: Task) -> str | None:
    statements = grade.get("statements")
    if not isinstance(statements, list):
        return "statements must be a list"
    for statement in statements:
        if (
            not isinstance(statement, dict)
            or not isinstance(statement.get("text"), str)
            or not statement["text"].strip()
            or type(statement.get("supported")) is not bool
        ):
            return "each statement needs text (non-empty string) and supported (true/false)"
    return None


PROBLEMS: dict[str, Problem] = dict.fromkeys(QA_KINDS, _statements_problem)


def load_qa_grades(directory: Path) -> Grades:
    return load_grades(directory, problems=PROBLEMS, required=QA_KINDS)


def statement_score(grade: Mapping[str, Any]) -> float | None:
    statements = grade["statements"]
    if not statements:
        return None
    return sum(s["supported"] for s in statements) / len(statements)


def score_row(row: Row, question: Question, grades: Grades) -> dict[str, Any]:
    """The row with `ragas` from the grades (the judged metrics, None when not graded) and
    the fields `build_summary` needs. The run's own RAGAS scores, if any, move to
    `ragas_api`."""
    run_id, repeat = str(row["run_id"]), int(row["repeat"])
    metrics = requested(row, question)
    scores: dict[str, float | None] = dict.fromkeys(METRICS)
    errors: list[str] = []
    for metric in metrics:
        tid = task_id(run_id, question.id, repeat, KIND_OF[metric])
        if tid not in grades.tasks:
            continue  # no reference answer: no recall task, as RAGAS skipped it
        if tid in grades.errors:
            errors.append(f"{KIND_OF[metric]}: {grades.errors[tid]}")
        grade = grades.grades.get(tid)
        scores[metric] = None if grade is None else statement_score(grade)
    checks = row.get("code_checks") or {}
    answer = str(row.get("answer") or "")
    return {
        **row,
        "ragas": scores if metrics else None,
        "ragas_api": row.get("ragas"),
        "requested_metrics": [
            m for m in metrics if task_id(run_id, question.id, repeat, KIND_OF[m]) in grades.tasks
        ],
        "passed": row.get("status") == "ok" and sample_passed(checks),
        "accept_fixed": question.accept_fixed,
        "fixed_answer": answer.strip() == PRIVATE_FALLBACK,
        "grading_errors": errors,
    }


def score_rows(
    rows: Sequence[Row], questions: Sequence[Question], grades: Grades
) -> list[dict[str, Any]]:
    by_id = _by_id(rows, questions)
    return [score_row(row, by_id[str(row["id"])], grades) for row in rows]


# --- command line ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _export(args: argparse.Namespace) -> int:
    from evals.run import load_questions

    rows = read_jsonl(args.run)
    if not rows:
        print("no rows in the run", file=sys.stderr)
        return 2
    run_id = str(rows[0]["run_id"])
    try:
        tasks = export_tasks(rows, load_questions(args.question_set))
    except ValueError as error:
        print(f"cannot export: {error}", file=sys.stderr)
        return 1
    meta = {
        "run_id": run_id,
        "kind": "qa",
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "section_texts": "run rows",
        "question_set_sha256": _sha256(args.question_set),
        "metrics": list(METRICS),
    }
    path = write_tasks(tasks, args.out / run_id, meta, kinds=QA_KINDS)
    counts = {kind: sum(t["kind"] == kind for t in tasks) for kind in QA_KINDS}
    print(json.dumps({"tasks": str(path), **counts}))
    return 0


def _run_date(run_id: str) -> str:
    try:
        return run_started(run_id).date().isoformat()
    except ValueError:  # a run id without the start time (tests, hand-made runs)
        return datetime.now(UTC).date().isoformat()


class LedgerMismatch(ValueError):
    """The ledger's rows for a run do not add up to what the run's rows say it cost."""


def ledger_pools(path: Path, run_id: str, rows_total: float) -> dict[str, float]:
    """What each spending pool paid for `run_id`, from the run's ledger (`evals.cost`
    writes one line per question, repeat and pool). The run rows only keep the total, so
    the summary's `cost_pools_usd` comes from here; the two must agree within US$0.000001
    (the summary's rounding), else the ledger is not this run's."""
    pools = dict.fromkeys(POOLS, 0.0)
    found = False
    with path.open(newline="", encoding="utf-8") as ledger:
        for line in csv.DictReader(ledger):
            if line.get("run_id") != run_id:
                continue
            pool = line.get("pool")
            if pool not in pools:
                raise LedgerMismatch(f"{path}: unknown pool {pool!r} for {run_id}")
            pools[pool] += float(line["cost_usd"])
            found = True
    if not found:
        raise LedgerMismatch(f"{path}: no lines for {run_id}")
    if abs(sum(pools.values()) - rows_total) > 1e-6:
        raise LedgerMismatch(
            f"{path}: {run_id} costs {sum(pools.values()):.6f} in the ledger, "
            f"{rows_total:.6f} in the run rows"
        )
    return pools


def _score(args: argparse.Namespace) -> int:
    from evals.run import load_questions
    from evals.summary import build_summary, write_summary

    rows = read_jsonl(args.run)
    run_id = str(rows[0]["run_id"])
    meta_path = args.run.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    pools = None
    if args.ledger is not None:
        try:
            pools = ledger_pools(args.ledger, run_id, sum(float(r["cost_usd"]) for r in rows))
        except (OSError, LedgerMismatch) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    questions = load_questions(args.question_set)
    directory = args.grading_dir
    grades = load_qa_grades(directory)
    for warning in grades.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    scored = score_rows(rows, questions, grades)
    (directory / "scored.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8"
    )
    route = str(meta.get("route") or "default")
    grader = ",".join(grades.graders) or None
    # The answer and check models come from the rows; run rows do not record the embedding
    # model, so it is given on the command line (`--embed-model`, None when left out).
    repeats = int(meta.get("repeats", 1))
    summary = build_summary(
        scored,
        run_id=run_id,
        date=_run_date(run_id),
        prompt_version=args.prompt_version or run_id[16:].removesuffix(f"-{route}"),
        route=route,
        split=str(meta.get("split", "all")),
        repeats=repeats,
        questions=int(meta.get("questions", len({r["id"] for r in rows}))),
        models={"judge": grader, "embed": args.embed_model},
        grader=grader,
        grading=grading_block(grades),
        cost_pools=pools,
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
    export.add_argument("--set", type=Path, required=True, dest="question_set")
    export.add_argument("--out", type=Path, required=True, help="grading dir (<out>/<run_id>/)")
    score = commands.add_parser("score", help="read grades.jsonl back and score the run")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--set", type=Path, required=True, dest="question_set")
    score.add_argument("--grading-dir", type=Path, required=True)
    score.add_argument("--summary-out", type=Path, help="default: the grading directory")
    score.add_argument("--prompt-version")
    score.add_argument("--embed-model", help="the run's embedding model, for the summary")
    score.add_argument(
        "--ledger", type=Path,
        help="the run's spend ledger: adds cost_pools_usd (real money, credit) to the summary",
    )  # fmt: skip
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return _export(args) if args.command == "export" else _score(args)


if __name__ == "__main__":
    raise SystemExit(main())
