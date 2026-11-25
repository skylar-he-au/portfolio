"""Subscription grading of Q&A runs (evals/qa_grading.py, step 10 10g)."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.qa.graph import PRIVATE_FALLBACK
from evals.checks import Question
from evals.grading import read_grades
from evals.qa_grading import (
    PROBLEMS,
    QA_KINDS,
    export_tasks,
    load_qa_grades,
    main,
    qa_context,
    score_rows,
    strip_citations,
)
from evals.summary import build_summary
from evals.types import METRICS, NOT_JUDGED

RUN = "20261118T103944-v9-default"
GRADER = "claude-sonnet-5-5 (subscription, GRADING.md)"
SECTIONS = [
    {"url": "/projects/kite#build", "heading": "How Kite is built", "text": "Kite uses Postgres."},
    {"url": "/projects/kite#tests", "heading": "Tests", "text": "Kite has 40 tests."},
]
ABOUT = [{"url": "/about#work", "heading": "Work", "text": "I am based in Sydney."}]

QUESTIONS = [
    Question(id="kite-db", category="project", turns=("What database?",), reference="Postgres."),
    Question(id="kite-noref", category="project", turns=("Why Kite?",)),
    Question(
        id="kite-zh",
        category="zh",
        turns=("Kite 用什么?", "测试呢?"),
        scored_question="Kite 有多少测试?",
        reference="Kite 有 40 个测试。",
    ),
    Question(
        id="pa-city", category="private_answerable", turns=("Where?",), accept_fixed=True,
        expect_route="private",
    ),
    Question(id="off-poem", category="off_topic", turns=("A poem?",), expect_route="off_topic"),
]  # fmt: skip


def row(qid: str, *, repeat: int = 1, **overrides: Any) -> dict[str, Any]:
    question = next(q for q in QUESTIONS if q.id == qid)
    base: dict[str, Any] = {
        "run_id": RUN,
        "id": qid,
        "category": question.category,
        "held_out": False,
        "repeat": repeat,
        "status": "ok",
        "error": None,
        "turns": list(question.turns),
        "scored_question": question.scored_question or question.turns[-1],
        "answer": "Kite uses Postgres [1]. It has 40 tests [1, 2].",
        "citations": [],
        "sections": SECTIONS,
        "route": "general",
        "nodes": [{"name": "answer", "model": "claude-haiku-5-5", "provider": "anthropic"}],
        "code_checks": {"language": True, "route": True},
        "ragas": {
            "faithfulness": 0.5,
            "response_relevancy": 0.9,
            "context_recall": 1.0,
            "factual_correctness": 0.3,
        },
        "cost_usd": 0.01,
    }
    return {**base, **overrides}


def rows() -> list[dict[str, Any]]:
    return [
        row("kite-db"),
        row("kite-noref"),
        row("kite-zh", answer="Kite 有 40 个测试 [2]。"),
        row("pa-city", answer="I am based in Sydney [1].", sections=ABOUT, route="private"),
        row("pa-city", repeat=2, answer=PRIVATE_FALLBACK, sections=ABOUT, route="private"),
        row("off-poem", answer="No.", sections=[], route="off_topic", ragas=None),
        row("kite-db", repeat=2, status="error", error="Timeout", answer="", sections=[]),
    ]


def ids(tasks: list[dict[str, Any]]) -> list[str]:
    return [t["task_id"] for t in tasks]


# --- export ---------------------------------------------------------------------------------


def test_metrics_are_the_two_with_a_pass_line() -> None:
    assert METRICS == ("faithfulness", "context_recall")
    assert NOT_JUDGED == ("response_relevancy", "factual_correctness")


def test_export_follows_metrics_for() -> None:
    tasks = export_tasks(rows(), QUESTIONS)
    assert ids(tasks) == [
        f"{RUN}/kite-db/1/qa_faith",
        f"{RUN}/kite-db/1/qa_recall",
        f"{RUN}/kite-noref/1/qa_faith",  # no reference answer: no recall, as RAGAS skipped it
        f"{RUN}/kite-zh/1/qa_faith",
        f"{RUN}/kite-zh/1/qa_recall",
        f"{RUN}/pa-city/1/qa_faith",  # private answered from About: faithfulness only
        # pa-city 2 gave the fixed sentence, off-poem is off topic, kite-db 2 is an error
    ]
    assert {t["kind"] for t in tasks} == set(QA_KINDS)


def test_export_payloads() -> None:
    tasks = {t["task_id"]: t for t in export_tasks(rows(), QUESTIONS)}
    faith = tasks[f"{RUN}/kite-db/1/qa_faith"]
    assert faith["sample"] == {"id": "kite-db", "repeat": 1, "category": "project"}
    assert faith["payload"] == {
        "question": "What database?",
        "response": "Kite uses Postgres. It has 40 tests.",
        "contexts": [
            'Skylar\'s page /projects/kite, section "How Kite is built":\nKite uses Postgres.',
            'Skylar\'s page /projects/kite, section "Tests":\nKite has 40 tests.',
        ],
    }
    recall = tasks[f"{RUN}/kite-zh/1/qa_recall"]
    # the scored question, not the last turn; the reference from the question set
    assert recall["payload"]["question"] == "Kite 有多少测试?"
    assert recall["payload"]["reference"] == "Kite 有 40 个测试。"
    assert recall["payload"]["contexts"] == faith["payload"]["contexts"]
    assert tasks[f"{RUN}/kite-zh/1/qa_faith"]["payload"]["response"] == "Kite 有 40 个测试。"


def test_strip_citations_and_context() -> None:
    assert strip_citations("A [1]. B [2, 3].") == "A. B."
    assert qa_context(ABOUT[0]) == 'Skylar\'s page /about, section "Work":\nI am based in Sydney.'


def test_export_refuses_a_hole() -> None:
    broken = [row("kite-db", sections=[{"url": "/projects/kite#build", "heading": "H"}])]
    with pytest.raises(ValueError, match="sections without text"):
        export_tasks(broken, QUESTIONS)
    # a row that needs no task may lack the texts (v9: off-topic and injection rows)
    textless = [row("off-poem", sections=[{"url": "/x#y", "heading": "H"}])]
    assert export_tasks(textless, QUESTIONS) == []


def test_export_refuses_rows_the_set_does_not_match() -> None:
    with pytest.raises(ValueError, match="not in the question set"):
        export_tasks([row("kite-db", id="gone")], QUESTIONS)
    with pytest.raises(ValueError, match="other turns"):
        export_tasks([row("kite-db", turns=["Another question?"])], QUESTIONS)
    with pytest.raises(ValueError, match="duplicate"):
        export_tasks([row("kite-db"), row("kite-db")], QUESTIONS)


# --- reading back ---------------------------------------------------------------------------


def grade(tid: str, *supported: bool, **overrides: Any) -> str:
    statements = [
        {"text": f"Statement {i}.", "supported": s, "reason": "r"} for i, s in enumerate(supported)
    ]
    return json.dumps({"task_id": tid, "grader": GRADER, "statements": statements, **overrides})


def all_grades(tasks: list[dict[str, Any]]) -> list[str]:
    return [grade(t["task_id"], True, True, False, True) for t in tasks]


def read(tasks: list[dict[str, Any]], lines: list[str]) -> Any:
    return read_grades(tasks, lines, problems=PROBLEMS, required=QA_KINDS)


def test_read_back_complete() -> None:
    tasks = export_tasks(rows(), QUESTIONS)
    grades = read(tasks, all_grades(tasks))
    assert grades.status == "complete"
    assert grades.errors == {} and grades.warnings == []
    assert grades.counts() == {
        "qa_faith": {"tasks": 4, "graded": 4, "errors": 0},
        "qa_recall": {"tasks": 2, "graded": 2, "errors": 0},
    }


def test_read_back_missing_line_is_an_error_not_zero() -> None:
    tasks = export_tasks(rows(), QUESTIONS)
    grades = read(tasks, all_grades(tasks)[1:])
    assert grades.status == "incomplete"
    assert grades.errors == {f"{RUN}/kite-db/1/qa_faith": "not graded"}
    scored = score_rows(rows(), QUESTIONS, grades)
    assert scored[0]["ragas"]["faithfulness"] is None
    assert scored[0]["grading_errors"] == ["qa_faith: not graded"]


def test_read_back_extra_and_repeated_lines_warn() -> None:
    tasks = export_tasks(rows(), QUESTIONS)
    first = f"{RUN}/kite-db/1/qa_faith"
    lines = [
        *all_grades(tasks),
        grade(f"{RUN}/kite-db/9/qa_faith", True),  # not exported
        grade(first, False),  # graded again: the last line counts
    ]
    grades = read(tasks, lines)
    assert grades.status == "complete"
    assert len(grades.warnings) == 2
    assert any("not in tasks.jsonl" in w for w in grades.warnings)
    assert any("graded again" in w for w in grades.warnings)
    assert grades.grades[first]["statements"][0]["supported"] is False


@pytest.mark.parametrize(
    ("line", "why"),
    [
        ('{"task_id": "%s", "grader": "g", "statements": "all good"}', "statements must be a list"),
        ('{"task_id": "%s", "grader": "g", "statements": [{"text": "A.", "supported": "yes"}]}',
         "supported"),
        ('{"task_id": "%s", "grader": "g", "statements": [{"text": "", "supported": true}]}',
         "non-empty"),
        ('{"task_id": "%s", "statements": []}', "grader"),
        ('{"task_id": "%s", "grader": "g", "map": {}}', "statements must be a list"),
    ],
)  # fmt: skip
def test_read_back_wrong_shape_is_an_error(line: str, why: str) -> None:
    tasks = export_tasks(rows(), QUESTIONS)
    tid = tasks[0]["task_id"]
    grades = read(tasks, [*all_grades(tasks)[1:], line % tid])
    assert grades.status == "incomplete"
    assert why in grades.errors[tid]


def test_read_back_line_that_is_not_json() -> None:
    tasks = export_tasks(rows(), QUESTIONS)
    grades = read(tasks, ["{not json", *all_grades(tasks)[1:]])
    assert grades.warnings == ["line 1: not JSON, skipped"]
    assert grades.errors == {tasks[0]["task_id"]: "not graded"}


# --- scores ---------------------------------------------------------------------------------


def test_scores_and_summary() -> None:
    tasks = export_tasks(rows(), QUESTIONS)
    lines = all_grades(tasks)  # 3 of 4 supported everywhere
    lines[2] = grade(f"{RUN}/kite-noref/1/qa_faith")  # nothing checkable: no score, not 0
    grades = read(tasks, lines)
    scored = score_rows(rows(), QUESTIONS, grades)
    by_key = {(r["id"], r["repeat"]): r for r in scored}
    assert by_key[("kite-db", 1)]["ragas"] == {"faithfulness": 0.75, "context_recall": 0.75}
    assert by_key[("kite-db", 1)]["ragas_api"]["factual_correctness"] == 0.3  # kept to compare
    assert by_key[("kite-db", 1)]["requested_metrics"] == ["faithfulness", "context_recall"]
    assert by_key[("kite-noref", 1)]["ragas"] == {"faithfulness": None, "context_recall": None}
    assert by_key[("kite-noref", 1)]["requested_metrics"] == ["faithfulness"]
    assert by_key[("pa-city", 1)]["requested_metrics"] == ["faithfulness"]
    assert by_key[("pa-city", 2)]["ragas"] is None
    assert by_key[("pa-city", 2)]["fixed_answer"] is True
    assert by_key[("kite-db", 2)]["passed"] is False  # error row
    summary = build_summary(
        scored, run_id=RUN, date="2026-11-18", prompt_version="v9", route="default",
        split="all", repeats=1, questions=5, models={"judge": GRADER}, grader=GRADER,
        grading={"status": grades.status},
    )  # fmt: skip
    assert summary["not_judged"] == ["response_relevancy", "factual_correctness"]
    assert set(summary["ragas"]["tune"]) == set(METRICS)
    assert summary["ragas"]["tune"]["faithfulness"]["scored"] == 3
    assert summary["ragas"]["tune"]["context_recall"]["mean"] == 0.75
    assert summary["ragas_nulls"] == 1  # kite-noref's empty statement list, nothing else
    assert summary["grader"] == GRADER and summary["grading"] == {"status": "complete"}
    assert summary["models"]["judge"] == GRADER


def test_ragas_nulls_never_count_metrics_not_judged() -> None:
    # A row built elsewhere that still asks for four metrics (an old API run).
    old = {
        **row("kite-db"),
        "requested_metrics": ["faithfulness", "response_relevancy", "context_recall",
                              "factual_correctness"],
        "ragas": {"faithfulness": 1.0, "context_recall": 1.0},
        "passed": True, "accept_fixed": False, "fixed_answer": False,
    }  # fmt: skip
    summary = build_summary(
        [old], run_id=RUN, date="2026-11-18", prompt_version="v9", route="default",
        split="all", repeats=1, questions=1, models={},
    )  # fmt: skip
    assert summary["ragas_nulls"] == 0
    assert "grader" not in summary and "grading" not in summary  # API-judged: as before


# --- command line ---------------------------------------------------------------------------


SET_YAML = """
version: 1
questions:
  - id: kite-db
    category: project
    turns: ["What database?"]
    reference: Postgres.
  - id: off-poem
    category: off_topic
    expect_route: off_topic
    turns: ["A poem?"]
"""


def write_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "runs" / f"{RUN}.jsonl"
    run.parent.mkdir()
    lines = [row("kite-db"), row("off-poem", answer="No.", sections=[], route="off_topic")]
    run.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    run.with_suffix(".meta.json").write_text(
        json.dumps({"run_id": RUN, "split": "all", "repeats": 1, "route": "default",
                    "questions": 2}), encoding="utf-8",
    )  # fmt: skip
    question_set = tmp_path / "qa-set.yaml"
    question_set.write_text(SET_YAML, encoding="utf-8")
    return run, question_set


def test_cli_export_then_score(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run, question_set = write_run(tmp_path)
    out = tmp_path / "grading"
    assert main(["export", "--run", str(run), "--set", str(question_set), "--out", str(out)]) == 0
    directory = out / RUN
    tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text().splitlines()]
    assert ids(tasks) == [f"{RUN}/kite-db/1/qa_faith", f"{RUN}/kite-db/1/qa_recall"]
    meta = json.loads((directory / "tasks.meta.json").read_text())
    assert meta["tasks"] == {"qa_faith": 1, "qa_recall": 1}
    assert meta["section_texts"] == "run rows"
    assert meta["question_set_sha256"].startswith("sha256:")

    score = ["score", "--run", str(run), "--set", str(question_set), "--grading-dir",
             str(directory), "--embed-model", "text-embedding-3-small"]  # fmt: skip
    # nobody graded yet: incomplete, exit 1, every task listed
    assert main(score) == 1
    assert capsys.readouterr().err.count("grading error") == 2

    (directory / "grades.jsonl").write_text(
        grade(tasks[0]["task_id"], True, False) + "\n" + grade(tasks[1]["task_id"], True) + "\n"
    )
    assert main(score) == 0
    summary = json.loads((directory / f"{RUN}.json").read_text())
    assert summary["kind"] == "qa"
    assert summary["prompt_version"] == "v9"
    assert summary["grading"]["status"] == "complete"
    assert summary["grading"]["kinds"]["qa_faith"] == {"tasks": 1, "graded": 1, "errors": 0}
    assert summary["ragas"]["tune"]["faithfulness"]["mean"] == 0.5
    assert summary["ragas"]["tune"]["context_recall"]["mean"] == 1.0
    assert summary["models"]["embed"] == "text-embedding-3-small"
    scored = [json.loads(line) for line in (directory / "scored.jsonl").read_text().splitlines()]
    assert scored[0]["ragas"] == {"faithfulness": 0.5, "context_recall": 1.0}

    # a re-grade of one kind with a changed guide replaces that kind's grades (GRADING.md 5.7)
    (directory / "grades-qa_faith-v2.jsonl").write_text(grade(tasks[0]["task_id"], True) + "\n")
    assert main(score) == 0
    assert load_qa_grades(directory).grades[tasks[0]["task_id"]]["statements"][0]["supported"]
    summary = json.loads((directory / f"{RUN}.json").read_text())
    assert summary["ragas"]["tune"]["faithfulness"]["mean"] == 1.0


def test_cli_export_refuses_a_set_that_does_not_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, question_set = write_run(tmp_path)
    question_set.write_text(SET_YAML.replace("What database?", "Which database?"))
    args = ["export", "--run", str(run), "--set", str(question_set), "--out", str(tmp_path / "g")]
    assert main(args) == 1
    assert "other turns" in capsys.readouterr().err
    assert not (tmp_path / "g").exists()


def test_cli_score_with_ledger_adds_cost_pools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, question_set = write_run(tmp_path)
    out = tmp_path / "grading"
    assert main(["export", "--run", str(run), "--set", str(question_set), "--out", str(out)]) == 0
    directory = out / RUN
    tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text().splitlines()]
    (directory / "grades.jsonl").write_text("".join(g + "\n" for g in all_grades(tasks)))
    rows_total = sum(json.loads(line)["cost_usd"] for line in run.read_text().splitlines())
    ledger = tmp_path / "spend.csv"
    header = "run_id,question_id,repeat,pool,cost_usd,at\n"
    at = "2026-11-25T00:00:00+00:00"
    ledger.write_text(
        header
        + f"{RUN},kite-db,1,real,{rows_total * 0.75!r},{at}\n"
        + f"{RUN},kite-db,1,credit,{rows_total * 0.25!r},{at}\n"
        + f"20261118T000000-v8-default,kite-db,1,real,5.0,{at}\n"  # another run: ignored
    )
    score = ["score", "--run", str(run), "--set", str(question_set),
             "--grading-dir", str(directory)]  # fmt: skip

    assert main(score) == 0  # without --ledger: the total only, as before
    summary = json.loads((directory / f"{RUN}.json").read_text())
    assert "cost_pools_usd" not in summary

    assert main([*score, "--ledger", str(ledger)]) == 0
    summary = json.loads((directory / f"{RUN}.json").read_text())
    assert summary["cost_usd"] == round(rows_total, 6)
    assert summary["cost_pools_usd"] == {
        "real": round(rows_total * 0.75, 6),
        "credit": round(rows_total * 0.25, 6),
    }
    capsys.readouterr()

    # a ledger that does not add up to the rows, or has no lines for the run, is refused
    # and nothing is rewritten
    before = (directory / f"{RUN}.json").read_text()
    ledger.write_text(header + f"{RUN},kite-db,1,real,{rows_total + 0.01!r},{at}\n")
    assert main([*score, "--ledger", str(ledger)]) == 2
    assert "in the run rows" in capsys.readouterr().err
    ledger.write_text(header + f"20261118T000000-v8-default,kite-db,1,real,1.0,{at}\n")
    assert main([*score, "--ledger", str(ledger)]) == 2
    assert "no lines for" in capsys.readouterr().err
    assert main([*score, "--ledger", str(tmp_path / "missing.csv")]) == 2
    assert (directory / f"{RUN}.json").read_text() == before
