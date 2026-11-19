"""Each step's own score and the comparison table (evals.step_scores): hand-counted fixtures,
plus the real 26-item set and v4 rows when the docs repo is there."""

import json
from pathlib import Path
from typing import Any

import pytest
from test_match_replay import JOBS, LAMBDA, PY, SQL, _ref, stored_row

from evals.grading import export_tasks, write_tasks
from evals.match_checks import load_match_set
from evals.match_replay import V4_RUN, read_rows
from evals.step_scores import (
    Book,
    build_compare,
    load_replays,
    main,
    rank,
    read_share,
    retrieval_and_match,
    step_cost_ms,
)

DOCS = Path("/Users/skylar/workspace/portfolio/docs/evals")
TEXTS = {url: f"Skylar's project page /projects/roleradar, section {url}:\ntext" for url in
         (PY, SQL, LAMBDA)}  # fmt: skip
HAIKU = "anthropic:claude-haiku-5-5:off"
GEMINI = "google:gemini-3.5-flash-lite:minimal"
LUNA = "openai:gpt-6-luna:none"


def _call(node: str, cost: float | None, ms: int) -> dict[str, Any]:
    return {"node": node, "provider": "anthropic", "ok": True, "cost_usd": cost, "ms": ms}


def baseline_rows() -> list[dict[str, Any]]:
    """The stored run, pass 1 (and a pass 2 the scores must leave out)."""
    rows = []
    for repeat in (1, 2):
        for job in JOBS:
            row = stored_row(job.id, repeat)
            row["code_checks"] = {"language": True, "schema": True}
            rows.append(row)
    return rows


def write_run(runs: Path, run_id: str, rows: list[dict[str, Any]], **meta: Any) -> None:
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{run_id}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (runs / f"{run_id}.meta.json").write_text(json.dumps({"run_id": run_id, **meta}))


def grade(grading: Path, run_id: str, rows: list[dict[str, Any]], answers: dict[str, Any],
          kinds: tuple[str, ...] = ("faith", "align", "pair"), pair_repeats: int | None = 1,
          ) -> None:  # fmt: skip
    """Export the rows' tasks and grade them: `answers` maps a task id suffix
    (`real-01/1/align`) to the grade's fields; a task with no answer stays ungraded."""
    tasks = [
        t
        for t in export_tasks(rows, JOBS, section_texts=lambda r: TEXTS, pair_repeats=pair_repeats)
        if t["kind"] in kinds
    ]
    write_tasks(tasks, grading / run_id, {"run_id": run_id})
    lines = []
    for task in tasks:
        suffix = task["task_id"].removeprefix(f"{run_id}/")
        if suffix in answers:
            lines.append(json.dumps({"task_id": task["task_id"], "grader": "t", **answers[suffix]}))
    (grading / run_id / "grades.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stmts(supported: int, total: int) -> dict[str, Any]:
    return {"statements": [{"text": f"s{i}", "supported": i < supported} for i in range(total)]}


BASELINE_GRADES = {
    "real-01/1/align": {"map": {"r1": ["real-01.r1"], "r2": ["real-01.r2"], "r3": ["real-01.r3"]}},
    "real-02/1/align": {"map": {"r1": ["real-02.r1"], "r2": [], "r3": []}},
    "real-01/1/pair/1": {"shows": True},
    "real-01/1/pair/2": {"shows": True},  # the stored guide dropped SQL: wrongly removed
    "real-02/1/pair/1": {"shows": True},
    "real-02/1/pair/2": {"shows": False},  # dropped, rightly
    "real-01/1/faith": _stmts(1, 2),
    "real-02/1/faith": _stmts(1, 1),
}


def replay_meta(step: str, first: str, option: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "replay",
        "step": step,
        "source_run": V4_RUN,
        "models": [first],
        "option": option,
        "order": first,
        "max_requirements": 8,
        "status": "complete",
        **extra,
    }


def setup(tmp_path: Path, *, graded: bool = True) -> tuple[Path, Path]:
    runs, grading = tmp_path / "runs", tmp_path / "grading"
    base = baseline_rows()
    write_run(runs, V4_RUN, base, kind="match")
    grade(grading, V4_RUN, base, BASELINE_GRADES if graded else {})

    # read: the current option and a 12-requirement one
    read_rows_ = []
    for job in JOBS:
        row = {"run_id": "R-read", "id": job.id, "repeat": 1, "status": "ok", "step": "read",
               "requirements": [{"id": "r1", "text": "Python and SQL", "query": "q"}],
               "calls": [_call("read", 0.001, 1000)]}  # fmt: skip
        read_rows_.append(row)
    write_run(runs, "R-read", read_rows_, **replay_meta("read", HAIKU, "haiku-5-5-off"))
    grade(
        grading, "R-read", read_rows_,
        {"real-01/1/align": {"map": {"r1": ["real-01.r1", "real-01.r2"]}}}
        if graded else {},
        kinds=("align",),
    )  # fmt: skip
    twelve = [
        {**r, "run_id": "R-read12",
         "requirements": [{"id": f"r{n}", "text": f"t{n}", "query": "q"} for n in range(1, 7)],
         "calls": [_call("read", 0.002, 1500)]}
        for r in read_rows_
    ]  # fmt: skip
    write_run(
        runs, "R-read12", twelve,
        **replay_meta("read", HAIKU, "haiku-5-5-off-max12", max_requirements=12),
    )  # fmt: skip

    # match: the current option links LAMBDA for r1 (not an expected section)
    match_rows = []
    for job in JOBS:
        source = stored_row(job.id)
        findings = [dict(f) for f in source["findings"]]
        findings[0] = {**findings[0], "sections": [_ref(LAMBDA)]}
        match_rows.append(
            {"run_id": "R-match", "id": job.id, "repeat": 1, "status": "ok", "step": "match",
             "requirements": source["requirements"], "retrieved": source["retrieved"],
             "findings": findings,
             "calls": [_call("match", 0.0005, 900), _call("match", 0.0005, 1200)]}
        )  # fmt: skip
    write_run(runs, "R-match", match_rows, **replay_meta("match", HAIKU, "haiku-5-5-off"))

    # check: Luna keeps both pairs (the current Gemini option has no replay yet)
    check_rows = []
    for job in JOBS:
        source = stored_row(job.id)
        check_rows.append(
            {"run_id": "R-check", "id": job.id, "repeat": 1, "status": "ok", "step": "check",
             "requirements": source["requirements"], "retrieved": source["retrieved"],
             "findings": source["findings"],
             "check_verdicts": [
                 {"requirement_id": "r1", "url": PY, "shows": True},
                 {"requirement_id": "r2", "url": SQL, "shows": True},
             ],
             "calls": [_call("check", 0.0001, 700)]}
        )  # fmt: skip
    write_run(runs, "R-check", check_rows, **replay_meta("check", LUNA, "luna-none"))

    # write: real-02's guide is in the wrong language
    write_rows = []
    for job in JOBS:
        source = stored_row(job.id)
        write_rows.append(
            {"run_id": "R-write", "id": job.id, "repeat": 1, "status": "ok", "step": "write",
             "requirements": source["requirements"], "retrieved": source["retrieved"],
             "findings": source["findings"], "result": source["result"],
             "code_checks": {"language": job.id == "real-01", "schema": True},
             "calls": [_call("write", 0.0008, 1100)]}
        )  # fmt: skip
    write_run(runs, "R-write", write_rows, **replay_meta("write", HAIKU, "haiku-5-5-off"))
    grade(
        grading, "R-write", write_rows,
        {"real-01/1/faith": _stmts(3, 4), "real-02/1/faith": _stmts(0, 0)} if graded else {},
        kinds=("faith",),
    )  # fmt: skip
    return runs, grading


def compare_of(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    runs, grading = setup(tmp_path, **kwargs)
    return build_compare(
        JOBS,
        load_replays(runs, V4_RUN),
        grading_dir=grading,
        source_rows=read_rows(runs / f"{V4_RUN}.jsonl"),
    )


def option(compare: dict[str, Any], step: str, run_id: str) -> dict[str, Any]:
    return next(e for e in compare["steps"][step]["options"] if e["run_id"] == run_id)


def value(entry: dict[str, Any], metric: str, split: str = "tune") -> Any:
    return entry["scores"][split][metric]["value"]


# --- the scores, counted by hand ------------------------------------------------------------


def test_old_prompt_column_scores_every_step_on_pass_one(tmp_path: Path) -> None:
    old = compare_of(tmp_path)["old_prompt"]["scores"]
    tune, held = old["tune"], old["held_out"]
    assert (tune["rows"], held["rows"]) == (1, 1)  # pass 2 left out
    assert tune["read_share"] | {} == {
        "value": 1.0, "num": 3, "den": 3, "status": "ok", "awaiting": 0, "grading_errors": 0,
    }  # fmt: skip
    assert (tune["retrieval_share"]["value"], tune["match_share"]["value"]) == (1.0, 1.0)
    assert tune["pairs_right"]["num"] == 1 and tune["pairs_right"]["den"] == 2
    assert tune["pairs_right"]["wrongly_removed"] == 1 and tune["pairs_right"]["wrongly_kept"] == 0
    assert held["pairs_right"]["value"] == 1.0  # the stored guide dropped SQL, which it isn't
    assert tune["language_pass"]["value"] == 1.0
    assert tune["faithfulness"]["value"] == 0.5 and held["faithfulness"]["value"] == 1.0


def test_a_faith_regrade_file_replaces_the_old_faith_grades(tmp_path: Path) -> None:
    """A7 option A: the old-prompt column takes the stored run's `grades-faith-v2.jsonl`
    (GRADING.md 5.7) in place of its faith lines; align and pair stay from `grades.jsonl`."""
    runs, grading = setup(tmp_path)
    regrade = {"real-01/1/faith": _stmts(4, 4)}  # real-02's v2 line is missing: not graded
    (grading / V4_RUN / "grades-faith-v2.jsonl").write_text(
        "".join(
            json.dumps({"task_id": f"{V4_RUN}/{k}", "grader": "t", **v}) + "\n"
            for k, v in regrade.items()
        ),
        encoding="utf-8",
    )
    compare = build_compare(
        JOBS, load_replays(runs, V4_RUN), grading_dir=grading,
        source_rows=read_rows(runs / f"{V4_RUN}.jsonl"),
    )  # fmt: skip
    old = compare["old_prompt"]["scores"]
    assert old["tune"]["faithfulness"]["value"] == 1.0  # v1 said 0.5
    assert old["held_out"]["faithfulness"]["status"] == "awaiting_grades"  # v1's line is gone
    assert old["tune"]["read_share"]["value"] == 1.0  # align untouched
    assert old["tune"]["pairs_right"]["num"] == 1  # pair untouched


def test_each_replay_scores_only_its_own_step(tmp_path: Path) -> None:
    compare = compare_of(tmp_path)
    read = option(compare, "read", "R-read")
    assert set(read["scores"]["tune"]) == {"read_share", "rows"}
    assert value(read, "read_share") == pytest.approx(
        2 / 3, abs=1e-4
    )  # one read requirement, two golds
    assert read["scores"]["held_out"]["read_share"]["status"] == "awaiting_grades"

    match = option(compare, "match", "R-match")
    assert set(match["scores"]["tune"]) == {"retrieval_share", "match_share", "rows"}
    assert value(match, "retrieval_share") == 1.0
    tune_match = match["scores"]["tune"]["match_share"]
    assert (tune_match["num"], tune_match["den"]) == (1, 2)  # LAMBDA is not Python's section
    assert value(match, "match_share", "held_out") == 0.0

    check = option(compare, "check", "R-check")
    pairs = check["scores"]["tune"]["pairs_right"]
    assert (pairs["num"], pairs["den"], pairs["wrongly_kept"]) == (2, 2, 0)
    held = check["scores"]["held_out"]["pairs_right"]
    assert (held["num"], held["den"], held["wrongly_kept"]) == (1, 2, 1)

    write = option(compare, "write", "R-write")
    assert value(write, "language_pass") == 1.0
    assert value(write, "language_pass", "held_out") == 0.0
    assert value(write, "faithfulness") == 0.75
    # no statements: left out (RAGAS's nan), not 0
    assert write["scores"]["held_out"]["faithfulness"] | {} == {
        "value": None, "n": 0, "status": "ok", "awaiting": 0, "grading_errors": 0,
    }  # fmt: skip


def test_scores_wait_for_the_grades(tmp_path: Path) -> None:
    compare = compare_of(tmp_path, graded=False)
    for step, run_id, metric in (
        ("read", "R-read", "read_share"),
        ("match", "R-match", "match_share"),
        ("check", "R-check", "pairs_right"),
        ("write", "R-write", "faithfulness"),
    ):
        score = option(compare, step, run_id)["scores"]["tune"][metric]
        assert score["status"] == "awaiting_grades" and score["value"] is None, step
    write = option(compare, "write", "R-write")
    assert value(write, "language_pass") == 1.0  # code: no grades needed
    assert compare["old_prompt"]["scores"]["tune"]["faithfulness"]["status"] == "awaiting_grades"


def test_a_grading_error_marks_the_score_incomplete(tmp_path: Path) -> None:
    runs, grading = setup(tmp_path)
    path = grading / "R-write" / "grades.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(
            line.replace('"statements"', '"oops"') if "real-01" in line else line for line in lines
        ),
        encoding="utf-8",
    )
    compare = build_compare(
        JOBS, load_replays(runs, V4_RUN), grading_dir=grading,
        source_rows=read_rows(runs / f"{V4_RUN}.jsonl"),
    )  # fmt: skip
    score = option(compare, "write", "R-write")["scores"]["tune"]["faithfulness"]
    assert score["status"] == "incomplete" and score["grading_errors"] == 1
    assert option(compare, "write", "R-write")["grading"]["status"] == "incomplete"


def test_stored_alignment_needs_the_same_requirements(tmp_path: Path) -> None:
    runs, grading = setup(tmp_path)
    rows = read_rows(runs / "R-match.jsonl")
    rows[0]["requirements"][0]["text"] = "Python 3"
    write_run(runs, "R-match", rows, **replay_meta("match", HAIKU, "haiku-5-5-off"))
    with pytest.raises(ValueError, match="requirements differ"):
        build_compare(
            JOBS, load_replays(runs, V4_RUN), grading_dir=grading,
            source_rows=read_rows(runs / f"{V4_RUN}.jsonl"),
        )  # fmt: skip


def test_error_rows_get_nothing_right(tmp_path: Path) -> None:
    job = JOBS[0]
    assert read_share(job, {}) == (0, 3)
    alignment = {"r1": ["real-01.r1"], "r2": ["real-01.r2"]}
    source = stored_row(job.id)
    assert retrieval_and_match(job, alignment, source["retrieved"], []) == ((2, 2), (0, 2))


# --- cost, latency, the table ---------------------------------------------------------------


def test_step_cost_and_latency_per_row() -> None:
    row = {"calls": [_call("match", 0.001, 900), _call("match", 0.002, 1500),
                     _call("read", 0.5, 10)]}  # fmt: skip
    assert step_cost_ms(row, "match") == (pytest.approx(0.003), 1500)  # parallel: the slowest
    assert step_cost_ms(row, "read") == (0.5, 10)
    assert step_cost_ms({"calls": [_call("write", None, 5), _call("write", 0.1, 5)]}, "write") == (
        None, 10,
    )  # fmt: skip
    assert step_cost_ms({"calls": []}, "check") == (0, 0)


def test_cost_per_match_adds_the_other_steps_current_options(tmp_path: Path) -> None:
    compare = compare_of(tmp_path)
    steps = compare["steps"]
    assert steps["read"]["current"] == "R-read" and steps["check"]["current"] is None
    read = option(compare, "read", "R-read")
    assert read["current"] is True
    assert read["cost_usd"]["step"] == 0.001
    # no replay of the current check yet: the totals stay open
    assert read["cost_usd"]["per_match"] is None and read["cost_usd"]["missing_steps"] == ["check"]
    check = option(compare, "check", "R-check")
    assert check["current"] is False  # Luna is not the product's check
    # check 0.0001 + read 0.001 + match 0.001 + write 0.0008
    assert check["cost_usd"]["per_match"] == pytest.approx(0.0029)
    assert check["cost_usd"]["over_cap"] is False
    # latency: check 700 + read 1000 + match (slowest) 1200 + write 1100
    assert check["latency_ms"]["per_match"] == 4000
    twelve = option(compare, "read", "R-read12")
    assert twelve["requirements_read"] == 6 and twelve["cost_usd"]["scale"] == 2.0  # 6 over 3


def test_twelve_scales_the_other_steps(tmp_path: Path) -> None:
    runs, grading = setup(tmp_path)
    rows = read_rows(runs / "R-check.jsonl")
    write_run(runs, "R-checkg", rows, **replay_meta("check", GEMINI, "35-flash-lite-minimal"))
    compare = build_compare(
        JOBS, load_replays(runs, V4_RUN), grading_dir=grading,
        source_rows=read_rows(runs / f"{V4_RUN}.jsonl"),
    )  # fmt: skip
    twelve = option(compare, "read", "R-read12")
    # read 0.002 + 2 x (match 0.001 + check 0.0001 + write 0.0008)
    assert twelve["cost_usd"]["per_match"] == pytest.approx(0.002 + 2 * 0.0019)
    assert twelve["cost_usd"]["missing_steps"] == []


def _entry(run_id: str, score: float | None, cost: float, language: float = 1.0) -> dict:
    status = "ok" if score is not None else "awaiting_grades"
    return {
        "run_id": run_id,
        "scores": {"tune": {
            "match_share": {"value": score, "status": status},
            "faithfulness": {"value": score, "status": status},
            "language_pass": {"value": language, "status": "ok"},
        }},
        "cost_usd": {"per_match": cost, "step": cost},
    }  # fmt: skip


def test_rank_puts_the_cheaper_first_among_equal_scores() -> None:
    entries = [
        _entry("a", 0.80, 0.010),
        _entry("b", 0.81, 0.020),
        _entry("c", 0.79, 0.005),  # within 0.02 of b: the same, and cheapest
        _entry("d", 0.70, 0.001),
        _entry("e", None, 0.0),
    ]
    assert rank(entries, "match") == ["c", "a", "b", "d", "e"]
    # write: a language failure ranks below every guide that passes it
    writes = [_entry("w1", 0.95, 0.01, language=0.9), _entry("w2", 0.80, 0.01)]
    assert rank(writes, "write") == ["w2", "w1"]


def test_main_writes_the_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs, grading = setup(tmp_path, graded=False)
    match_set = tmp_path / "set.yaml"
    match_set.write_text(
        json.dumps({"version": 1, "jobs": [
            {"id": j.id, "kind": j.kind, "holdout": j.holdout, "jd": j.jd,
             "requirements": [
                 {"id": g.id, "text": g.text,
                  "expect": {"sections": list(g.sections)} if g.sections
                  else {"not_covered": True}}
                 for g in j.requirements]}
            for j in JOBS]}),
        encoding="utf-8",
    )  # fmt: skip
    out = tmp_path / "compare" / "step6-r2.json"
    code = main(["--set", str(match_set), "--runs", str(runs), "--grading", str(grading),
                 "--out", str(out)])  # fmt: skip
    assert code == 0
    table = json.loads(out.read_text(encoding="utf-8"))
    assert table["source_run"] == V4_RUN and set(table["steps"]) == {
        "read", "match", "check", "write",
    }  # fmt: skip
    assert "等判分" in capsys.readouterr().out
    # nothing from the job ads or the guides in the table
    text = out.read_text(encoding="utf-8")
    assert "Python" not in text and "Kubernetes" not in text and "Old text" not in text


# --- the real 26 items ----------------------------------------------------------------------


@pytest.mark.skipif(not (DOCS / "runs" / f"{V4_RUN}.jsonl").exists(), reason="docs repo only")
def test_the_real_set_and_v4_rows_split_21_and_5(tmp_path: Path) -> None:
    jobs = load_match_set(DOCS / "match-set.yaml")
    rows = read_rows(DOCS / "runs" / f"{V4_RUN}.jsonl")
    pass_one = [r for r in rows if r["repeat"] == 1]
    gold = {j.id: j for j in jobs}
    # every read requirement aligned to every gold id: each item reads all of its golds
    tasks = [
        {"task_id": f"{V4_RUN}/{r['id']}/1/align", "kind": "align", "sample": {},
         "payload": {"read": [{"id": q["id"], "text": q["text"]} for q in r["requirements"]],
                     "reference": [{"id": g.id, "text": g.text}
                                   for g in gold[r["id"]].requirements]}}
        for r in pass_one
    ]  # fmt: skip
    directory = tmp_path / V4_RUN
    write_tasks(tasks, directory, {})
    (directory / "grades.jsonl").write_text(
        "".join(
            json.dumps({"task_id": t["task_id"], "grader": "t", "map": {
                q["id"]: [g["id"] for g in t["payload"]["reference"]]
                for q in t["payload"]["read"]}}) + "\n"
            for t in tasks
        ),
        encoding="utf-8",
    )  # fmt: skip
    compare = build_compare(jobs, [], grading_dir=tmp_path, source_rows=rows)
    assert compare["items"] == {"tune": 21, "held_out": 5}
    scores = compare["old_prompt"]["scores"]
    assert (scores["tune"]["rows"], scores["held_out"]["rows"]) == (21, 5)
    assert scores["tune"]["read_share"]["value"] == 1.0
    assert scores["tune"]["read_share"]["den"] == sum(
        len(j.requirements) for j in jobs if not j.holdout
    )
    # the retrieval and match shares are code over v4's own links: between 0 and 1, counted
    for metric in ("retrieval_share", "match_share"):
        score = scores["tune"][metric]
        assert score["status"] == "ok" and 0 < score["value"] <= 1 and score["den"] > 0
    assert scores["tune"]["language_pass"]["den"] == 21
    assert scores["tune"]["faithfulness"]["n"] == 0  # no faith tasks in this export
    assert Book.load(tmp_path / "nothing", "x").grades is None
