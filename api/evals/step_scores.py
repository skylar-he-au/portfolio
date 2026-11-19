"""Each job-match step's own score, and the comparison table of the replays (6 r2,
contract 8). No model calls: everything comes from run rows and the subscription grades.

Scores, per gold requirement (`match-set.yaml`) through the `align` grades (which gold ids
each read requirement asks for):
- read `read_share`: of the gold requirements, the share some read requirement maps to.
- search `retrieval_share`: of the gold requirements with expected sections that some read
  requirement maps to, the share where such a read requirement got an expected section
  among its retrieved ones.
- match `match_share`: of those (an expected section was retrieved), the share where the
  match agent linked an expected section (before the check).
- check `pairs_right`: of the pairs the check judged (the stored run's pairs, its `pair`
  grades), the share it got right; `wrongly_removed` and `wrongly_kept` apart.
- write `language_pass` (code check), `code_checks_pass` (all six) and `faithfulness`
  (mean of the guides' `faith` grades; a guide with no statements is left out).

Error rows count as getting nothing right (an empty read, no link, every pair removed, a
failed language check); they are counted in `errors`. A score whose grades are not back
is `awaiting_grades` (no value); one with a grading error on a needed task is
`incomplete` (A6: shown, never declared over the line). Scores are given for the tuned
items and the held-out items apart; only the tuned ones rank (A3).

Each step's cost and latency per item come from its own calls (`calls`, the tokens of
r2b): the cost is the sum; the latency the sum, except match, whose eight agents run in
parallel: the slowest. Per match: this option's step plus the other three steps' current
option ("① 现在的", the default first model, 8 requirements), measured by their own
replays; a read option that reads more than 8 scales the other three by the requirements
it read over the stored run's. A step without a replay of its current option leaves the
total open (`missing_steps`).

Ranking per step (AI 定, 2026-11-19, review X12), tuned items only: read by `read_share`,
match by `match_share`, check by `pairs_right`, write first by all languages passing, then
`faithfulness`. Scores within 0.02 of the best of their group are the same; among those the
cheaper per match goes first. Options with no score yet go last.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals import grading
from evals.match_checks import MatchJob, load_match_set
from evals.match_replay import V4_RUN, read_rows

CAP_PER_MATCH_USD = 0.015  # SPEC "职位对照第二轮的预算": the live match, four steps
TIE = 0.02
STEPS = ("read", "match", "check", "write")
SPLITS = ("tune", "held_out")
# The score that ranks each step, and the scores each kind of run can give.
RANK_METRIC = {
    "read": "read_share",
    "match": "match_share",
    "check": "pairs_right",
    "write": "faithfulness",
}
STEP_METRICS: Mapping[str, tuple[str, ...]] = {
    "read": ("read_share",),
    "match": ("retrieval_share", "match_share"),
    "check": ("pairs_right",),
    "write": ("language_pass", "code_checks_pass", "faithfulness"),
}
ALL_METRICS = (
    "read_share",
    "retrieval_share",
    "match_share",
    "pairs_right",
    "language_pass",
    "code_checks_pass",
    "faithfulness",
)

type Row = Mapping[str, Any]


# --- grades ---------------------------------------------------------------------------------


@dataclass
class Book:
    """One run's grades (`evals.grading`), or none yet."""

    run_id: str
    grades: grading.Grades | None

    @classmethod
    def load(cls, directory: Path, run_id: str) -> "Book":
        if not (directory / "tasks.jsonl").exists() or not (directory / "grades.jsonl").exists():
            return cls(run_id, None)
        return cls(run_id, grading.load_grades(directory))

    def get(
        self, job: str, repeat: int, kind: str, n: int | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        """("ok", grade); ("awaiting", None): no grades yet, or no line for this task yet;
        ("error", None): a line with the wrong shape; ("absent", None): no such task
        (nothing to grade)."""
        if self.grades is None:
            return "awaiting", None
        tid = grading.task_id(self.run_id, job, repeat, kind, n)
        if tid not in self.grades.tasks:
            return "absent", None
        if tid in self.grades.grades:
            return "ok", self.grades.grades[tid]
        if self.grades.errors.get(tid) == "not graded":
            return "awaiting", None  # no line for it yet: the graders are not done
        return "error", None


# --- one row --------------------------------------------------------------------------------


def _alignment(grade: Mapping[str, Any]) -> dict[str, list[str]]:
    return {rid: list(gids) for rid, gids in grade["map"].items()}


def read_share(job: MatchJob, alignment: Mapping[str, Sequence[str]]) -> tuple[int, int]:
    gold = {g.id for g in job.requirements}
    covered = {gid for gids in alignment.values() for gid in gids if gid in gold}
    return len(covered), len(gold)


def _found_links(findings: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    return {
        str(f["requirement_id"]): {s["url"] for s in f.get("sections") or []}
        for f in findings
        if f.get("status") == "found"
    }


def retrieval_and_match(
    job: MatchJob,
    alignment: Mapping[str, Sequence[str]],
    retrieved: Mapping[str, Sequence[str] | None],
    findings: Iterable[Mapping[str, Any]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """((retrieved, mapped), (linked, retrieved)) over the gold requirements with
    expected sections."""
    links = _found_links(findings)
    mapped = retrieved_n = linked = 0
    for gold in job.requirements:
        if not gold.sections:
            continue
        rids = [rid for rid, gids in alignment.items() if gold.id in gids]
        if not rids:
            continue
        mapped += 1
        expected = set(gold.sections)
        got = [rid for rid in rids if set(retrieved.get(rid) or []) & expected]
        if not got:
            continue
        retrieved_n += 1
        linked += any(links.get(rid, set()) & expected for rid in got)
    return (retrieved_n, mapped), (linked, retrieved_n)


def _kept(row: Row) -> dict[tuple[str, str], bool | None]:
    if row.get("status") != "ok":
        # the check failed: `fail_all` removes every link
        return dict.fromkeys(grading.check_pairs(row), False)
    return grading.check_kept(row)


@dataclass
class Tally:
    """One score over rows: a ratio (`num`/`den`) or a mean of values."""

    num: float = 0.0
    den: int = 0
    values: list[float] = field(default_factory=list)
    awaiting: int = 0
    errors: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def add(self, num: float, den: int) -> None:
        self.num += num
        self.den += den

    def out(self, *, mean: bool = False) -> dict[str, Any]:
        status = "incomplete" if self.errors else ("awaiting_grades" if self.awaiting else "ok")
        if mean:
            value = sum(self.values) / len(self.values) if self.values else None
            body: dict[str, Any] = {"value": value, "n": len(self.values)}
        else:
            value = self.num / self.den if self.den else None
            body = {"value": value, "num": self.num, "den": self.den}
        if status == "awaiting_grades":
            body["value"] = None
        return {**body, **self.extra, "status": status, "awaiting": self.awaiting,
                "grading_errors": self.errors}  # fmt: skip


def _round_scores(out: dict[str, Any]) -> dict[str, Any]:
    if isinstance(out.get("value"), float):
        out["value"] = round(out["value"], 4)
    return out


def score_rows(
    rows: Sequence[Row],
    jobs: Mapping[str, MatchJob],
    *,
    metrics: Sequence[str],
    align_book: Book,
    align_repeat: int | None = None,
    pair_book: Book | None = None,
    pair_repeat: int | None = None,
    faith_book: Book | None = None,
) -> dict[str, dict[str, Any]]:
    """`metrics` over `rows`, per split. `align_book` holds the alignment of each row's
    requirements (the row's own run, or the stored run's at `align_repeat` with the same
    requirements); `pair_book` the stored run's pair grades (at `pair_repeat`);
    `faith_book` the row's own faith grades."""
    tallies = {split: {m: Tally() for m in metrics} for split in SPLITS}
    for row in rows:
        job = jobs[str(row["id"])]
        t = tallies["held_out" if job.holdout else "tune"]
        ok = row.get("status") == "ok"
        alignment: dict[str, list[str]] | None = None
        state = "absent"
        if {"read_share", "retrieval_share", "match_share"} & set(metrics):
            repeat = int(align_repeat if align_repeat is not None else row["repeat"])
            state, grade = align_book.get(job.id, repeat, "align")
            if state == "ok" and grade is not None:
                if align_repeat is not None:
                    _same_requirements(align_book, job.id, repeat, row)
                alignment = _alignment(grade)
            elif state == "absent":
                alignment = {}  # nothing read: nothing to align (an error row)
        for metric in ("read_share", "retrieval_share", "match_share"):
            if metric not in metrics:
                continue
            if alignment is None:
                if state == "error":
                    t[metric].errors += 1
                else:
                    t[metric].awaiting += 1
                continue
            if metric == "read_share":
                t[metric].add(*read_share(job, alignment))
                continue
            retrieval, match = retrieval_and_match(
                job,
                alignment,
                row.get("retrieved") or {},
                row.get("findings") or [] if ok else [],
            )
            t[metric].add(*(retrieval if metric == "retrieval_share" else match))
        if "pairs_right" in metrics and pair_book is not None:
            _pairs(t["pairs_right"], row, job, pair_book, pair_repeat)
        if "language_pass" in metrics:
            checks = row.get("code_checks") or {}
            t["language_pass"].add(1 if ok and checks.get("language") else 0, 1)
        if "code_checks_pass" in metrics:
            checks = row.get("code_checks") or {}
            t["code_checks_pass"].add(1 if ok and checks and all(checks.values()) else 0, 1)
        if "faithfulness" in metrics and faith_book is not None and ok:
            state, grade = faith_book.get(job.id, int(row["repeat"]), "faith")
            if state == "ok" and grade is not None:
                value = grading.faithfulness(grade)
                if value is not None:
                    t["faithfulness"].values.append(value)
            elif state == "awaiting":
                t["faithfulness"].awaiting += 1
            elif state == "error":
                t["faithfulness"].errors += 1
    return {
        split: {m: _round_scores(tallies[split][m].out(mean=m == "faithfulness")) for m in metrics}
        | {"rows": sum(1 for r in rows if jobs[str(r["id"])].holdout == (split == "held_out"))}
        for split in SPLITS
    }


def _same_requirements(book: Book, job: str, repeat: int, row: Row) -> None:
    """The stored alignment fits the row only for the very same requirements (by item and
    text, contract 7)."""
    if book.grades is None:
        raise ValueError(f"{job}: no grades in {book.run_id}")
    task = book.grades.tasks[grading.task_id(book.run_id, job, repeat, "align")]
    stored = [(r["id"], r["text"]) for r in task["payload"]["read"]]
    mine = [(r["id"], r["text"]) for r in row.get("requirements") or []]
    if stored != mine:
        raise ValueError(f"{job}: the requirements differ from the graded ones of {book.run_id}")


def _pairs(tally: Tally, row: Row, job: MatchJob, book: Book, repeat: int | None) -> None:
    kept = _kept(row)
    at = int(repeat if repeat is not None else row["repeat"])
    counts = {"wrongly_removed": 0, "wrongly_kept": 0}
    for n, pair in enumerate(grading.check_pairs(row), start=1):
        state, grade = book.get(job.id, at, "pair", n)
        if state == "absent":
            continue
        if state != "ok" or grade is None:
            if state == "error":
                tally.errors += 1
            else:
                tally.awaiting += 1
            continue
        tasks = book.grades.tasks if book.grades is not None else {}
        payload = tasks[grading.task_id(book.run_id, job.id, at, "pair", n)]["payload"]
        if (payload["requirement_id"], payload["url"]) != pair:
            raise ValueError(f"{job.id} pair {n}: not the pair graded in {book.run_id}")
        if kept[pair] is None:
            continue
        shows, keep = bool(grade["shows"]), bool(kept[pair])
        tally.add(int(shows == keep), 1)
        counts["wrongly_removed"] += int(shows and not keep)
        counts["wrongly_kept"] += int(keep and not shows)
    for key, value in counts.items():
        tally.extra[key] = tally.extra.get(key, 0) + value


# --- cost and latency -----------------------------------------------------------------------


def step_cost_ms(row: Row, step: str) -> tuple[float | None, int | None]:
    """This step's calls in one row: total cost (None if a call has no cost) and latency
    (match: the slowest of the parallel agents; else the sum). No call: 0 and 0."""
    calls = [c for c in row.get("calls") or [] if c.get("node") == step]
    if any(c.get("cost_usd") is None for c in calls):
        cost = None
    else:
        cost = sum(float(c["cost_usd"]) for c in calls)
    times = [int(c.get("ms") or 0) for c in calls]
    ms = (max(times) if times else 0) if step == "match" else sum(times)
    return cost, ms


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def step_means(rows: Sequence[Row], step: str) -> dict[str, Any]:
    costs, times, unknown = [], [], 0
    for row in rows:
        cost, ms = step_cost_ms(row, step)
        if ms is not None:
            times.append(ms)
        if cost is None:
            unknown += 1
        else:
            costs.append(cost)
    return {
        "cost_usd": None if unknown or not costs else _mean(costs),
        "ms": _mean(times),
        "rows_without_cost": unknown,
    }


def mean_requirements(rows: Sequence[Row]) -> float | None:
    return _mean([len(r.get("requirements") or []) for r in rows if r.get("status") == "ok"])


# --- runs and the table ---------------------------------------------------------------------


@dataclass
class Replay:
    meta: dict[str, Any]
    rows: list[dict[str, Any]]

    @property
    def run_id(self) -> str:
        return str(self.meta["run_id"])

    @property
    def step(self) -> str:
        return str(self.meta["step"])

    def is_current(self) -> bool:
        """The step's current option: the product's first model, 8 requirements."""
        from app.match.models import default_models

        first = str(default_models()[self.step].first)  # type: ignore[index]
        return (
            bool(self.meta.get("models"))
            and self.meta["models"][0] == first
            and int(self.meta.get("max_requirements") or 8) == 8
            and self.meta.get("status") == "complete"
        )


def load_replays(runs_dir: Path, source_run: str, only: Sequence[str] = ()) -> list[Replay]:
    replays = []
    for meta_path in sorted(runs_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "replay" or meta.get("source_run") != source_run:
            continue
        if only and meta.get("run_id") not in only:
            continue
        rows_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".jsonl")
        replays.append(Replay(meta, read_rows(rows_path)))
    return replays


def option_scores(
    replay: Replay,
    jobs: Mapping[str, MatchJob],
    *,
    own: Book,
    stored: Book,
    source_pass: int,
) -> dict[str, dict[str, Any]]:
    step = replay.step
    metrics = STEP_METRICS[step]
    if step == "read":
        return score_rows(replay.rows, jobs, metrics=metrics, align_book=own)
    return score_rows(
        replay.rows,
        jobs,
        metrics=metrics,
        align_book=stored,
        align_repeat=source_pass,
        pair_book=stored,
        pair_repeat=source_pass,
        faith_book=own,
    )


def baseline_scores(
    rows: Sequence[Row], jobs: Mapping[str, MatchJob], book: Book, source_pass: int
) -> dict[str, Any]:
    """Every step's score on the stored run's pass, with its own (re-graded) grades: the
    "old prompt" column."""
    chosen = [r for r in rows if int(r["repeat"]) == source_pass and str(r["id"]) in jobs]
    return score_rows(
        chosen,
        jobs,
        metrics=ALL_METRICS,
        align_book=book,
        pair_book=book,
        faith_book=book,
    )


def _sort_key(entry: Mapping[str, Any], step: str) -> tuple[int, float]:
    tune = entry["scores"]["tune"]
    score = tune[RANK_METRIC[step]]
    if score["status"] == "awaiting_grades" or score["value"] is None:
        return (2, 0.0)
    if step == "write" and (tune["language_pass"]["value"] or 0.0) < 1.0:
        return (1, -float(score["value"]))
    return (0, -float(score["value"]))


def rank(entries: Sequence[dict[str, Any]], step: str) -> list[str]:
    """Option run ids, best first: by the step's score (tuned items), equal within `TIE`
    of the best of a group, cheaper first among equals."""
    ordered = sorted(entries, key=lambda e: _sort_key(e, step))
    out: list[str] = []
    i = 0
    while i < len(ordered):
        tier, top = _sort_key(ordered[i], step)
        group = [ordered[i]]
        j = i + 1
        while j < len(ordered):
            t, score = _sort_key(ordered[j], step)
            if t != tier or tier == 2 or round(score - top, 9) > TIE:
                break
            group.append(ordered[j])
            j += 1

        def cost(entry: Mapping[str, Any]) -> float:
            total = entry["cost_usd"]["per_match"]
            own = entry["cost_usd"]["step"]
            value = total if total is not None else own
            return float("inf") if value is None else float(value)

        out += [e["run_id"] for e in sorted(group, key=cost)]
        i = j
    return out


def build_compare(
    jobs: Sequence[MatchJob],
    replays: Sequence[Replay],
    *,
    grading_dir: Path,
    source_rows: Sequence[Row],
    source_run: str = V4_RUN,
    source_pass: int = 1,
) -> dict[str, Any]:
    by_id = {j.id: j for j in jobs}
    stored = Book.load(grading_dir / source_run, source_run)
    source_pass_rows = [r for r in source_rows if int(r["repeat"]) == source_pass]
    source_requirements = mean_requirements(source_pass_rows)
    current: dict[str, Replay] = {}
    for replay in sorted(replays, key=lambda r: r.run_id):
        if replay.is_current():
            current[replay.step] = replay  # the latest wins
    current_means = {step: step_means(r.rows, step) for step, r in current.items()}

    steps: dict[str, Any] = {}
    for step in STEPS:
        entries = []
        for replay in sorted((r for r in replays if r.step == step), key=lambda r: r.run_id):
            own = Book.load(grading_dir / replay.run_id, replay.run_id)
            means = step_means(replay.rows, step)
            scale = 1.0
            if step == "read" and int(replay.meta.get("max_requirements") or 8) != 8:
                read = mean_requirements(replay.rows)
                if read and source_requirements:
                    scale = read / source_requirements
            cost_total: float | None = means["cost_usd"]
            ms_total: float | None = means["ms"]
            missing = []
            for other in STEPS:
                if other == step:
                    continue
                if other not in current_means or current_means[other]["cost_usd"] is None:
                    missing.append(other)
                    continue
                if cost_total is not None:
                    cost_total += current_means[other]["cost_usd"] * scale
                if ms_total is not None and current_means[other]["ms"] is not None:
                    ms_total += current_means[other]["ms"] * scale
            if missing:
                cost_total = ms_total = None
            entries.append(
                {
                    "run_id": replay.run_id,
                    "option": replay.meta.get("option"),
                    "order": replay.meta.get("order"),
                    "max_requirements": replay.meta.get("max_requirements"),
                    "current": replay.is_current(),
                    "status": replay.meta.get("status"),
                    "rows": len(replay.rows),
                    "errors": sum(r.get("status") != "ok" for r in replay.rows),
                    "fallback_rows": sum(bool(r.get("fallback")) for r in replay.rows),
                    "requirements_read": _r(mean_requirements(replay.rows))
                    if step == "read"
                    else None,
                    "scores": option_scores(
                        replay, by_id, own=own, stored=stored, source_pass=source_pass
                    ),
                    "cost_usd": {
                        "step": _r(means["cost_usd"], 8),
                        "per_match": _r(cost_total, 8),
                        "over_cap": None if cost_total is None else cost_total > CAP_PER_MATCH_USD,
                        "scale": round(scale, 4),
                        "missing_steps": missing,
                    },
                    "latency_ms": {"step": _r(means["ms"], 0), "per_match": _r(ms_total, 0)},
                    "grading": None if own.grades is None else grading.grading_block(own.grades),
                }
            )
        steps[step] = {
            "metric": RANK_METRIC[step],
            "current": current[step].run_id if step in current else None,
            "options": entries,
            "ranking": rank(entries, step),
        }
    return {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_run": source_run,
        "source_pass": source_pass,
        "items": {
            "tune": sum(not j.holdout for j in jobs),
            "held_out": sum(j.holdout for j in jobs),
        },
        "cap_per_match_usd": CAP_PER_MATCH_USD,
        "tie": TIE,
        "old_prompt": {
            "run_id": source_run,
            "pass": source_pass,
            "grading": None if stored.grades is None else grading.grading_block(stored.grades),
            "scores": baseline_scores(source_rows, by_id, stored, source_pass),
        },
        "steps": steps,
    }


def _r(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits) if digits else float(round(value))


def _line(entry: Mapping[str, Any], step: str) -> str:
    score = entry["scores"]["tune"][RANK_METRIC[step]]
    shown = score["status"] if score["value"] is None else f"{score['value']:.3f}"
    if score["status"] == "awaiting_grades":
        shown = "等判分"
    cost = entry["cost_usd"]["per_match"]
    return (
        f"  {entry['option']:<24} {RANK_METRIC[step]}={shown:<10}"
        f" step=${entry['cost_usd']['step']} per_match="
        f"{'open' if cost is None else f'${cost:.6f}'} ({entry['run_id']})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--set", type=Path, required=True, dest="match_set")
    parser.add_argument("--runs", type=Path, required=True, help="runs dir (docs/evals/runs)")
    parser.add_argument("--grading", type=Path, required=True, help="docs/evals/grading")
    parser.add_argument("--out", type=Path, required=True, help="docs/evals/compare/step6-r2.json")
    parser.add_argument("--source-run", default=V4_RUN)
    parser.add_argument("--source-pass", type=int, default=1)
    parser.add_argument("--run", action="append", default=[], help="only these replay run ids")
    args = parser.parse_args(argv)
    jobs = load_match_set(args.match_set)
    replays = load_replays(args.runs, args.source_run, args.run)
    try:
        compare = build_compare(
            jobs,
            replays,
            grading_dir=args.grading,
            source_rows=read_rows(args.runs / f"{args.source_run}.jsonl"),
            source_run=args.source_run,
            source_pass=args.source_pass,
        )
    except (OSError, ValueError) as error:
        print(f"cannot build the table: {error}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(compare, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for step in STEPS:
        print(f"{step}:")
        for entry in compare["steps"][step]["options"]:
            print(_line(entry, step))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
