"""Public aggregate output, built from an explicit whitelist without sample text."""

import json
from collections import Counter
from contextlib import suppress
from pathlib import Path
from statistics import mean
from typing import Any

from evals.checks import CATEGORIES, quote_counts
from evals.pricing import POOLS
from evals.types import METRICS, NOT_JUDGED


def _cost_fields(rows: list[dict[str, Any]], pools: dict[str, float] | None) -> dict[str, Any]:
    """The total, plus what each spending pool paid (real money, provider credit) when known."""
    fields: dict[str, Any] = {"cost_usd": round(sum(row["cost_usd"] for row in rows), 6)}
    if pools is not None:
        fields["cost_pools_usd"] = {pool: round(pools.get(pool, 0.0), 6) for pool in POOLS}
    return fields


def _metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    passes: dict[int, list[float]] = {}
    values: list[float] = []
    for row in rows:
        value = (row["ragas"] or {}).get(metric)
        if value is not None:
            values.append(value)
            passes.setdefault(row["repeat"], []).append(value)
    pass_means = [mean(scores) for scores in passes.values()]
    return {
        "mean": round(mean(pass_means), 6) if pass_means else None,
        "min": round(min(pass_means), 6) if pass_means else None,
        "max": round(max(pass_means), 6) if pass_means else None,
        "scored": len(values),
        "below_floor": sum(value < 0.7 for value in values) if metric == "faithfulness" else None,
    }


def _quote_match(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """引句核上的比例: quotes code found in the cited section / quotes the model gave.

    Over the general answers of v10 and later (`checks.quote_counts`); `rate` is None when
    no quote was given. `citations_quoted` / `citations` is how many citations can mark a
    sentence instead of the whole section. `passes` has the same rate for each pass.
    """
    totals = dict.fromkeys(("samples", "given", "kept", "citations", "citations_quoted"), 0)
    by_pass: dict[int, list[int]] = {}
    for row in rows:
        counts = quote_counts(row["citations"], row["nodes"]) if row["status"] == "ok" else None
        if counts is None:
            continue
        totals["samples"] += 1
        for key, value in counts.items():
            totals[key] += value
        given_kept = by_pass.setdefault(row["repeat"], [0, 0])
        given_kept[0] += counts["given"]
        given_kept[1] += counts["kept"]

    def rate(given: int, kept: int) -> float | None:
        return round(kept / given, 6) if given else None

    return {
        **totals,
        "rate": rate(totals["given"], totals["kept"]),
        "passes": {str(repeat): rate(*by_pass[repeat]) for repeat in sorted(by_pass)},
    }


def _model(rows: list[dict[str, Any]], name: str) -> str | None:
    counts = Counter(
        node["model"]
        for row in rows
        if row["status"] == "ok"
        for node in row["nodes"]
        if node["name"] == name and node.get("model") is not None
    )
    return counts.most_common(1)[0][0] if counts else None


def build_summary(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    date: str,
    prompt_version: str,
    route: str,
    split: str,
    repeats: int,
    questions: int,
    models: dict[str, str | None],
    cost_pools: dict[str, float] | None = None,
    grader: str | None = None,
    grading: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregates of a Q&A run. `ragas` holds the judged metrics (`METRICS`) and
    `not_judged` the ones no longer judged; `ragas_nulls` counts only judged metrics that
    have no score. `grader` and `grading` are set when the scores come from subscription
    grades (`evals.qa_grading`), as in `build_match_summary`."""
    averaged = [row for row in rows if row["repeat"] <= repeats]
    groups = {
        "tune": [row for row in averaged if not row["held_out"]],
        "held_out": [row for row in averaged if row["held_out"]],
        "zh": [row for row in averaged if row["category"] == "zh"],
    }
    private = [row for row in rows if row["accept_fixed"]]
    fixed = sum(row["fixed_answer"] for row in private)
    answered = sum(
        row["status"] == "ok" and not row["fixed_answer"] and row["passed"] for row in private
    )
    return {
        "kind": "qa",
        "run_id": run_id,
        "date": date,
        "prompt_version": prompt_version,
        "route": route,
        "split": split,
        "repeats": repeats,
        "questions": questions,
        "samples": len(rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "ragas_nulls": sum(
            (row["ragas"] or {}).get(metric) is None
            for row in rows
            for metric in row["requested_metrics"]
            if metric in METRICS
        ),
        "models": {
            "answer": _model(rows, "answer"),
            "check": _model(rows, "check"),
            "judge": models.get("judge"),
            "embed": models.get("embed"),
        },
        "ragas": {
            group: {metric: _metric(samples, metric) for metric in METRICS}
            for group, samples in groups.items()
            if samples
        },
        "not_judged": list(NOT_JUDGED),
        "code_checks": {
            category: {
                "passed": sum(
                    row["status"] == "ok" and row["passed"]
                    for row in averaged
                    if row["category"] == category
                ),
                "total": sum(row["category"] == category for row in averaged),
            }
            for category in CATEGORIES
        },
        "quote_match": _quote_match(averaged),
        "private_repeat": (
            {
                "runs": len(private),
                "fixed": fixed,
                "answered": answered,
                "failed": len(private) - fixed - answered,
            }
            if private
            else None
        ),
        **_cost_fields(rows, cost_pools),
        **({"grader": grader} if grader is not None or grading is not None else {}),
        **({"grading": grading} if grading is not None else {}),
    }


MATCH_SCORES = ("recall", "precision", "not_covered_accuracy")
MATCH_FAITH_FLOOR = 0.7


def _spread(values_by_pass: dict[int, list[float]]) -> dict[str, Any]:
    """Mean of each pass's mean, and the lowest and highest pass mean."""
    pass_means = [mean(values) for values in values_by_pass.values() if values]
    return {
        "mean": round(mean(pass_means), 6) if pass_means else None,
        "min": round(min(pass_means), 6) if pass_means else None,
        "max": round(max(pass_means), 6) if pass_means else None,
        "scored": sum(len(values) for values in values_by_pass.values()),
    }


def _match_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for name in MATCH_SCORES:
        by_pass: dict[int, list[float]] = {}
        for row in rows:
            value = (row.get("scores") or {}).get(name)
            if value is not None:
                by_pass.setdefault(row["repeat"], []).append(value)
        scores[name] = _spread(by_pass)
    faith: dict[int, list[float]] = {}
    for row in rows:
        if row.get("faithfulness") is not None:
            faith.setdefault(row["repeat"], []).append(row["faithfulness"])
    scores["faithfulness"] = {
        **_spread(faith),
        "below_floor": sum(v < MATCH_FAITH_FLOOR for values in faith.values() for v in values),
        "missing": sum(
            row.get("faith_scored", False) and row.get("faithfulness") is None for row in rows
        ),
    }
    return scores


def build_match_summary(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    date: str,
    prompt_version: str,
    route: str,
    split: str,
    repeats: int,
    questions: int,
    models: dict[str, str | None],
    cost_pools: dict[str, float] | None = None,
    grader: str | None = None,
    grading: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregates of a job-match run, no job ad, requirement or guide text.

    `grader` and `grading` are set when the scores come from subscription grades
    (`evals.grading`): who graded, and whether every required grade came back
    (`grading["status"]` "incomplete": the run cannot be declared over the line). Runs
    judged through the API leave both out, so their summary is as before."""
    from evals.match_checks import CODE_CHECKS, KINDS

    groups = {
        "tune": [row for row in rows if not row["held_out"]],
        "held_out": [row for row in rows if row["held_out"]],
    }
    ok = [row for row in rows if row["status"] == "ok"]
    return {
        "kind": "match",
        "run_id": run_id,
        "date": date,
        "prompt_version": prompt_version,
        "route": route,
        "split": split,
        "repeats": repeats,
        "questions": questions,
        "samples": len(rows),
        "errors": len(rows) - len(ok),
        "fallback_samples": sum(bool(row["fallback"]) for row in rows),
        "models": {
            key: models.get(key)
            for key in ("answer", "check", "embed", "judge_align", "judge_faithfulness")
        },
        "scores": {group: _match_scores(samples) for group, samples in groups.items() if samples},
        "code_checks": {
            "all": {"passed": sum(row["passed"] for row in rows), "total": len(rows)},
            **{
                name: {
                    "passed": sum(row["code_checks"].get(name, False) for row in rows),
                    "total": len(rows),
                }
                for name in CODE_CHECKS
            },
        },
        "by_kind": {
            kind: {
                "passed": sum(row["passed"] for row in rows if row["kind"] == kind),
                "total": sum(row["kind"] == kind for row in rows),
            }
            for kind in KINDS
            if any(row["kind"] == kind for row in rows)
        },
        "requirements": {
            "read": sum(len(row["requirements"]) for row in ok),
            "unchecked": sum((row["result"] or {}).get("unchecked", 0) for row in ok),
            "extra": sum(len((row["scores"] or {}).get("extra", [])) for row in ok),
            "missing": sum(len((row["scores"] or {}).get("missing", [])) for row in ok),
        },
        **_cost_fields(rows, cost_pools),
        **({"grader": grader} if grader is not None or grading is not None else {}),
        **({"grading": grading} if grading is not None else {}),
    }


def write_summary(summary: dict[str, Any], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{summary['run_id']}.json"
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return path
