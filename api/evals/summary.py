"""Public aggregate output, built from an explicit whitelist without sample text."""

import json
from collections import Counter
from contextlib import suppress
from pathlib import Path
from statistics import mean
from typing import Any

from evals.checks import CATEGORIES
from evals.types import METRICS


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
) -> dict[str, Any]:
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
        "cost_usd": round(sum(row["cost_usd"] for row in rows), 6),
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
