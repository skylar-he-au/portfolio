"""Run an offline YAML question set through the real Q&A graph."""

import argparse
import asyncio
import dataclasses
import functools
import json
import math
import re
import sys
from collections.abc import Awaitable, Callable, Iterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from langgraph.checkpoint.memory import InMemorySaver

from app.qa import memory
from app.qa.graph import PRIVATE_FALLBACK, GraphDeps, build_graph
from app.qa.output import events
from app.qa.prompts import PROMPT_VERSION
from app.rag.types import Embedder
from evals.checks import (
    CATEGORIES,
    ROUTES,
    Question,
    SampleResult,
    failed_checks,
    metrics_for,
    run_code_checks,
    sample_passed,
)
from evals.summary import build_summary, write_summary
from evals.types import METRICS, CapReached, MeterLike, ScoreInput, UnknownModel

SAMPLE_TIMEOUT_S = 180.0
# The RAGAS judges of one sample (up to four metrics, each a few calls) together; past it
# the sample's scores are missing like any judge failure, never 0.
SCORE_TIMEOUT_S = 300.0
JUDGE_MODEL = "claude-sonnet-5-5"  # the default `judge_model` of evals.ragas_score.score
type Scorer = Callable[[ScoreInput, MeterLike], Awaitable[dict[str, float | None]]]


class MeteredEmbedder:
    def __init__(self, inner: Embedder, meter: MeterLike, model: str) -> None:
        self.inner = inner
        self.meter = meter
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.meter.add_embedding_chars(self.model, sum(len(text) for text in texts))
        return await self.inner.embed(texts)


def load_questions(path: Path | str) -> list[Question]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read question set: {type(error).__name__}") from error
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("question set version must be 1")
    entries = data.get("questions")
    if not isinstance(entries, list):
        raise ValueError("questions must be a list")
    questions: list[Question] = []
    ids: set[str] = set()
    for index, item in enumerate(entries, 1):
        prefix = f"question {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be a mapping")
        ident = item.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", ident):
            raise ValueError(f"{prefix}: id must be lowercase letters, digits and hyphens")
        if ident in ids:
            raise ValueError(f"duplicate question id: {ident}")
        ids.add(ident)
        category = item.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"{prefix}: invalid category")
        if item.get("expect_route", "general") not in ROUTES:
            raise ValueError(f"{prefix}: invalid expect_route")
        turns = item.get("turns")
        if (
            not isinstance(turns, list)
            or not turns
            or not all(isinstance(turn, str) and turn.strip() for turn in turns)
        ):
            raise ValueError(f"{prefix}: turns must be a non-empty list of non-empty strings")
        scored = item.get("scored_question")
        if scored is not None and (not isinstance(scored, str) or not scored.strip()):
            raise ValueError(f"{prefix}: scored_question must be a non-empty string or null")
        if len(turns) > 1 and scored is None:
            raise ValueError(f"{prefix}: multi-turn questions require scored_question")
        repeat = item.get("repeat", 1)
        if type(repeat) is not int or repeat < 1:
            raise ValueError(f"{prefix}: repeat must be an integer >= 1")
        for key in ("held_out", "expect_fixed", "accept_fixed"):
            if type(item.get(key, False)) is not bool:
                raise ValueError(f"{prefix}: {key} must be a boolean")
        reference = item.get("reference")
        if reference is not None and not isinstance(reference, str):
            raise ValueError(f"{prefix}: reference must be a string or null")
        lists: dict[str, tuple[str, ...]] = {}
        for key in ("expect_sections", "must_contain", "must_not_contain"):
            values = item.get(key, [])
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"{prefix}: {key} must be a list of strings")
            lists[key] = tuple(values)
        questions.append(
            Question(
                id=ident,
                category=category,
                turns=tuple(turns),
                held_out=item.get("held_out", False),
                scored_question=scored,
                reference=reference,
                expect_route=item.get("expect_route", "general"),
                expect_fixed=item.get("expect_fixed", False),
                accept_fixed=item.get("accept_fixed", False),
                repeat=repeat,
                **lists,
            )
        )
    return questions


def select_questions(
    questions: list[Question], *, split: str = "tune", only: str | None = None
) -> list[Question]:
    if split not in ("tune", "held_out", "all"):
        raise ValueError("invalid split")
    wanted = set(only.split(",")) if only is not None else None
    if wanted is not None:
        unknown = wanted - {question.id for question in questions}
        if unknown:
            raise ValueError(f"unknown --only id: {', '.join(sorted(unknown))}")
    return [
        question
        for question in questions
        if (split == "all" or question.held_out == (split == "held_out"))
        and (wanted is None or question.id in wanted)
    ]


def make_graph(deps: GraphDeps, *, route: str = "default") -> Any:
    if route == "gemini-answer":
        from evals.failing_chat import FailingChat

        claude = dataclasses.replace(deps.claude, chat=FailingChat())
        deps = dataclasses.replace(deps, claude=claude)
    return build_graph(deps, InMemorySaver(serde=memory.checkpoint_serde()))


async def _attempt(question: Question, graph: Any, meter: MeterLike) -> SampleResult:
    async with asyncio.timeout(SAMPLE_TIMEOUT_S):
        config = {"configurable": {"thread_id": uuid4().hex}, "callbacks": meter.callbacks()}
        for index, text in enumerate(question.turns):
            stream = graph.astream(
                {"question": text, "history": []},
                config,
                stream_mode=["messages", "custom"],
                version="v2",
            )
            async for _ in events(stream):
                pass
            state = (await graph.aget_state(config)).values
            if index < len(question.turns) - 1:
                turn: memory.StoredTurn = {
                    "turn_id": uuid4().hex,
                    "question": text,
                    "answer": state.get("answer_text") or "",
                    "embedding": state.get("question_embedding") or b"",
                }
                turns = memory.add_turn(state.get("turns") or [], turn)
                await graph.aupdate_state(config, {"turns": turns}, as_node="finish")
        return SampleResult(
            answer=state["answer_text"],
            citations=tuple(state["citations"]),
            sections=tuple(
                {"url": section.url, "heading": section.heading, "text": section.text}
                for section in state["sections"]
            ),
            route=(
                "private" if state["private"] else "off_topic" if state["off_topic"] else "general"
            ),
            check=state["check"],
            private_check=state["private_check"],
            nodes=tuple(state["nodes"]),
        )


def _fallback_providers(result: SampleResult) -> list[str]:
    expected = {"rewrite": "anthropic", "answer": "anthropic", "check": "google"}
    return sorted(
        {
            node["provider"]
            for node in result.nodes
            if node["name"] in expected
            and node.get("provider") is not None
            and node["provider"] != expected[node["name"]]
        }
    )


def _passes(questions: list[Question], repeats: int) -> Iterator[list[tuple[Question, int]]]:
    for repeat in range(1, repeats + 1):
        yield [(question, repeat) for question in questions]
    extra = [
        (question, repeat)
        for question in questions
        for repeat in range(repeats + 1, max(question.repeat, repeats) + 1)
    ]
    if extra:
        yield extra


@dataclasses.dataclass(frozen=True)
class RunOutcome:
    rows: list[dict[str, Any]]
    status: str
    reason: str | None

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "complete" else 1


async def evaluate(
    questions: list[Question],
    *,
    graph: Any,
    meter: MeterLike,
    scorer: Scorer | None,
    repeats: int,
    run_id: str,
    out_path: Path,
    route: str = "default",
    no_ragas: bool = False,
    split: str = "tune",
    summary_out: Path | None = None,
    date: str | None = None,
    prompt_version: str = PROMPT_VERSION,
    models: dict[str, str | None] | None = None,
) -> RunOutcome:
    rows: list[dict[str, Any]] = []
    status, reason = "complete", None
    date = date or datetime.now(UTC).date().isoformat()
    streak = 0
    error_streak = 0
    seen_providers: set[str] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("a", encoding="utf-8") as output:
            for samples in _passes(questions, repeats):
                meter.start_pass()
                for question, repeat in samples:
                    recorded = False
                    try:
                        # Ledger reads and writes off the loop: another process holding the
                        # ledger's lock must not stall the event loop (evals/cost.py).
                        await asyncio.to_thread(meter.check)
                        result = SampleResult()
                        error_name = None
                        for attempt in range(2):
                            if attempt:
                                await asyncio.to_thread(meter.check)
                            try:
                                result = await _attempt(question, graph, meter)
                                break
                            except (CapReached, UnknownModel):
                                raise
                            except Exception as error:  # noqa: BLE001 - retry whole sample once
                                if attempt:
                                    error_name = type(error).__name__
                        providers = _fallback_providers(result) if route == "default" else []
                        if providers:
                            error_name = "fallback"
                        if error_name:
                            result = SampleResult()
                        checks = (
                            failed_checks(question)
                            if error_name
                            else run_code_checks(question, result)
                        )
                        requested = (
                            metrics_for(question, result.answer)
                            if not error_name and not no_ragas
                            else ()
                        )
                        ragas = None
                        if requested:
                            await asyncio.to_thread(meter.check)
                            value = ScoreInput(
                                user_input=question.scored_question or question.turns[-1],
                                response=result.answer,
                                retrieved_contexts=[section["text"] for section in result.sections],
                                reference=question.reference,
                                metrics=requested,
                            )
                            try:
                                if scorer is None:
                                    raise ValueError("a scorer is required unless no_ragas is set")
                                async with asyncio.timeout(SCORE_TIMEOUT_S):
                                    ragas = await scorer(value, meter)
                            except (CapReached, UnknownModel):
                                raise
                            except Exception:  # noqa: BLE001 - judge failure becomes missing scores
                                ragas = dict.fromkeys(METRICS)
                        cost = await asyncio.to_thread(meter.record, question.id, repeat)
                        recorded = True
                        row = {
                            "run_id": run_id,
                            "id": question.id,
                            "category": question.category,
                            "held_out": question.held_out,
                            "repeat": repeat,
                            "status": "error" if error_name else "ok",
                            "error": error_name,
                            "turns": list(question.turns),
                            "scored_question": question.scored_question or question.turns[-1],
                            "answer": result.answer,
                            "citations": list(result.citations),
                            "sections": list(result.sections),
                            "route": result.route,
                            "check": result.check,
                            "private_check": result.private_check,
                            "nodes": list(result.nodes),
                            "code_checks": checks,
                            "ragas": ragas,
                            "cost_usd": cost,
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        rows.append(
                            {
                                **row,
                                "passed": not error_name and sample_passed(checks),
                                "requested_metrics": requested,
                                "accept_fixed": question.accept_fixed,
                                "fixed_answer": result.answer.strip() == PRIVATE_FALLBACK,
                            }
                        )
                        print(f"{question.id} {repeat} {row['status']} {cost:.6f}", flush=True)
                    finally:
                        if not recorded:
                            with suppress(CapReached, UnknownModel):
                                await asyncio.to_thread(meter.record, question.id, repeat)
                    if error_name == "fallback":
                        streak += 1
                        seen_providers.update(providers)
                    else:
                        streak = 0
                        seen_providers.clear()
                    error_streak = error_streak + 1 if error_name else 0
                    if streak >= 3:
                        status = "incomplete"
                        reason = "three consecutive fallbacks; providers: " + ", ".join(
                            sorted(seen_providers)
                        )
                        break
                    if error_streak >= 3:
                        status = "incomplete"
                        reason = "three consecutive errors; last: " + str(error_name)
                        break
                if status == "incomplete":
                    break
    except (CapReached, UnknownModel) as error:
        status, reason = "incomplete", type(error).__name__
    except Exception as error:  # noqa: BLE001 - preserve completed rows after any run failure
        status, reason = "incomplete", type(error).__name__
    if status == "incomplete":
        print(f"eval incomplete: {reason}", file=sys.stderr)
    meta = {
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "split": split,
        "repeats": repeats,
        "route": route,
        "questions": len(questions),
        "samples": len(rows),
    }
    try:
        out_path.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if status == "complete" and summary_out is not None:
            write_summary(
                build_summary(
                    rows,
                    run_id=run_id,
                    date=date,
                    prompt_version=prompt_version,
                    route=route,
                    split=split,
                    repeats=repeats,
                    questions=len(questions),
                    models={**(models or {}), **({"judge": None} if no_ragas else {})},
                    cost_pools=meter.run_cost,
                ),
                summary_out,
            )
    except Exception as error:  # noqa: BLE001 - output failures make the run incomplete too
        status, reason = "incomplete", type(error).__name__
        print(f"eval incomplete: {reason}", file=sys.stderr)
        meta.update(status=status, reason=reason)
        with suppress(OSError):
            out_path.with_suffix(".meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return RunOutcome(rows, status, reason)


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite float > 0")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True, dest="question_set")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    for flag in ("pass-cap", "total-cap", "credit-pass-cap", "credit-total-cap"):
        parser.add_argument(f"--{flag}", type=_positive_float, required=True)
    parser.add_argument("--split", choices=("tune", "held_out", "all"), default="tune")
    parser.add_argument("--only")
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--route", choices=("default", "gemini-answer"), default="default")
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--summary-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        questions = select_questions(
            load_questions(args.question_set), split=args.split, only=args.only
        )
    except ValueError as error:
        print(f"invalid question set: {error}", file=sys.stderr)
        return 2
    try:
        from app.config import get_settings
        from app.db import Database, make_pool
        from app.qa.graph import build_deps
        from app.rag.embed import AzureEmbedder
        from app.rag.retrieve import retrieve
        from evals.cost import Meter

        score = None
        if not args.no_ragas:
            from evals.ragas_score import score
    except ImportError:
        print(
            "eval modules missing: install the eval group and the cost/scoring modules",
            file=sys.stderr,
        )
        return 2
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%dT%H%M%S}-{PROMPT_VERSION}-{args.route}"
    caps = {
        "real": (args.pass_cap, args.total_cap),
        "credit": (args.credit_pass_cap, args.credit_total_cap),
    }

    async def run() -> int:
        settings = get_settings()
        db = Database(make_pool(settings.database_url.get_secret_value()))
        try:
            meter = Meter(caps, args.ledger, run_id)
            embedder = MeteredEmbedder(AzureEmbedder(settings), meter, settings.embed_model)
            deps = build_deps(settings, functools.partial(retrieve, db, embedder), embedder)
            outcome = await evaluate(
                questions,
                graph=make_graph(deps, route=args.route),
                meter=meter,
                scorer=score,
                repeats=args.repeats,
                run_id=run_id,
                out_path=args.out / f"{run_id}.jsonl",
                route=args.route,
                no_ragas=args.no_ragas,
                split=args.split,
                summary_out=args.summary_out,
                date=started.date().isoformat(),
                models={
                    "judge": None if args.no_ragas else JUDGE_MODEL,
                    "embed": settings.embed_model,
                },
            )
            return outcome.exit_code
        finally:
            await db.close()

    try:
        return asyncio.run(run())
    except Exception as error:  # noqa: BLE001 - setup failures must not expose config or prompts
        print(f"eval setup/cleanup failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
