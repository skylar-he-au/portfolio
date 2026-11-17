# ruff: noqa: RUF001 - Chinese fixtures use full-width sentence punctuation
"""Offline eval contracts, using scripted meters, scorers and graph providers."""

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fakes import FakeEmbedder
from test_graph import FakeChat, FakeRetriever, check_reply, rewrite_reply

from app.qa.graph import NOT_FOUND, PRIVATE_FALLBACK, GraphDeps
from app.qa.models import ChatProvider
from app.rag.types import Section
from evals.checks import (
    Question,
    SampleResult,
    count_sentences,
    failed_checks,
    metrics_for,
    run_code_checks,
    sample_passed,
)
from evals.failing_chat import FailingChat
from evals.run import MeteredEmbedder, evaluate, load_questions, main, make_graph, select_questions
from evals.summary import build_summary
from evals.types import METRICS, CapReached, ScoreInput, UnknownModel

pytestmark = pytest.mark.anyio

ANSWER = "The kite sorts seeds [1]. It uses colours [1]. It records ties [1]."
SECTION = Section("/projects/kite", "sorting", "Sorting", "Kite sorts invented seeds.")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeMeter:
    def __init__(self, fail_check: int | None = None, error: type[Exception] = CapReached) -> None:
        self.calls: list[Any] = []
        self.checks = 0
        self.fail_check = fail_check
        self.error = error
        self.fail_record = False

    def check(self) -> None:
        self.calls.append("check")
        self.checks += 1
        if self.checks == self.fail_check:
            raise self.error("scripted stop")

    def start_pass(self) -> None:
        self.calls.append("start_pass")

    def record(self, question_id: str, repeat: int) -> float:
        self.calls.append(("record", question_id, repeat))
        if self.fail_record:
            raise self.error("scripted record stop")
        return 0.01

    def callbacks(self) -> list[Any]:
        return []

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.calls.append(("usage", model, input_tokens, output_tokens))

    def add_embedding_chars(self, model: str, chars: int) -> None:
        self.calls.append(("embedding", model, chars))

    @property
    def run_cost(self) -> dict[str, float]:
        return {"real": 0.0, "credit": 0.0}


def state(*, fallback: bool = False) -> dict[str, Any]:
    return {
        "answer_text": ANSWER,
        "citations": [{"n": 1, "url": SECTION.url, "heading": SECTION.heading}],
        "sections": [SECTION],
        "private": False,
        "off_topic": False,
        "check": {"passed": True},
        "private_check": None,
        "nodes": [
            {"name": "answer", "provider": "anthropic", "model": "fake-answer", "ms": 0},
            {
                "name": "check",
                "provider": "anthropic" if fallback else "google",
                "model": "fake-check",
                "ms": 0,
            },
        ],
    }


class TinyGraph:
    def __init__(self, script: list[Any] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[tuple[str, str]] = []
        self.current = state()

    async def astream(self, value: dict, config: dict, **kwargs: Any) -> AsyncIterator[dict]:
        assert kwargs == {"stream_mode": ["messages", "custom"], "version": "v2"}
        self.calls.append((value["question"], config["configurable"]["thread_id"]))
        reply = self.script.pop(0) if self.script else state()
        if isinstance(reply, Exception):
            raise reply
        if reply == "timeout":
            await asyncio.sleep(1)
        self.current = reply
        if False:
            yield {}

    async def aget_state(self, config: dict) -> SimpleNamespace:
        return SimpleNamespace(values=self.current)

    async def aupdate_state(self, config: dict, update: dict, *, as_node: str) -> None:
        assert as_node == "finish"
        self.current.update(update)


async def scorer(value: ScoreInput, meter: FakeMeter) -> dict[str, float | None]:
    return {metric: 0.8 if metric in value.metrics else None for metric in METRICS}


def question(**kwargs: Any) -> Question:
    base = Question(id="kite-sort", category="project", turns=("How does Kite sort?",))
    return replace(base, **kwargs)


async def run_eval(tmp_path: Path, questions: list[Question] | None = None, **kw: Any) -> Any:
    options = {
        "graph": TinyGraph(),
        "meter": FakeMeter(),
        "scorer": scorer,
        "repeats": 1,
        "run_id": "invented-run",
        "out_path": tmp_path / "raw.jsonl",
        "route": "default",
        "no_ragas": True,
        "summary_out": tmp_path / "public",
    }
    options.update(kw)
    return await evaluate(questions or [question()], **options)


def summary(rows: list[dict], repeats: int = 1, questions: int = 1) -> dict:
    return build_summary(
        rows,
        run_id="invented-run",
        date="2026-01-01",
        prompt_version="v-test",
        route="default",
        split="all",
        repeats=repeats,
        questions=questions,
        models={"judge": "fake-judge", "embed": "fake-embed"},
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("First [1][2]. Second [1, 2]! Third?", 3),
        ("It is 0.5. Use e.g.x here. Last", 3),
        ("甲。乙！？丙", 3),
        ("甲乙", 1),
        ("", 0),
    ],
)
def test_sentence_count(text: str, expected: int) -> None:
    assert count_sentences(text) == expected


@pytest.mark.parametrize(
    ("changes", "result", "key", "expected"),
    [
        ({}, SampleResult(answer=ANSWER, citations=({},), route="general"), "has_citation", True),
        ({}, SampleResult(answer=ANSWER, route="general"), "has_citation", False),
        ({}, SampleResult(answer=ANSWER), "sentence_count", True),
        ({}, SampleResult(answer="One."), "sentence_count", False),
        ({"turns": ("甲？",)}, SampleResult(answer="甲。乙。丙。"), "sentence_count", True),
        ({"turns": ("甲？",)}, SampleResult(answer="甲。"), "sentence_count", False),
        ({"expect_route": "private"}, SampleResult(answer="One."), "sentence_count", True),
        ({"expect_route": "private"}, SampleResult(answer="A. B. C. D."), "sentence_count", False),
        ({"expect_fixed": True}, SampleResult(answer=PRIVATE_FALLBACK), "fixed", True),
        ({"expect_fixed": True}, SampleResult(answer="Other."), "fixed", False),
        ({"must_not_contain": ("SEED",)}, SampleResult(answer=ANSWER), "must_not_contain", False),
        ({"must_not_contain": ("secret",)}, SampleResult(answer=ANSWER), "must_not_contain", True),
        ({"must_contain": ("SEED",)}, SampleResult(answer=ANSWER), "must_contain", True),
        ({"must_contain": ("missing",)}, SampleResult(answer=ANSWER), "must_contain", False),
        ({}, SampleResult(answer="English."), "language", True),
        ({}, SampleResult(answer="甲。"), "language", False),
        ({"turns": ("甲？",)}, SampleResult(answer="甲。"), "language", True),
        ({"turns": ("甲？",)}, SampleResult(answer="English."), "language", False),
        ({}, SampleResult(route="general"), "route", True),
        ({}, SampleResult(route="private"), "route", False),
        (
            {"expect_sections": (SECTION.url,)},
            SampleResult(sections=({"url": SECTION.url},)),
            "sections_hit",
            True,
        ),
        ({"expect_sections": (SECTION.url,)}, SampleResult(), "sections_hit", False),
    ],
)
def test_checks(changes: dict, result: SampleResult, key: str, expected: bool) -> None:
    assert run_code_checks(question(**changes), result)[key] is expected


def test_applicability_and_metrics() -> None:
    q = question(accept_fixed=True, must_contain=("seed",), expect_sections=(SECTION.url,))
    checks = run_code_checks(q, SampleResult(answer=PRIVATE_FALLBACK, route="general"))
    assert sample_passed(checks)
    assert all(
        checks[key] is None
        for key in ("fixed", "has_citation", "sentence_count", "must_contain", "sections_hit")
    )
    assert run_code_checks(q, SampleResult(answer=NOT_FOUND))["has_citation"] is None
    assert metrics_for(question(), ANSWER) == METRICS
    assert metrics_for(question(category="private_answerable"), ANSWER) == ("faithfulness",)
    assert metrics_for(question(category="private_answerable"), PRIVATE_FALLBACK) == ()
    assert metrics_for(question(category="injection"), ANSWER) == ()
    assert not sample_passed(failed_checks(q))


@pytest.mark.parametrize("error", [CapReached, UnknownModel])
async def test_cap_before_third_sample(tmp_path: Path, error: type[Exception]) -> None:
    meter = FakeMeter(fail_check=3, error=error)
    graph = TinyGraph()
    outcome = await run_eval(tmp_path, repeats=4, graph=graph, meter=meter)
    assert outcome.status == "incomplete" and outcome.exit_code == 1
    assert len(outcome.rows) == len(graph.calls) == 2
    assert meter.calls.count(("record", "kite-sort", 3)) == 1
    assert len((tmp_path / "raw.jsonl").read_text().splitlines()) == 2
    assert not (tmp_path / "public").exists()
    assert json.loads((tmp_path / "raw.meta.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize("error", [CapReached, UnknownModel])
@pytest.mark.parametrize("source", ["graph", "scorer", "record"])
async def test_stops_are_not_retried(tmp_path: Path, error: type[Exception], source: str) -> None:
    graph = TinyGraph([error("stop")]) if source == "graph" else TinyGraph()
    meter = FakeMeter(error=error)
    meter.fail_record = source == "record"

    async def stop(value: ScoreInput, meter: FakeMeter) -> dict:
        raise error("stop")

    outcome = await run_eval(
        tmp_path,
        graph=graph,
        meter=meter,
        scorer=stop if source == "scorer" else scorer,
        no_ragas=False,
    )
    assert outcome.exit_code == 1 and outcome.rows == []
    assert len(graph.calls) == 1
    assert meter.calls.count(("record", "kite-sort", 1)) == (2 if source == "record" else 1)


async def test_first_check_still_records(tmp_path: Path) -> None:
    meter = FakeMeter(fail_check=1)
    graph = TinyGraph()
    outcome = await run_eval(tmp_path, graph=graph, meter=meter)
    assert outcome.rows == [] and graph.calls == [] and outcome.exit_code == 1
    assert meter.calls == ["start_pass", "check", ("record", "kite-sort", 1)]


@pytest.mark.parametrize("retrying", [False, True])
async def test_stop_before_scoring_or_retry(tmp_path: Path, retrying: bool) -> None:
    meter = FakeMeter(fail_check=2)
    graph = TinyGraph([ValueError("hidden")]) if retrying else TinyGraph()
    outcome = await run_eval(tmp_path, graph=graph, meter=meter, no_ragas=retrying)
    assert outcome.exit_code == 1 and outcome.rows == [] and len(graph.calls) == 1
    assert meter.calls == ["start_pass", "check", "check", ("record", "kite-sort", 1)]


async def test_pass_order_and_extra_pass(tmp_path: Path) -> None:
    meter = FakeMeter()
    outcome = await run_eval(
        tmp_path, [question(repeat=3), question(id="kite-ties")], repeats=2, meter=meter
    )
    assert [(r["id"], r["repeat"]) for r in outcome.rows] == [
        ("kite-sort", 1),
        ("kite-ties", 1),
        ("kite-sort", 2),
        ("kite-ties", 2),
        ("kite-sort", 3),
    ]
    assert meter.calls == [
        "start_pass",
        "check",
        ("record", "kite-sort", 1),
        "check",
        ("record", "kite-ties", 1),
        "start_pass",
        "check",
        ("record", "kite-sort", 2),
        "check",
        ("record", "kite-ties", 2),
        "start_pass",
        "check",
        ("record", "kite-sort", 3),
    ]


async def test_fallback_streak(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    graph = TinyGraph([state(fallback=True), state(), *[state(fallback=True) for _ in range(3)]])
    outcome = await run_eval(tmp_path, graph=graph, repeats=6)
    assert outcome.status == "incomplete" and len(graph.calls) == 5
    assert [r["error"] for r in outcome.rows] == [
        "fallback",
        None,
        "fallback",
        "fallback",
        "fallback",
    ]
    assert outcome.rows[0]["answer"] == "" and outcome.rows[0]["nodes"] == []
    assert "anthropic" in capsys.readouterr().err


async def test_embedder_metering() -> None:
    inner, meter = FakeEmbedder(), FakeMeter()
    wrapped = MeteredEmbedder(inner, meter, "fake-embed")
    result = await wrapped.embed(["abc", "甲乙"])
    assert result == await inner.embed(["abc", "甲乙"])
    await wrapped.embed(["x"])
    assert meter.calls == [("embedding", "fake-embed", 5), ("embedding", "fake-embed", 1)]


def deps(claude: FakeChat, gemini: FakeChat, count: int = 1) -> GraphDeps:
    return GraphDeps(
        ChatProvider("anthropic", "invented-claude", claude),
        ChatProvider("google", "invented-gemini", gemini),
        FakeRetriever(*[[SECTION] for _ in range(count)]),
        "fake-embed",
        embedder=FakeEmbedder(),
    )


async def test_multiturn_real_graph(tmp_path: Path) -> None:
    claude = FakeChat(
        script=[rewrite_reply("kite seeds"), ANSWER, rewrite_reply("kite ties"), ANSWER],
    )
    graph = make_graph(deps(claude, FakeChat(script=[check_reply(), check_reply()]), 2))
    q = question(
        turns=("Describe the invented Kite.", "How are its ties stored?"),
        scored_question="How does Kite store ties?",
    )
    captured: list[ScoreInput] = []

    async def capture(value: ScoreInput, meter: FakeMeter) -> dict:
        captured.append(value)
        return await scorer(value, meter)

    outcome = await run_eval(tmp_path, [q], graph=graph, scorer=capture, no_ragas=False)
    assert outcome.rows[0]["status"] == "ok"
    assert q.turns[0] in claude.calls[2][-1].text
    assert ANSWER in claude.calls[2][-1].text
    assert captured[0].user_input == q.scored_question
    assert captured[0].retrieved_contexts == [SECTION.text]


async def test_first_turn_failure_never_asks_second(tmp_path: Path) -> None:
    graph = TinyGraph([ValueError("private prompt"), ValueError("private prompt")])
    q = question(
        turns=("First invented turn?", "Second invented turn?"),
        scored_question="Standalone?",
    )
    outcome = await run_eval(tmp_path, [q], graph=graph)
    assert [text for text, _ in graph.calls] == [q.turns[0], q.turns[0]]
    assert outcome.rows[0]["error"] == "ValueError"
    assert graph.calls[0][1] != graph.calls[1][1]


async def test_retry_then_continue(tmp_path: Path) -> None:
    graph = TinyGraph(
        [ValueError("hidden"), state(), TypeError("hidden"), TypeError("hidden"), state()],
    )
    qs = [question(), question(id="kite-two"), question(id="kite-three")]
    outcome = await run_eval(tmp_path, qs, graph=graph)
    assert [r["status"] for r in outcome.rows] == ["ok", "error", "ok"]
    row = outcome.rows[1]
    assert row["error"] == "TypeError" and row["ragas"] is None
    assert all(v is False for v in row["code_checks"].values() if v is not None)
    assert "hidden" not in (tmp_path / "raw.jsonl").read_text()


async def test_timeout_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.run.SAMPLE_TIMEOUT_S", 0.001)
    graph = TinyGraph(["timeout", state()])
    outcome = await run_eval(tmp_path, graph=graph)
    assert outcome.rows[0]["status"] == "ok" and len(graph.calls) == 2


@pytest.mark.parametrize("raises", [False, True])
async def test_ragas_nulls(tmp_path: Path, raises: bool) -> None:
    async def null_score(value: ScoreInput, meter: FakeMeter) -> dict:
        if raises:
            raise ValueError("hidden prompt")
        return dict.fromkeys(METRICS)

    outcome = await run_eval(tmp_path, scorer=null_score, no_ragas=False)
    assert outcome.rows[0]["status"] == "ok"
    assert summary(outcome.rows)["ragas_nulls"] == len(METRICS)


async def test_extra_repeats_excluded_from_averages(tmp_path: Path) -> None:
    scores = iter([0.2, 0.4, 0.9, *[0.0] * 7])

    async def score(value: ScoreInput, meter: FakeMeter) -> dict:
        return dict.fromkeys(METRICS, next(scores))

    meter = FakeMeter()
    outcome = await run_eval(
        tmp_path,
        [question(repeat=10, accept_fixed=True)],
        repeats=3,
        scorer=score,
        meter=meter,
        no_ragas=False,
    )
    result = summary(outcome.rows, repeats=3)
    assert len(outcome.rows) == 10 and meter.calls.count("start_pass") == 4
    assert result["private_repeat"] == {"runs": 10, "fixed": 0, "answered": 10, "failed": 0}
    assert result["ragas"]["tune"]["faithfulness"] == {
        "mean": 0.5,
        "min": 0.2,
        "max": 0.9,
        "scored": 3,
        "below_floor": 2,
    }
    assert result["code_checks"]["project"] == {"passed": 3, "total": 3}


async def test_private_repeat_counts_and_scoring_selection(tmp_path: Path) -> None:
    fixed = dict(state(), answer_text=PRIVATE_FALLBACK, private=True)
    answered = dict(state(), answer_text="Kite keeps a colour [1].", private=True)
    bad = dict(state(), answer_text="Kite keeps a secret [1].", private=True)
    graph = TinyGraph([fixed, answered, bad, ValueError("hidden"), ValueError("hidden")])
    q = question(
        category="private_answerable",
        expect_route="private",
        accept_fixed=True,
        must_not_contain=("secret",),
        repeat=4,
    )
    requested: list[tuple[str, ...]] = []

    async def capture(value: ScoreInput, meter: FakeMeter) -> dict:
        requested.append(value.metrics)
        return await scorer(value, meter)

    outcome = await run_eval(tmp_path, [q], graph=graph, scorer=capture, no_ragas=False)
    assert requested == [("faithfulness",), ("faithfulness",)]
    assert outcome.rows[0]["ragas"] is None
    result = summary(outcome.rows)
    assert result["private_repeat"] == {"runs": 4, "fixed": 1, "answered": 1, "failed": 2}
    assert result["errors"] == 1
    assert result["code_checks"]["private_answerable"] == {"passed": 1, "total": 1}
    raw = json.loads((tmp_path / "raw.jsonl").read_text().splitlines()[0])
    assert not {"requested_metrics", "passed", "fixed_answer", "accept_fixed"} & raw.keys()


async def test_unexpected_run_failure_writes_metadata(tmp_path: Path) -> None:
    class BrokenMeter(FakeMeter):
        def start_pass(self) -> None:
            raise RuntimeError("hidden configuration")

    outcome = await run_eval(tmp_path, meter=BrokenMeter())
    assert outcome.exit_code == 1 and outcome.rows == []
    assert not (tmp_path / "public").exists()
    meta = json.loads((tmp_path / "raw.meta.json").read_text())
    assert meta["reason"] == "RuntimeError"
    assert "hidden" not in json.dumps(meta)


async def test_complete_publishes_summary_after_recording(tmp_path: Path) -> None:
    meter = FakeMeter()

    async def capture(value: ScoreInput, meter: FakeMeter) -> dict:
        meter.calls.append("score")
        return await scorer(value, meter)

    outcome = await run_eval(tmp_path, meter=meter, scorer=capture, no_ragas=False)
    assert outcome.exit_code == 0
    assert meter.calls == ["start_pass", "check", "check", "score", ("record", "kite-sort", 1)]
    public = json.loads((tmp_path / "public" / "invented-run.json").read_text())
    assert public["samples"] == 1 and public["errors"] == 0
    assert public["cost_usd"] == 0.01
    assert ANSWER not in json.dumps(public) and SECTION.text not in json.dumps(public)
    assert not list((tmp_path / "public").glob("*.tmp"))


async def test_summary_whitelist_and_pass_means(tmp_path: Path) -> None:
    secret = "ZZ-SECRET-QUESTION-ZZ"
    qs = [question(turns=(secret,)), question(id="kite-held", held_out=True, category="zh")]
    outcome = await run_eval(tmp_path, qs, repeats=3, no_ragas=False)
    for row in outcome.rows:
        row["answer"] = secret
        row["sections"] = [{"text": secret}]
    result = summary(outcome.rows, repeats=3, questions=2)
    assert secret not in json.dumps(result)
    assert "kite-held" not in json.dumps(result)
    assert set(result) == {
        "kind",
        "run_id",
        "date",
        "prompt_version",
        "route",
        "split",
        "repeats",
        "questions",
        "samples",
        "errors",
        "ragas_nulls",
        "models",
        "ragas",
        "code_checks",
        "private_repeat",
        "cost_usd",
    }
    assert set(result["ragas"]) == {"tune", "held_out", "zh"}
    assert result["models"]["answer"] == "fake-answer"
    # Unequal scored counts: average pass means, not all sample scores.
    rows = [
        dict(outcome.rows[0], repeat=p, ragas=dict.fromkeys(METRICS, v))
        for p, v in [(1, 0.2), (1, 0.4), (2, 0.9), (3, None)]
    ]
    metric = summary(rows, repeats=3)["ragas"]["tune"]["faithfulness"]
    assert metric == {"mean": 0.6, "min": 0.3, "max": 0.9, "scored": 3, "below_floor": 2}


@pytest.mark.parametrize("split", ["tune", "held_out", "all"])
async def test_split_and_only(tmp_path: Path, split: str) -> None:
    qs = [question(), question(id="kite-held", held_out=True)]
    selected = select_questions(qs, split=split)
    assert len(selected) == (2 if split == "all" else 1)
    assert select_questions(qs, split="all", only="kite-held") == [qs[1]]
    with pytest.raises(ValueError, match="unknown"):
        select_questions(qs, split=split, only="missing")
    outcome = await run_eval(tmp_path, selected, split=split)
    assert ("held_out" in summary(outcome.rows)["ragas"]) == (split != "tune")


async def test_gemini_route(tmp_path: Path) -> None:
    gemini = FakeChat(script=[rewrite_reply("kite sorting"), ANSWER, check_reply()])
    graph = make_graph(deps(FakeChat(), gemini), route="gemini-answer")
    outcome = await run_eval(tmp_path, graph=graph, route="gemini-answer")
    row = outcome.rows[0]
    answer_node = next(n for n in row["nodes"] if n["name"] == "answer")
    assert answer_node["provider"] == "google" and row["status"] == "ok"
    assert summary(outcome.rows)["models"]["answer"] == answer_node["model"]


async def test_gemini_check_failure_is_sample_error(tmp_path: Path) -> None:
    gemini = FakeChat(
        script=[rewrite_reply("kite sorting"), ANSWER, ValueError("hidden check")] * 2,
    )
    graph = make_graph(deps(FakeChat(), gemini, 2), route="gemini-answer")
    outcome = await run_eval(tmp_path, graph=graph, route="gemini-answer")
    assert outcome.rows[0]["error"] == "ModelsUnavailable"
    assert len(gemini.calls) == 6


async def test_failing_chat() -> None:
    chat = FailingChat()
    assert chat.with_structured_output(dict) is chat
    with pytest.raises(RuntimeError, match="Claude is switched off"):
        await chat.ainvoke("invented")
    with pytest.raises(RuntimeError, match="Claude is switched off"):
        async for _ in chat.astream("invented"):
            pass


def test_cli_help_and_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for option in (
        "set",
        "out",
        "ledger",
        "pass-cap",
        "total-cap",
        "credit-pass-cap",
        "credit-total-cap",
        "split",
        "only",
        "repeats",
        "route",
        "no-ragas",
        "summary-out",
    ):
        assert f"--{option}" in text
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_cli_caps_are_required_without_defaults() -> None:
    from evals.run import _parser

    parser = _parser()
    actions = {action.dest: action for action in parser._actions}
    for name in ("pass_cap", "total_cap", "credit_pass_cap", "credit_total_cap"):
        action = actions[name]
        assert action.required and action.default is None
        for invalid in ("nan", "inf", "-inf"):
            with pytest.raises(argparse.ArgumentTypeError, match="finite float"):
                action.type(invalid)


def test_loader(tmp_path: Path) -> None:
    path = tmp_path / "set.yaml"
    base = {"id": "kite-sort", "category": "project", "turns": ["Invented sorting?"]}
    path.write_text(yaml.safe_dump({"version": 1, "questions": [base]}))
    q = load_questions(path)[0]
    assert q.repeat == 1 and not q.held_out and q.expect_route == "general"
    assert q.scored_question is None and q.must_contain == ()
    for questions, message in [
        ([base, base], "duplicate"),
        ([dict(base, turns=["Invented first?", "Invented next?"])], "scored_question"),
        ([dict(base, category="unknown")], "category"),
        ([dict(base, repeat=0)], "repeat"),
        ([dict(base, id="BAD_ID")], "id"),
        ([dict(base, expect_route="unknown")], "route"),
        ([dict(base, turns=[])], "turns"),
    ]:
        path.write_text(yaml.safe_dump({"version": 1, "questions": questions}))
        with pytest.raises(ValueError, match=message):
            load_questions(path)


def test_private_answerable_with_empty_must_contain_loads_and_passes(tmp_path: Path) -> None:
    path = tmp_path / "set.yaml"
    item = {
        "id": "pa-kite-2",
        "category": "private_answerable",
        "turns": ["Invented availability question?"],
        "expect_route": "private",
        "accept_fixed": True,
        "must_contain": [],
    }
    path.write_text(yaml.safe_dump({"version": 1, "questions": [item]}))
    q = load_questions(path)[0]
    assert q.accept_fixed and q.must_contain == ()
    checks = run_code_checks(q, SampleResult(answer="Invented answer.", route="private"))
    assert checks["must_contain"] is None
    assert sample_passed(checks)
