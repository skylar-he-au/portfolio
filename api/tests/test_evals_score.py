"""Metric orchestration and the Anthropic request contract without network access."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic
from pydantic import BaseModel, SecretStr

from evals.cost import Meter
from evals.types import METRICS, CapReached, MeterLike, ScoreInput, UnknownModel

pytest.importorskip("ragas")

from evals import ragas_score


class FakeMeter:
    def __init__(self, error: Exception | None = None) -> None:
        self.checks = 0
        self.error = error

    def check(self) -> None:
        self.checks += 1
        if self.error is not None:
            raise self.error

    def callbacks(self) -> list[Any]:
        return []

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        pass

    def add_embedding_chars(self, model: str, chars: int) -> None:
        pass

    def record(self, question_id: str, repeat: int) -> float:
        return 0.0

    def start_pass(self) -> None:
        pass

    @property
    def run_cost(self) -> dict[str, float]:
        return {"real": 0.0, "credit": 0.0}


class FakeMetric:
    def __init__(
        self, name: str, outcome: float | Exception, calls: list[tuple[str, dict[str, Any]]]
    ) -> None:
        self.name = name
        self.outcome = outcome
        self.calls = calls

    async def ascore(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((self.name, kwargs))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return SimpleNamespace(value=self.outcome)


class WireModel(BaseModel):
    x: int


def _item(*, metrics: tuple[str, ...] = METRICS, reference: str | None = "reference") -> ScoreInput:
    return ScoreInput(
        user_input="question",
        response="answer",
        retrieved_contexts=["context one", "context two"],
        reference=reference,
        metrics=metrics,
    )


def _install_metrics(
    monkeypatch: pytest.MonkeyPatch, outcomes: dict[str, float | Exception] | None = None
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[MeterLike, str, str]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    builds: list[tuple[MeterLike, str, str]] = []
    values = outcomes or {}

    def build(meter: MeterLike, judge_model: str, embed_model: str) -> dict[str, Any]:
        builds.append((meter, judge_model, embed_model))
        return {name: FakeMetric(name, values.get(name, 0.75), calls) for name in METRICS}

    monkeypatch.setattr(ragas_score, "_build_metrics", build)
    return calls, builds


@pytest.mark.parametrize(
    "bad_value", [float("nan"), ValueError("metric failure")], ids=["nan", "error"]
)
def test_metric_failure_is_isolated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad_value: float | Exception
) -> None:
    calls, builds = _install_metrics(monkeypatch, {"faithfulness": bad_value})
    meter = FakeMeter()
    with caplog.at_level("WARNING", logger=ragas_score.__name__):
        result = asyncio.run(ragas_score.score(_item(), meter))
    assert result == {"faithfulness": None, "context_recall": 0.75}
    assert [name for name, _ in calls] == list(METRICS)
    assert meter.checks == 2
    assert len(builds) == 1
    if isinstance(bad_value, Exception):
        assert any(
            "faithfulness" in record.message and record.levelname == "WARNING"
            for record in caplog.records
        )


def test_metric_kwargs_match_ragas_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _ = _install_metrics(monkeypatch)
    meter = FakeMeter()
    result = asyncio.run(ragas_score.score(_item(), meter))
    assert list(result) == list(METRICS)
    assert calls == [
        (
            "faithfulness",
            {
                "user_input": "question",
                "response": "answer",
                "retrieved_contexts": ["context one", "context two"],
            },
        ),
        (
            "context_recall",
            {
                "user_input": "question",
                "retrieved_contexts": ["context one", "context two"],
                "reference": "reference",
            },
        ),
    ]


def test_subset_runs_and_metrics_no_longer_judged_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Step 10: response_relevancy and factual_correctness are not judged even when asked.
    calls, _ = _install_metrics(monkeypatch)
    meter = FakeMeter()
    result = asyncio.run(
        ragas_score.score(_item(metrics=("factual_correctness", "faithfulness")), meter)
    )
    assert result == {"faithfulness": 0.75, "context_recall": None}
    assert [name for name, _ in calls] == ["faithfulness"]
    assert meter.checks == 1


def test_missing_reference_skips_reference_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _ = _install_metrics(monkeypatch)
    meter = FakeMeter()
    result = asyncio.run(ragas_score.score(_item(reference=None), meter))
    assert result == {"faithfulness": 0.75, "context_recall": None}
    assert [name for name, _ in calls] == ["faithfulness"]
    assert meter.checks == 1


def test_nothing_requested_does_not_build_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, builds = _install_metrics(monkeypatch)
    meter = FakeMeter()
    result = asyncio.run(ragas_score.score(_item(metrics=()), meter))
    assert result == dict.fromkeys(METRICS)
    assert calls == []
    assert builds == []
    assert meter.checks == 0


@pytest.mark.parametrize("error_type", [CapReached, UnknownModel])
def test_check_errors_propagate_before_building(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    calls, builds = _install_metrics(monkeypatch)
    error = error_type("stop")
    meter = FakeMeter(error)
    with pytest.raises(error_type) as caught:
        asyncio.run(ragas_score.score(_item(), meter))
    assert caught.value is error
    assert meter.checks == 1
    assert calls == []
    assert builds == []


@pytest.mark.parametrize("error_type", [CapReached, UnknownModel])
def test_control_errors_inside_metrics_are_never_swallowed(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    error = error_type("stop")
    calls, _ = _install_metrics(monkeypatch, {"faithfulness": error})
    meter = FakeMeter()
    with pytest.raises(error_type) as caught:
        asyncio.run(ragas_score.score(_item(), meter))
    assert caught.value is error
    assert meter.checks == 1
    assert [name for name, _ in calls] == ["faithfulness"]


def test_checks_run_before_each_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    meter = FakeMeter()

    class SpendingMetric:
        async def ascore(self, **kwargs: Any) -> SimpleNamespace:
            meter.error = CapReached("spent")
            return SimpleNamespace(value=0.5)

    def build(meter: MeterLike, judge_model: str, embed_model: str) -> dict[str, Any]:
        return dict.fromkeys(METRICS, SpendingMetric())

    monkeypatch.setattr(ragas_score, "_build_metrics", build)
    with pytest.raises(CapReached, match="spent"):
        asyncio.run(ragas_score.score(_item(), meter))
    assert meter.checks == 2


def test_cache_is_per_meter_and_model_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    _, builds = _install_metrics(monkeypatch)
    meter = FakeMeter()
    item = _item(metrics=("faithfulness",))

    async def invoke() -> None:
        await ragas_score.score(item, meter)
        await ragas_score.score(item, meter)
        await ragas_score.score(item, meter, judge_model="another-judge")
        await ragas_score.score(item, meter, embed_model="another-embedding")
        await ragas_score.score(item, FakeMeter())

    asyncio.run(invoke())
    assert len(builds) == 4
    assert [entry[1:] for entry in builds] == [
        ("claude-sonnet-5-5", "text-embedding-3-small"),
        ("another-judge", "text-embedding-3-small"),
        ("claude-sonnet-5-5", "another-embedding"),
        ("claude-sonnet-5-5", "text-embedding-3-small"),
    ]
    assert builds[0][0] is meter
    assert builds[-1][0] is not meter


def test_make_llm_wire_request_and_metering(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": '{"x": 1}'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )

    meter = Meter({"real": (3.0, 10.0), "credit": (3.0, 10.0)}, tmp_path / "ledger.csv", "wire")

    async def invoke() -> WireModel:
        async with AsyncAnthropic(
            api_key="test",
            base_url="http://eval.test",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
            max_retries=0,
        ) as client:
            llm = ragas_score.make_llm(meter, "claude-sonnet-5-5", client=client)
            return await llm.agenerate("prompt", WireModel)

    assert asyncio.run(invoke()).x == 1
    assert len(bodies) == 1
    body = bodies[0]
    assert not {"temperature", "top_p", "top_k", "thinking"}.intersection(body)
    assert body["output_config"] == {"effort": "low"}
    assert body["max_tokens"] == 4096
    assert meter.run_cost["credit"] == pytest.approx(100 / 1e6 * 2 + 20 / 1e6 * 10)
    assert meter.run_cost["real"] == 0.0


@pytest.mark.parametrize(
    ("builder_name", "model", "key_name", "environment_name"),
    [
        ("make_llm", "claude-sonnet-5-5", "anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("make_embeddings", "text-embedding-3-small", "openai_api_key", "OPENAI_API_KEY"),
    ],
)
def test_missing_keys_are_clear(
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    model: str,
    key_name: str,
    environment_name: str,
) -> None:
    monkeypatch.setattr(ragas_score, "get_settings", lambda: SimpleNamespace(**{key_name: None}))
    with pytest.raises(RuntimeError, match=environment_name):
        getattr(ragas_score, builder_name)(FakeMeter(), model)


def test_make_embeddings_reads_secret_and_wraps_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meter = Meter(
        {"real": (3.0, 10.0), "credit": (3.0, 10.0)}, tmp_path / "ledger.csv", "embeddings"
    )
    response = SimpleNamespace(
        model="text-embedding-3-small", usage=SimpleNamespace(prompt_tokens=123)
    )

    class Embeddings:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            return response

    client = SimpleNamespace(embeddings=Embeddings())
    constructor_keys: list[str] = []
    factory_calls: list[tuple[str, str, Any]] = []
    expected = object()

    def constructor(*, api_key: str) -> Any:
        constructor_keys.append(api_key)
        return client

    def factory(provider: str, model: str, *, client: Any) -> Any:
        factory_calls.append((provider, model, client))
        return expected

    monkeypatch.setattr(
        ragas_score, "get_settings", lambda: SimpleNamespace(openai_api_key=SecretStr("test"))
    )
    monkeypatch.setattr(ragas_score, "AsyncOpenAI", constructor)
    monkeypatch.setattr(ragas_score, "embedding_factory", factory)
    assert ragas_score.make_embeddings(meter, "text-embedding-3-small") is expected
    assert constructor_keys == ["test"]
    assert factory_calls == [("openai", "text-embedding-3-small", client)]
    assert asyncio.run(client.embeddings.create()) is response
    assert meter.run_cost["real"] == pytest.approx(123 / 1e6 * 0.024)


def test_ragas_analytics_are_off() -> None:
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
