"""Pricing, ledger caps, callbacks and SDK metering without paid calls."""

import asyncio
import csv
import fcntl
import multiprocessing
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from evals import cost
from evals.cost import LedgerLocked, Meter, meter_anthropic, meter_openai
from evals.pricing import POOLS, Price, cost_usd, price_for
from evals.types import CapReached, UnknownModel

CAPS = {"real": (3.0, 10.0), "credit": (3.0, 10.0)}
HEADER = ("run_id", "question_id", "repeat", "pool", "cost_usd", "at")


def _append_ledger(path: Path, rows: list[tuple[str, float | str]]) -> None:
    empty = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as ledger:
        writer = csv.writer(ledger)
        if empty:
            writer.writerow(HEADER)
        for pool, cost in rows:
            writer.writerow(("older-run", "q", 0, pool, cost, "2026-01-01T00:00:00+00:00"))


def _read_ledger(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as ledger:
        return list(csv.DictReader(ledger))


def _record_worker(ledger_path: str, run_id: str, ready: Any) -> None:
    """Spawn-safe writer sharing one ledger with another process."""
    meter = Meter(CAPS, ledger_path, run_id)
    if not ready.wait(timeout=30):
        raise RuntimeError("Concurrent ledger test did not start")
    for index in range(50):
        meter.add_usage("claude-haiku-5-5", 100, 20)
        meter.record(f"q,{index}", index)


class FakeEndpoint:
    def __init__(self, response: Any, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-haiku-5-5-20261001", Price("credit", 0.10, 0.50)),
        ("models/claude-sonnet-5-5-20261001", Price("credit", 2.00, 10.00)),
        ("models/gemini-3.5-flash-lite-001", Price("real", 0.30, 2.50)),
        ("text-embedding-3-small", Price("real", 0.02, 0.00)),
    ],
)
def test_price_prefixes(model: str, expected: Price) -> None:
    assert price_for(model) == expected
    assert cost_usd(model, 1234, 56) == pytest.approx(
        (1234 * expected.input_per_m + 56 * expected.output_per_m) / 1e6
    )
    assert POOLS == ("real", "credit")


@pytest.mark.parametrize("model", ["unknown", "", "models/", "gpt-unpriced"])
def test_unknown_price(model: str) -> None:
    with pytest.raises(UnknownModel) as caught:
        price_for(model)
    assert caught.value.args == (model,)
    with pytest.raises(UnknownModel):
        cost_usd(model, 1, 1)


@pytest.mark.parametrize("missing", POOLS)
def test_both_pool_caps_are_required(tmp_path: Path, missing: str) -> None:
    caps = {pool: cap for pool, cap in CAPS.items() if pool != missing}
    with pytest.raises(ValueError, match=missing):
        Meter(caps, tmp_path / "ledger.csv", "run")


@pytest.mark.parametrize(
    ("pool", "model", "tokens"),
    [
        ("real", "gemini-3.5-flash-lite", 1_000_000),
        ("credit", "claude-haiku-5-5", 3_000_000),
    ],
)
def test_total_cap_includes_unrecorded_usage(
    tmp_path: Path, pool: str, model: str, tokens: int
) -> None:
    path = tmp_path / "ledger.csv"
    _append_ledger(path, [(pool, 9.7)])
    meter = Meter(CAPS, path, "new-run")
    meter.check()
    meter.add_usage(model, tokens, 0)
    with pytest.raises(CapReached, match=rf"{pool}: total cap") as caught:
        meter.check()
    message = str(caught.value)
    assert "ledger=9.7" in message
    assert "unrecorded=0.3" in message
    assert "cap=10.0" in message
    assert "pass cap" not in message


@pytest.mark.parametrize("pool", POOLS)
def test_exhausted_pool_stops_an_unaffected_pool(tmp_path: Path, pool: str) -> None:
    path = tmp_path / "ledger.csv"
    _append_ledger(path, [(pool, 10.0)])
    meter = Meter(CAPS, path, "run")
    assert meter.run_cost == {"real": 0.0, "credit": 0.0}
    with pytest.raises(CapReached, match=rf"{pool}: total cap"):
        meter.check()


def test_start_pass_resets_pass_cost_but_preserves_ledger_totals(tmp_path: Path) -> None:
    caps = {"real": (3.0, 5.0), "credit": (3.0, 10.0)}
    meter = Meter(caps, tmp_path / "ledger.csv", "run")
    meter.add_usage("gemini-3.5-flash-lite", 10_000_000, 0)
    with pytest.raises(CapReached, match=r"real: pass cap: cost=3\.0, cap=3\.0"):
        meter.check()
    assert meter.record("first", 0) == pytest.approx(3.0)
    meter.start_pass()
    meter.check()
    meter.add_usage("gemini-3.5-flash-lite", 7_000_000, 0)
    with pytest.raises(CapReached, match="real: total cap"):
        meter.check()
    assert meter.run_cost["real"] == pytest.approx(5.1)
    meter.record("second", 0)
    meter.start_pass()
    with pytest.raises(CapReached, match="real: total cap"):
        meter.check()


def test_start_pass_keeps_unrecorded_usage(tmp_path: Path) -> None:
    meter = Meter(CAPS, tmp_path / "ledger.csv", "run")
    meter.add_usage("gemini-3.5-flash-lite", 40_000_000, 0)
    meter.start_pass()
    with pytest.raises(CapReached, match="real: total cap"):
        meter.check()
    assert meter.record("q", 0) == pytest.approx(12.0)


def test_check_rereads_other_runs_and_ignores_unparseable_costs(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    meter = Meter(CAPS, path, "run")
    meter.check()
    _append_ledger(path, [("real", "bad"), ("real", ""), ("other", 999.0), ("real", 1.0)])
    meter.check()
    _append_ledger(path, [("credit", 10.0)])
    with pytest.raises(CapReached, match="credit: total cap"):
        meter.check()


def test_unknown_usage_is_deferred_and_record_writes_before_raising(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "ledger.csv"
    meter = Meter(CAPS, path, "run")
    meter.add_usage("new-model", 100, 200)
    meter.add_usage("", 100, 200)
    meter.add_usage("claude-haiku-5-5", 100, 20)
    with pytest.raises(UnknownModel, match="new-model") as caught:
        meter.check()
    assert "''" in str(caught.value)
    with pytest.raises(UnknownModel, match="new-model"):
        meter.record("q", 1)
    rows = _read_ledger(path)
    assert [row["pool"] for row in rows] == list(POOLS)
    assert [float(row["cost_usd"]) for row in rows] == pytest.approx(
        [0.0, cost_usd("claude-haiku-5-5", 100, 20)]
    )
    assert all(row["question_id"] == "q" and row["repeat"] == "1" for row in rows)
    with pytest.raises(UnknownModel):
        meter.record("next", 2)
    assert [float(row["cost_usd"]) for row in _read_ledger(path)[2:]] == [0.0, 0.0]


def test_unknown_models_take_precedence_over_caps(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    _append_ledger(path, [("real", 10.0)])
    meter = Meter(CAPS, path, "run")
    meter.add_usage("unpriced", 1, 1)
    with pytest.raises(UnknownModel, match="unpriced"):
        meter.check()


def test_embedding_character_estimate_and_unknown_model(tmp_path: Path) -> None:
    meter = Meter(CAPS, tmp_path / "ledger.csv", "run")
    meter.add_embedding_chars("text-embedding-3-small", 8001)
    assert meter.run_cost["real"] == pytest.approx(
        cost_usd("text-embedding-3-small", 8001 / 4 * 2, 0)
    )
    snapshot = meter.run_cost
    snapshot["real"] = 100.0
    assert meter.run_cost["real"] != snapshot["real"]
    meter.add_embedding_chars("unknown-embedding", 10)
    with pytest.raises(UnknownModel, match="unknown-embedding"):
        meter.check()


def test_concurrent_record_has_one_header_and_complete_rows(tmp_path: Path) -> None:
    path = tmp_path / "concurrent" / "ledger.csv"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    processes = [
        context.Process(target=_record_worker, args=(str(path), f"run-{index}", ready))
        for index in range(2)
    ]
    try:
        for process in processes:
            process.start()
        ready.set()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 201
    assert lines.count(",".join(HEADER)) == 1
    rows = _read_ledger(path)
    assert len(rows) == 200
    expected_cost = cost_usd("claude-haiku-5-5", 100, 20)
    for row in rows:
        assert len(row) == len(HEADER)
        assert None not in row and all(value is not None for value in row.values())
        assert datetime.fromisoformat(row["at"]).utcoffset() == timedelta(0)
        expected = expected_cost if row["pool"] == "credit" else 0.0
        assert float(row["cost_usd"]) == pytest.approx(expected)
    for run_id in ("run-0", "run-1"):
        for index in range(50):
            pair = [row for row in rows if row["run_id"] == run_id and row["repeat"] == str(index)]
            assert [row["pool"] for row in pair] == list(POOLS)
            assert all(row["question_id"] == f"q,{index}" for row in pair)


def test_callback_meters_model_name_and_model_version(tmp_path: Path) -> None:
    meter = Meter(CAPS, tmp_path / "ledger.csv", "run")
    handler = meter.callbacks()[0]
    assert handler.run_inline is True
    assert handler.raise_error is False
    result = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="x",
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        },
                        response_metadata={"model_name": "claude-haiku-5-5-20261001"},
                    ),
                ),
            ],
            [
                ChatGeneration(
                    message=AIMessage(
                        content="x",
                        usage_metadata={
                            "input_tokens": 200,
                            "output_tokens": 30,
                            "total_tokens": 230,
                        },
                        response_metadata={"model_version": "gemini-3.5-flash-lite-001"},
                    ),
                ),
                ChatGeneration(message=AIMessage(content="no usage")),
            ],
        ],
    )
    handler.on_llm_end(result)
    assert meter.run_cost == pytest.approx(
        {
            "real": cost_usd("gemini-3.5-flash-lite", 200, 30),
            "credit": cost_usd("claude-haiku-5-5", 100, 20),
        }
    )
    meter.check()


def test_callback_defers_empty_model_names(tmp_path: Path) -> None:
    meter = Meter(CAPS, tmp_path / "ledger.csv", "run")
    message = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    meter.callbacks()[0].on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    with pytest.raises(UnknownModel, match="''"):
        meter.check()


def test_sdk_wrappers_meter_responses_and_record_resets(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    meter = Meter(CAPS, path, "run")
    anthropic_response = SimpleNamespace(
        model="claude-sonnet-5-5", usage=SimpleNamespace(input_tokens=100, output_tokens=20)
    )
    openai_response = SimpleNamespace(
        model="text-embedding-3-small", usage=SimpleNamespace(prompt_tokens=800)
    )
    anthropic_client = SimpleNamespace(messages=FakeEndpoint(anthropic_response))
    openai_client = SimpleNamespace(embeddings=FakeEndpoint(openai_response))
    assert meter_anthropic(anthropic_client, meter) is anthropic_client
    assert meter_openai(openai_client, meter) is openai_client

    async def invoke() -> None:
        assert await anthropic_client.messages.create(model="judge") is anthropic_response
        assert await openai_client.embeddings.create(model="embedding") is openai_response

    asyncio.run(invoke())
    expected = {
        "real": cost_usd("text-embedding-3-small", 800, 0),
        "credit": cost_usd("claude-sonnet-5-5", 100, 20),
    }
    assert meter.run_cost == pytest.approx(expected)
    assert meter.record("q", 0) == pytest.approx(sum(expected.values()))
    assert meter.record("q", 1) == 0.0
    assert meter.run_cost == pytest.approx(expected)
    rows = _read_ledger(path)
    assert [float(row["cost_usd"]) for row in rows] == pytest.approx(
        [expected["real"], expected["credit"], 0.0, 0.0]
    )


@pytest.mark.parametrize("model", [None, ""])
def test_openai_wrapper_uses_requested_model_as_fallback(tmp_path: Path, model: str | None) -> None:
    meter = Meter(CAPS, tmp_path / "ledger.csv", "run")
    response = SimpleNamespace(model=model, usage=SimpleNamespace(prompt_tokens=400))
    client = SimpleNamespace(embeddings=FakeEndpoint(response))
    meter_openai(client, meter)
    assert asyncio.run(client.embeddings.create(model="text-embedding-3-small")) is response
    assert meter.run_cost["real"] == pytest.approx(cost_usd("text-embedding-3-small", 400, 0))


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_callbacks_and_wrappers_swallow_metering_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    meter = Meter(CAPS, tmp_path / "ledger.csv", "run")

    def broken_meter(*args: Any) -> None:
        raise ValueError("meter failure")

    monkeypatch.setattr(meter, "add_usage", broken_meter)
    response = SimpleNamespace(
        model="model",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2, prompt_tokens=3),
    )
    endpoint = FakeEndpoint(response)
    if provider == "anthropic":
        client = SimpleNamespace(messages=endpoint)
        meter_anthropic(client, meter)
    else:
        client = SimpleNamespace(embeddings=endpoint)
        meter_openai(client, meter)
    assert asyncio.run(endpoint.create(model="model")) is response
    endpoint.response = SimpleNamespace()
    assert asyncio.run(endpoint.create()) is endpoint.response
    endpoint.error = ValueError("API failure")
    with pytest.raises(ValueError, match="API failure"):
        asyncio.run(endpoint.create())
    message = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        response_metadata={"model_name": "claude-haiku-5-5"},
    )
    handler = meter.callbacks()[0]
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    handler.on_llm_end(SimpleNamespace(generations=None))


def test_eval_prices_are_the_app_price_table() -> None:
    from app.guard import prices as app_prices

    for model, (input_price, output_price) in app_prices._PRICES.items():
        price = price_for(model)
        assert (price.input_per_m, price.output_per_m) == (float(input_price), float(output_price))
        assert cost_usd(model, 1234, 56) == pytest.approx(
            float(app_prices.cost_usd(model, {"input_tokens": 1234, "output_tokens": 56}))
        )


def test_app_unknown_model_becomes_the_eval_exception() -> None:
    from app.guard import prices as app_prices

    with pytest.raises(UnknownModel) as caught:
        cost_usd("claude-opus-unpriced", 1, 1)
    assert type(caught.value) is UnknownModel
    assert not isinstance(caught.value, app_prices.UnknownModel)


def test_a_priced_model_without_a_pool_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from decimal import Decimal

    from app.guard import prices as app_prices

    monkeypatch.setitem(app_prices._PRICES, "mistral-small", (Decimal("1"), Decimal("1")))
    assert app_prices.cost_usd("mistral-small", {"input_tokens": 1}) > 0
    with pytest.raises(UnknownModel, match="mistral-small"):
        price_for("mistral-small")
    with pytest.raises(UnknownModel, match="mistral-small"):
        cost_usd("mistral-small", 1, 1)


# --- a ledger another process keeps locked (step 7 7t) -----------------------------------------

LOCK_WAIT = 0.3  # `LOCK_TIMEOUT_S` in these tests, seconds
BACKSTOP = 3.0  # only so that a blocking `flock` fails the test instead of hanging it


@contextmanager
def held_lock(path: Path, *, release_after: float | None = None) -> Iterator[None]:
    """Another holder of the ledger's exclusive lock: a second open file, so `flock` treats
    it like another process. `release_after` lets go from a timer thread (the loop itself
    may be blocked by then)."""
    if not path.exists():
        _append_ledger(path, [("real", 0.5)])
    holder = path.open("a", encoding="utf-8")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    released = threading.Event()

    def release() -> None:
        if not released.is_set():
            released.set()
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)

    timer = threading.Timer(release_after, release) if release_after is not None else None
    if timer is not None:
        timer.start()
    try:
        yield
    finally:
        if timer is not None:
            timer.cancel()
        release()
        holder.close()


async def run_with_ticker(work: Callable[[], Awaitable[Any]]) -> tuple[Any, int, float]:
    """Run `work` next to a coroutine that ticks every 10 ms: (result or error, ticks, s)."""
    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    started = time.monotonic()
    try:
        async with asyncio.timeout(BACKSTOP):
            outcome: Any = await work()
    except Exception as error:  # noqa: BLE001 - the test looks at what came out
        outcome = error
    took = time.monotonic() - started
    stop.set()
    await task
    return outcome, ticks, took


@pytest.mark.parametrize("call", ["check", "record"])
def test_a_held_ledger_lock_ends_check_and_record_in_time_while_the_loop_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    """Another process holding the ledger: `check` and `record` give up after
    `LOCK_TIMEOUT_S` with a readable error, and (run off the loop, as the runners do) the
    other coroutines keep going. Before the fix the blocking `flock` never returned."""
    monkeypatch.setattr(cost, "LOCK_TIMEOUT_S", LOCK_WAIT)
    path = tmp_path / "ledger.csv"
    meter = Meter(CAPS, path, "run-7t")
    meter.add_usage("claude-haiku-5-5", 1_000_000, 0)  # $0.10 credit, not recorded yet
    with held_lock(path, release_after=BACKSTOP + 1):
        method = meter.check if call == "check" else lambda: meter.record("q", 0)
        outcome, ticks, took = asyncio.run(run_with_ticker(lambda: asyncio.to_thread(method)))
    assert isinstance(outcome, LedgerLocked), repr(outcome)
    assert isinstance(outcome, CapReached)  # every runner stops on it like on a cap
    assert str(outcome) == f"ledger {path} stayed locked by another process for 0.3 s"
    assert LOCK_WAIT <= took < LOCK_WAIT + 0.5
    assert ticks >= 10  # the loop was free the whole time
    assert [row["run_id"] for row in _read_ledger(path)] == ["older-run"]
    # nothing lost: the unrecorded cost is written once the ledger is free again
    assert meter.record("q", 0) == pytest.approx(0.10)
    assert [r["cost_usd"] for r in _read_ledger(path) if r["run_id"] == "run-7t"] == [
        "0.0",
        "0.1",
    ]


def test_a_lock_let_go_within_the_limit_is_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cost, "LOCK_TIMEOUT_S", 2.0)
    path = tmp_path / "ledger.csv"
    meter = Meter(CAPS, path, "run-7t")
    with held_lock(path, release_after=0.2):
        started = time.monotonic()
        meter.check()
        meter.record("q", 0)
        took = time.monotonic() - started
    assert 0.15 <= took < 1.5
    assert [row["run_id"] for row in _read_ledger(path)] == ["older-run", "run-7t", "run-7t"]


def test_readers_share_the_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A shared lock (another run's `check`) does not hold up this run's `check`."""
    monkeypatch.setattr(cost, "LOCK_TIMEOUT_S", LOCK_WAIT)
    path = tmp_path / "ledger.csv"
    _append_ledger(path, [("real", 0.5)])
    with path.open(encoding="utf-8") as reader:
        fcntl.flock(reader.fileno(), fcntl.LOCK_SH)
        started = time.monotonic()
        Meter(CAPS, path, "run-7t").check()
        assert time.monotonic() - started < 0.1


def test_waiting_for_the_ledger_does_not_hold_up_usage_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The usage callbacks run on the event loop's thread and take the meter's own lock: a
    `check` waiting for the ledger in a worker thread must not hold that lock meanwhile."""
    monkeypatch.setattr(cost, "LOCK_TIMEOUT_S", 1.0)
    path = tmp_path / "ledger.csv"
    meter = Meter(CAPS, path, "run-7t")
    with held_lock(path, release_after=BACKSTOP + 1):
        waiting = threading.Thread(target=lambda: suppress_locked(meter.check))
        waiting.start()
        time.sleep(0.1)
        started = time.monotonic()
        meter.add_usage("claude-haiku-5-5", 100, 20)
        took = time.monotonic() - started
        waiting.join()
    assert took < 0.05


def suppress_locked(call: Callable[[], Any]) -> None:
    try:
        call()
    except LedgerLocked:
        pass


@pytest.mark.parametrize("runner", ["qa", "match"])
def test_the_eval_runners_stop_on_a_held_ledger_and_keep_the_loop_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: str
) -> None:
    """`evals/run.py` and `evals/match_run.py` read the ledger off the loop: the run ends
    `incomplete` with a `LedgerLocked` reason in time, and the loop keeps running. Before
    the fix the runner's `check` blocked the loop until the holder let go."""
    monkeypatch.setattr(cost, "LOCK_TIMEOUT_S", LOCK_WAIT)
    path = tmp_path / "ledger.csv"
    with held_lock(path, release_after=BACKSTOP + 1):
        if runner == "qa":
            from test_evals_run import run_eval

            work = lambda: run_eval(tmp_path, meter=Meter(CAPS, path, "run-7t"))  # noqa: E731
        else:
            from test_evals_match import EvalScript, _evaluate, eval_deps

            from evals.match_run import MatchMeter

            meter = MatchMeter(CAPS, path, "run-7t")
            work = lambda: _evaluate(tmp_path, eval_deps(EvalScript()), meter=meter)  # noqa: E731
        outcome, ticks, took = asyncio.run(run_with_ticker(work))
    assert not isinstance(outcome, Exception), repr(outcome)
    assert outcome.status == "incomplete" and outcome.rows == []
    assert outcome.reason.startswith("LedgerLocked")
    assert took < LOCK_WAIT + 1.0
    assert ticks >= 10
