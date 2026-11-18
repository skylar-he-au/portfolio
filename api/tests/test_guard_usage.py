import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import anthropic
import httpx
import openai
import pytest
from google.genai.errors import ClientError
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel, SecretStr
from test_guard_ledger import MODELS, limits, rows

from app.db import Database, make_pool
from app.guard.ledger import (
    PROVIDERS,
    Ledger,
    ProviderBudgetExhausted,
    SpendLimits,
    SpendScope,
    current_scope,
    next_month_start,
    sydney_day,
)
from app.guard.prices import UnknownModel, cost_usd
from app.guard.usage import (
    UsageRecorder,
    estimate_input_tokens,
    is_budget_exhausted,
    metered,
    raise_if_budget_exhausted,
)
from app.migrate import migrate
from app.qa.models import MAX_OUTPUT_TOKENS, ChatProvider, stream_text, structured
from app.rag.embed import OpenAIEmbedder
from app.rag.types import ModelsUnavailable

pytestmark = pytest.mark.anyio

BUDGET_CASES = [
    ("anthropic", 400, "You have reached your specified API usage limits"),
    ("anthropic", 400, "Your credit balance is too low"),
    ("anthropic", 429, "enforced_spend_limit_reached"),
    ("openai", 429, "credit_balance_exhausted"),
    ("openai", 429, "organization_spend_limit_exceeded"),
    ("openai", 429, "project_spend_limit_exceeded"),
    ("openai", 429, "organization_usage_limit_exceeded"),
    ("google", 402, "payment required"),
    ("google", 429, "prepayment credits are depleted"),
]
USAGE = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def migrated_url(db_url: str) -> str:
    migrate(db_url)
    return db_url


@pytest.fixture
async def ledger(migrated_url: str) -> AsyncIterator[Ledger]:
    db = Database(make_pool(migrated_url))
    try:
        yield Ledger(db, limits(), MAX_OUTPUT_TOKENS)
    finally:
        await db.close()


def sdk_error(provider: str, status: int, message: str) -> Exception:
    response = httpx.Response(status, request=httpx.Request("POST", "https://x"))
    body = {"error": {"message": message, "code": message}}
    if provider == "anthropic":
        cls = anthropic.BadRequestError if status == 400 else anthropic.RateLimitError
        return cls(message, response=response, body=body)
    if provider == "openai":
        return openai.RateLimitError(message, response=response, body=body)
    return ClientError(status, {"error": {"code": status, "message": message}})


class Reply(BaseModel):
    answer: str


class FakeChat:
    def __init__(self, error: Exception | None = None, usage=USAGE) -> None:
        self.error = error
        self.usage = usage
        self.calls = 0
        self.configs: list[Any] = []
        self.tags: list[str] = []

    def with_structured_output(self, schema):
        return self

    def with_config(self, *, tags):
        self.tags.extend(tags)
        return self

    def _call(self, config) -> None:
        self.calls += 1
        self.configs.append(config)
        if self.error is not None:
            raise self.error
        if config:
            response = result(self.usage)
            for callback in config["callbacks"]:
                callback.on_llm_end(response)

    async def ainvoke(self, messages, config=None):
        self._call(config)
        return {"answer": "ok"}

    async def astream(self, messages, config=None):
        self._call(config)
        yield AIMessageChunk(content="ok")


async def invoke(method: str, providers, messages=None, **kwargs):
    messages = messages if messages is not None else [HumanMessage(content="hello")]
    if method == "structured":
        return await structured("test", providers, Reply, messages, deadline_s=1)
    return await stream_text("test", providers, messages, deadline_s=1, **kwargs)


def provider_pair(first: str, error: Exception | None = None, usage=USAGE):
    second = "google" if first != "google" else "anthropic"
    bad, good = FakeChat(error, usage), FakeChat(usage=usage)
    return (
        [ChatProvider(first, MODELS[first], bad), ChatProvider(second, MODELS[second], good)],
        bad,
        good,
    )


def memory_scope() -> SpendScope:
    ledger = Ledger(None, limits(), MAX_OUTPUT_TOKENS)
    return SpendScope(
        ledger,
        uuid4(),
        sydney_day(datetime.now(UTC)),
        dict.fromkeys(PROVIDERS, Decimal("10")),
    )


def result(usage) -> LLMResult:
    message = AIMessage(content="ok", usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_estimate_counts_text_blocks_and_rounds_up() -> None:
    messages = [
        HumanMessage(content="abcd"),
        HumanMessage(
            content=[
                "ef",
                {"type": "text", "text": "ghij"},
                {"type": "image_url", "image_url": "ignored"},
            ],
        ),
    ]
    assert estimate_input_tokens(messages) == 4
    assert estimate_input_tokens([]) == 0


def test_recorder_sums_generations_retries_and_cache_details() -> None:
    recorder = UsageRecorder()
    assert recorder.usage is None
    usage = {
        **USAGE,
        "input_token_details": {"cache_creation": 20, "cache_read": 30},
        "output_token_details": {"reasoning": 10},
    }
    generation = result(usage).generations[0][0]
    recorder.on_llm_end(LLMResult(generations=[[generation, generation], [generation]]))
    recorder.on_llm_end(result(usage))
    assert recorder.usage == {
        "input_tokens": 400,
        "output_tokens": 200,
        "input_token_details": {"cache_creation": 80, "cache_read": 120},
        "output_token_details": {"reasoning": 40},
    }


def test_recorder_never_raises_and_logs_only_exception_types(caplog) -> None:
    recorder = UsageRecorder()
    malformed = SimpleNamespace(message=SimpleNamespace(usage_metadata="SECRET"))
    valid = result(USAGE).generations[0][0]
    with caplog.at_level(logging.WARNING):
        recorder.on_llm_end(SimpleNamespace(generations=[[malformed, valid]]))
        recorder.on_llm_end(object())
    assert recorder.usage["input_tokens"] == 100
    assert "AttributeError" in caplog.text
    assert "SECRET" not in caplog.text


async def test_metered_without_scope_does_not_price_or_pass_callbacks() -> None:
    assert current_scope.get() is None
    async with metered("anthropic", "unknown", input_tokens=100) as meter:
        assert meter.callbacks == []


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_metered_finishes_synchronously_on_every_exit(outcome: str) -> None:
    scope = memory_scope()
    token = current_scope.set(scope)
    try:

        async def call() -> None:
            async with metered(
                "anthropic",
                MODELS["anthropic"],
                input_tokens=1000,
                max_output_tokens=1000,
            ) as meter:
                meter.callbacks[0].on_llm_end(result(USAGE))
                if outcome == "error":
                    raise RuntimeError("failed")
                if outcome == "cancel":
                    asyncio.current_task().cancel()
                    await asyncio.sleep(0)

        task = asyncio.create_task(call())
        if outcome == "success":
            await task
        else:
            with pytest.raises(RuntimeError if outcome == "error" else asyncio.CancelledError):
                await task
    finally:
        current_scope.reset(token)
    assert scope.pending["anthropic"] == 0
    assert scope.used["anthropic"] == cost_usd(MODELS["anthropic"], USAGE)


async def test_metered_unknown_model_leaves_nothing_pending() -> None:
    scope = memory_scope()
    token = current_scope.set(scope)
    try:
        with pytest.raises(UnknownModel):
            async with metered("anthropic", "unknown", input_tokens=1000):
                pytest.fail("unknown model entered")
    finally:
        current_scope.reset(token)
    assert scope.pending == dict.fromkeys(PROVIDERS, Decimal(0))
    assert scope.called == set()


@pytest.mark.parametrize("method", ["structured", "stream"])
async def test_missing_callback_usage_is_settled_at_worst(ledger: Ledger, method: str) -> None:
    scope = await ledger.reserve_question(uuid4())
    token = current_scope.set(scope)
    providers, first, second = provider_pair("anthropic", usage=None)
    try:
        await invoke(method, providers)
    finally:
        current_scope.reset(token)
        await scope.settle()
    worst = cost_usd(
        MODELS["anthropic"],
        {"input_tokens": 2, "output_tokens": MAX_OUTPUT_TOKENS},
    )
    assert first.calls == 1 and second.calls == 0
    assert scope.used["anthropic"] == worst
    # numeric(12, 6) is the schema's storage precision.
    spent = await rows(ledger, "SELECT spent_usd FROM spend_daily")
    assert spent == [(worst.quantize(Decimal("0.000001")),)]


@pytest.mark.parametrize("method", ["structured", "stream"])
async def test_model_callbacks_charge_actual_usage(method: str) -> None:
    scope = memory_scope()
    token = current_scope.set(scope)
    providers, first, second = provider_pair("anthropic")
    try:
        await invoke(method, providers)
    finally:
        current_scope.reset(token)
    assert first.configs[0]["callbacks"]
    assert second.calls == 0
    assert scope.used["anthropic"] == cost_usd(MODELS["anthropic"], USAGE)
    assert scope.pending["anthropic"] == 0


@pytest.mark.parametrize("method", ["structured", "stream"])
async def test_no_scope_supports_fakes_without_config_parameter(method: str) -> None:
    class LegacyChat:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            return {"answer": "unchanged"}

        async def astream(self, messages):
            yield AIMessageChunk(content="unchanged")

    providers = [ChatProvider("anthropic", MODELS["anthropic"], LegacyChat())]
    value, used = await invoke(method, providers)
    assert used is providers[0]
    assert value == (Reply(answer="unchanged") if method == "structured" else "unchanged")


async def test_stream_preserves_hidden_tags_with_callbacks() -> None:
    scope = memory_scope()
    token = current_scope.set(scope)
    providers, first, _ = provider_pair("anthropic")
    try:
        await invoke("stream", providers, visible=False)
    finally:
        current_scope.reset(token)
    assert first.tags == ["nostream"]
    assert first.configs[0]["callbacks"]


@pytest.mark.parametrize("case", BUDGET_CASES)
def test_budget_detection_with_real_sdk_errors(case) -> None:
    provider, status, message = case
    error = sdk_error(provider, status, message)
    assert is_budget_exhausted(provider, error)
    wrapper = RuntimeError("outer")
    wrapper.__cause__ = error
    assert is_budget_exhausted(provider, wrapper)
    wrapper.__cause__ = None
    wrapper.__context__ = error
    assert is_budget_exhausted(provider, wrapper)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_ordinary_rate_limit_is_not_budget_exhaustion(provider: str) -> None:
    assert not is_budget_exhausted(provider, sdk_error(provider, 429, "rate limit"))


def test_budget_detection_uses_body_response_status_and_bounded_chains() -> None:
    class ResponseError(Exception):
        response = httpx.Response(
            429,
            json={"code": "enforced_spend_limit_reached"},
            request=httpx.Request("POST", "https://x"),
        )

    error = ResponseError("SDK wrapper")
    assert is_budget_exhausted("anthropic", error)
    error.__context__ = error
    assert not is_budget_exhausted("google", error)
    wrapper = error
    for _ in range(5):
        outer = RuntimeError("wrapped")
        outer.__cause__ = wrapper
        wrapper = outer
    assert not is_budget_exhausted("anthropic", wrapper)
    code_error = RuntimeError("payment")
    code_error.code = 402
    assert is_budget_exhausted("google", code_error)


@pytest.mark.parametrize("case", BUDGET_CASES)
@pytest.mark.parametrize("method", ["structured", "stream"])
async def test_budget_error_pauses_and_stops_fallback(ledger: Ledger, case, method: str) -> None:
    provider, status, message = case
    scope = await ledger.reserve_question(uuid4())
    token = current_scope.set(scope)
    error = sdk_error(provider, status, message)
    providers, first, second = provider_pair(provider, error)
    before = datetime.now(UTC)
    try:
        with pytest.raises(ProviderBudgetExhausted) as caught:
            await invoke(method, providers)
    finally:
        current_scope.reset(token)
        await scope.settle()
    assert caught.value.provider == provider
    assert caught.value.__cause__ is error
    assert first.calls == 1
    assert second.calls == 0
    pauses = await rows(ledger, "SELECT provider, until FROM ai_pause")
    assert pauses == [(provider, next_month_start(before))]


@pytest.mark.parametrize("case", BUDGET_CASES)
@pytest.mark.parametrize("method", ["structured", "stream"])
async def test_budget_error_without_scope_uses_existing_fallback(case, method: str) -> None:
    provider, status, message = case
    providers, first, second = provider_pair(provider, sdk_error(provider, status, message))
    _, chosen = await invoke(method, providers)
    assert chosen is providers[1]
    assert first.calls == second.calls == 1
    assert first.configs == second.configs == [None]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("with_scope", [False, True])
@pytest.mark.parametrize("method", ["structured", "stream"])
async def test_ordinary_429_falls_back_without_pause(
    ledger: Ledger, provider: str, with_scope: bool, method: str
) -> None:
    scope = await ledger.reserve_question(uuid4()) if with_scope else None
    token = current_scope.set(scope)
    providers, first, second = provider_pair(provider, sdk_error(provider, 429, "rate limit"))
    try:
        _, chosen = await invoke(method, providers)
    finally:
        current_scope.reset(token)
        if scope is not None:
            await scope.settle()
    assert chosen is providers[1]
    assert first.calls == second.calls == 1
    assert await rows(ledger, "SELECT * FROM ai_pause") == []


async def test_pause_write_failure_does_not_hide_budget_error(caplog) -> None:
    scope = memory_scope()
    scope.ledger.pause_provider = AsyncMock(side_effect=RuntimeError("SECRET database url"))
    token = current_scope.set(scope)
    error = sdk_error("anthropic", 400, "usage limits")
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(ProviderBudgetExhausted) as caught:
            await raise_if_budget_exhausted("anthropic", error)
    finally:
        current_scope.reset(token)
    assert caught.value.__cause__ is error
    assert "RuntimeError" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.parametrize("deny_second", [False, True])
async def test_stream_topup_failure_switches(ledger: Ledger, deny_second: bool) -> None:
    scope = await ledger.reserve_question(uuid4())
    daily = dict(ledger.limits.daily)
    daily["anthropic"] = scope.reserved["anthropic"]
    if deny_second:
        daily["google"] = scope.reserved["google"]
    ledger.limits = SpendLimits(daily, ledger.limits.monthly, MODELS)
    token = current_scope.set(scope)
    providers, first, second = provider_pair("anthropic")
    switches = []
    try:
        if deny_second:
            with pytest.raises(ModelsUnavailable):
                await invoke(
                    "stream",
                    providers,
                    [HumanMessage(content="x" * 360000)],
                    on_switch=lambda: switches.append(True),
                )
        else:
            _, chosen = await invoke(
                "stream",
                providers,
                [HumanMessage(content="x" * 360000)],
                on_switch=lambda: switches.append(True),
            )
            assert chosen is providers[1]
    finally:
        current_scope.reset(token)
        await scope.settle()
    assert switches == [True]
    assert first.calls == 0
    assert second.calls == (0 if deny_second else 1)
    assert scope.pending == dict.fromkeys(PROVIDERS, Decimal(0))


@pytest.mark.parametrize("method", ["structured", "stream"])
async def test_unknown_model_with_scope_switches_provider(method: str) -> None:
    scope = memory_scope()
    providers, first, second = provider_pair("anthropic")
    providers[0] = ChatProvider("anthropic", "unknown", first)
    token = current_scope.set(scope)
    try:
        _, chosen = await invoke(method, providers)
    finally:
        current_scope.reset(token)
    assert chosen is providers[1]
    assert first.calls == 0 and second.calls == 1


def embedder(monkeypatch, error: Exception | None = None):
    client = SimpleNamespace(aembed_documents=AsyncMock(return_value=[[0.5]], side_effect=error))
    monkeypatch.setattr("app.rag.embed.OpenAIEmbeddings", lambda **kwargs: client)
    settings = SimpleNamespace(
        openai_api_key=SecretStr("offline"),
        embed_model=MODELS["openai"],
        embed_dims=1,
    )
    return OpenAIEmbedder(settings), client


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_embedding_estimate_is_charged_on_every_exit(monkeypatch, outcome: str) -> None:
    error = {
        "success": None,
        "error": RuntimeError("offline"),
        "cancel": asyncio.CancelledError(),
    }[outcome]
    adapter, client = embedder(monkeypatch, error)
    scope = memory_scope()
    token = current_scope.set(scope)
    try:
        if outcome == "success":
            assert await adapter.embed(["abcd", "efg"]) == [[0.5]]
        else:
            with pytest.raises(RuntimeError if outcome == "error" else asyncio.CancelledError):
                await adapter.embed(["abcd", "efg"])
    finally:
        current_scope.reset(token)
    client.aembed_documents.assert_awaited_once_with(["abcd", "efg"])
    assert scope.used["openai"] == cost_usd(MODELS["openai"], {"input_tokens": 3})
    assert scope.pending["openai"] == 0


@pytest.mark.parametrize("code", [case[2] for case in BUDGET_CASES if case[0] == "openai"])
async def test_openai_embed_budget_error_pauses(ledger: Ledger, monkeypatch, code: str) -> None:
    error = sdk_error("openai", 429, code)
    adapter, _ = embedder(monkeypatch, error)
    scope = await ledger.reserve_question(uuid4())
    token = current_scope.set(scope)
    before = datetime.now(UTC)
    try:
        with pytest.raises(ProviderBudgetExhausted) as caught:
            await adapter.embed(["question"])
    finally:
        current_scope.reset(token)
        await scope.settle()
    assert caught.value.provider == "openai"
    assert caught.value.__cause__ is error
    pauses = await rows(ledger, "SELECT provider, until FROM ai_pause")
    assert pauses == [("openai", next_month_start(before))]


@pytest.mark.parametrize("code", [case[2] for case in BUDGET_CASES if case[0] == "openai"])
async def test_embed_without_scope_reraises_original_budget_error(monkeypatch, code: str) -> None:
    error = sdk_error("openai", 429, code)
    adapter, _ = embedder(monkeypatch, error)
    with pytest.raises(openai.RateLimitError) as caught:
        await adapter.embed(["question"])
    assert caught.value is error


async def test_embed_without_scope_is_unchanged(monkeypatch) -> None:
    adapter, client = embedder(monkeypatch)
    assert await adapter.embed(["question"]) == [[0.5]]
    client.aembed_documents.assert_awaited_once_with(["question"])


async def test_embed_ordinary_rate_limit_does_not_pause(ledger: Ledger, monkeypatch) -> None:
    error = sdk_error("openai", 429, "rate limit")
    adapter, _ = embedder(monkeypatch, error)
    scope = await ledger.reserve_question(uuid4())
    token = current_scope.set(scope)
    try:
        with pytest.raises(openai.RateLimitError) as caught:
            await adapter.embed(["question"])
    finally:
        current_scope.reset(token)
        await scope.settle()
    assert caught.value is error
    assert await rows(ledger, "SELECT * FROM ai_pause") == []
