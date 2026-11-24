"""The Azure OpenAI embedder: which credential it sends, what it never reads from the
environment, and how a 429 that outlasts the SDK's retries reaches the visitor."""

import asyncio
import json
from typing import Any
from uuid import uuid4

import httpx
import openai
import pytest
from fakes import FakeEmbedder, unit_vector
from langchain_openai import AzureOpenAIEmbeddings
from test_ask import (
    MemoryCheckpoints,
    MemoryConversations,
    Models,
    collect,
    general_turn,
    make_deps,
)
from test_graph import EXAMPLE_Q, MATCH, FakeRetriever
from test_guard_ledger import limits, rows

from app.api.ask import AskService
from app.config import AZURE_OPENAI_API_VERSION, Settings
from app.db import Database, make_pool
from app.guard.ledger import Ledger, current_scope
from app.migrate import migrate
from app.qa.models import MAX_OUTPUT_TOKENS
from app.rag.embed import AzureEmbedder, estimate_tokens

ENDPOINT = "https://unit-test.openai.azure.com/"
DEPLOYMENT = "unit-deployment"
# What Azure sends back when a deployment's rate limit is exceeded.
AZURE_429 = {
    "error": {
        "code": "429",
        "message": "Requests to the Embeddings_Create Operation under Azure OpenAI API "
        "version 2024-10-21 have exceeded call rate limit of your current OpenAI S0 pricing "
        "tier. Please retry after 1 second.",
    }
}


@pytest.fixture(autouse=True)
def no_credentials_in_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """`AzureOpenAIEmbeddings` fills unset fields from these; none may be used."""
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_API_VERSION",
        "OPENAI_ORG_ID",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_AD_TOKEN",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)


def azure_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": "postgresql://unused",
        "azure_openai_endpoint": ENDPOINT,
        "azure_openai_deployment": DEPLOYMENT,
    }
    return Settings(_env_file=None, **(values | overrides))


def sdk_client(embedder: AzureEmbedder) -> openai.AsyncAzureOpenAI:
    return embedder._client.async_client._client


def test_key_only_sends_the_key() -> None:
    embedder = AzureEmbedder(azure_settings(azure_openai_api_key="unit-key"))
    client = sdk_client(embedder)
    assert client.api_key == "unit-key"
    assert client._azure_ad_token is None
    assert embedder._client.azure_ad_token is None


def test_token_only_sends_the_token() -> None:
    embedder = AzureEmbedder(azure_settings(azure_openai_ad_token="unit-token"))
    client = sdk_client(embedder)
    assert client._azure_ad_token == "unit-token"
    assert embedder._client.openai_api_key is None
    assert client.api_key == "<missing API key>"


def test_key_wins_when_both_are_set() -> None:
    embedder = AzureEmbedder(
        azure_settings(azure_openai_api_key="unit-key", azure_openai_ad_token="unit-token")
    )
    client = sdk_client(embedder)
    assert client.api_key == "unit-key"
    assert client._azure_ad_token is None
    assert embedder._client.azure_ad_token is None


@pytest.mark.parametrize("blank", [None, ""])
def test_no_credential_fails_before_building_the_client(
    blank: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_client(**_kwargs: object) -> None:
        pytest.fail("client must not be created without a credential")

    monkeypatch.setattr("app.rag.embed.AzureOpenAIEmbeddings", unexpected_client)
    # Values from the environment must not stand in for the missing ones either.
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-azure")
    settings = Settings(
        _env_file=None,
        database_url="postgresql://unused",
        azure_openai_endpoint=ENDPOINT,
        azure_openai_deployment=DEPLOYMENT,
        azure_openai_api_key=blank,
        azure_openai_ad_token=blank,
    )
    with pytest.raises(RuntimeError) as caught:
        AzureEmbedder(settings)
    assert str(caught.value) == (
        "Missing embedding settings: AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN"
    )


def test_missing_endpoint_and_deployment_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        _env_file=None, database_url="postgresql://unused", azure_openai_api_key="k"
    )
    # Set after the settings are read: the library must not pick these up either.
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")
    with pytest.raises(RuntimeError) as caught:
        AzureEmbedder(settings)
    assert str(caught.value) == (
        "Missing embedding settings: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT"
    )


def test_blank_values_from_an_env_file_count_as_unset() -> None:
    """`.env.example` ships `AZURE_OPENAI_*=` with nothing after the `=`."""
    settings = Settings(
        _env_file=None,
        database_url="postgresql://unused",
        azure_openai_endpoint="",
        azure_openai_deployment="",
        azure_openai_api_key="",
        azure_openai_ad_token="",
    )
    assert settings.azure_openai_endpoint is None
    assert settings.azure_openai_deployment is None
    assert settings.azure_openai_api_key is None
    assert settings.azure_openai_ad_token is None


@pytest.mark.parametrize("configured", ["key", "token"])
def test_environment_credentials_never_fill_in(
    configured: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Variables the library reads on its own, set after our settings were read, stay
    unused: no key next to a token (it would be dropped anyway), no token next to a key
    (it would silently replace the key), no other base URL."""
    if configured == "key":
        settings = azure_settings(azure_openai_api_key="unit-key")
    else:
        settings = azure_settings(azure_openai_ad_token="unit-token")
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-azure")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "env-token")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    embedder = AzureEmbedder(settings)
    client = sdk_client(embedder)
    if configured == "key":
        assert client.api_key == "unit-key"
        assert embedder._client.azure_ad_token is None
        assert client._azure_ad_token is None
    else:
        assert embedder._client.openai_api_key is None
        assert client.api_key == "<missing API key>"
        assert client._azure_ad_token == "unit-token"
    assert str(client.base_url) == f"{ENDPOINT}openai/deployments/{DEPLOYMENT}/"


def test_client_options(monkeypatch: pytest.MonkeyPatch) -> None:
    options: dict[str, object] = {}
    calls: list[list[str]] = []
    vectors = [unit_vector(), unit_vector(1)]

    class StubEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            options.update(kwargs)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return vectors

    monkeypatch.setattr("app.rag.embed.AzureOpenAIEmbeddings", StubEmbeddings)
    settings = azure_settings(
        azure_openai_api_key="unit-key", embed_model="configured-model", embed_dims=1536
    )
    embedder = AzureEmbedder(settings)
    assert asyncio.run(embedder.embed(["one", "two"])) == vectors
    assert calls == [["one", "two"]]
    assert options == {
        "model": "configured-model",
        "dimensions": 1536,
        "azure_endpoint": ENDPOINT,
        "azure_deployment": DEPLOYMENT,
        "api_version": AZURE_OPENAI_API_VERSION,
        "api_key": settings.azure_openai_api_key,
        "azure_ad_token": None,
        "base_url": None,
        "organization": None,
        "timeout": 30,
        "max_retries": 2,
        "check_embedding_ctx_length": False,
    }
    assert AZURE_OPENAI_API_VERSION == "2024-10-21"


def test_sends_text_without_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real client sends raw strings and never loads a tiktoken encoding."""
    sent: list[object] = []

    class StubResponse:
        def model_dump(self) -> dict[str, object]:
            return {"data": [{"embedding": unit_vector()}]}

    class StubAsyncClient:
        async def create(self, *, input: object, **_kwargs: object) -> StubResponse:
            sent.append(input)
            return StubResponse()

    def no_tiktoken(*_args: object, **_kwargs: object) -> None:
        pytest.fail("tiktoken must not be loaded")

    monkeypatch.setattr("tiktoken.encoding_for_model", no_tiktoken)
    monkeypatch.setattr("tiktoken.get_encoding", no_tiktoken)
    embedder = AzureEmbedder(azure_settings(azure_openai_api_key="unit-key"))
    assert embedder._client.check_embedding_ctx_length is False
    embedder._client.async_client = StubAsyncClient()
    assert asyncio.run(embedder.embed(["What did Skylar build?"])) == [unit_vector()]
    assert sent == [["What did Skylar build?"]]


def test_estimate_tokens_rounds_up_by_thirds() -> None:
    assert estimate_tokens([]) == 0
    assert estimate_tokens(["a"]) == 1
    assert estimate_tokens(["abc", "abcd"]) == 3


# --- a 429 that outlasts the SDK's retries ---------------------------------------------


def always_429(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Build the real SDK client over a transport that always answers like Azure's 429."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, headers={"retry-after-ms": "1"}, json=AZURE_429)

    def with_transport(**kwargs: Any) -> AzureOpenAIEmbeddings:
        transport = httpx.MockTransport(handler)
        return AzureOpenAIEmbeddings(
            **kwargs, http_async_client=httpx.AsyncClient(transport=transport)
        )

    monkeypatch.setattr("app.rag.embed.AzureOpenAIEmbeddings", with_transport)
    return requests


def embedding_retriever(embedder: AzureEmbedder) -> FakeRetriever:
    """Embeds the query first, like `retrieve`, then would return sections."""

    class Retriever(FakeRetriever):
        async def __call__(self, query: str, *, pages: Any = None) -> list:
            await embedder.embed([query])
            return await super().__call__(query, pages=pages)

    return Retriever([MATCH])


@pytest.mark.anyio
async def test_azure_429_after_retries_is_unavailable_to_the_visitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = always_429(monkeypatch)
    embedder = AzureEmbedder(azure_settings(azure_openai_api_key="unit-key"))
    claude, gemini = general_turn()
    deps = make_deps(Models(claude, gemini), embedding_retriever(embedder), embedder=FakeEmbedder())
    service = AskService(MemoryConversations(), MemoryCheckpoints(), lambda: deps, timeout_s=10.0)

    events = await collect(service, EXAMPLE_Q)

    assert events[-1] == (
        "error",
        {
            "code": "unavailable",
            "message": "The assistant is unavailable right now — please try again later "
            "or email me.",
        },
    )
    # The first try and the SDK's two retries, all to the configured deployment.
    assert len(requests) == 3
    assert {request.url.path for request in requests} == {
        f"/openai/deployments/{DEPLOYMENT}/embeddings"
    }
    assert {request.url.params["api-version"] for request in requests} == {"2024-10-21"}
    assert {request.headers["api-key"] for request in requests} == {"unit-key"}
    assert len(json.loads(requests[0].content)["input"]) == 1  # the rewritten query


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_azure_429_inside_a_budget_scope_never_pauses_the_ai(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rate limits are not budget errors: no `ai_pause` row, so no "AI resting" month."""
    migrate(db_url)
    requests = always_429(monkeypatch)
    embedder = AzureEmbedder(azure_settings(azure_openai_api_key="unit-key"))
    db = Database(make_pool(db_url))
    try:
        ledger = Ledger(db, limits(), MAX_OUTPUT_TOKENS)
        scope = await ledger.reserve_question(uuid4())
        token = current_scope.set(scope)
        try:
            with pytest.raises(openai.RateLimitError):
                await embedder.embed([EXAMPLE_Q])
        finally:
            current_scope.reset(token)
            await scope.settle()
        assert len(requests) == 3
        assert await rows(ledger, "SELECT * FROM ai_pause") == []
    finally:
        await db.close()
