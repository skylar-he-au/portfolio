"""Deterministic, network-free fakes: embeddings (ingest and retrieval tests) and a
structured-output chat model (job-match graph tests)."""

import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

DIMENSIONS = 1536


def unit_vector(index: int = 0) -> list[float]:
    vector = [0.0] * DIMENSIONS
    vector[index] = 1.0
    return vector


class FakeEmbedder:
    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.vectors = vectors or {}
        self.fail_on_call = fail_on_call
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("fake embedding failure")
        result = []
        for text in texts:
            if text in self.vectors:
                result.append(list(self.vectors[text]))
            else:
                digest = hashlib.sha256(text.encode()).digest()
                result.append(
                    [(digest[index % len(digest)] - 127.5) / 128 for index in range(DIMENSIONS)]
                )
        return result


class FakeJsonChat(BaseChatModel):
    """Chat model for structured output. Each call asks `reply(messages)` for its reply:
    a dict (sent as JSON), a str, or an Exception (raised); `reply` may be async, so a test
    can make calls wait. Records every call and the `with_structured_output` keyword
    arguments (the match graph must pass `method="json_schema"`)."""

    reply: Callable[[list[BaseMessage]], Any]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    structured_kwargs: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-json-chat"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("async only")

    async def _agenerate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        self.calls.append(list(messages))
        result = self.reply(list(messages))
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Exception):
            raise result
        text = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        self.structured_kwargs.append(dict(kwargs))
        return self | RunnableLambda(lambda message: schema.model_validate_json(message.text))
