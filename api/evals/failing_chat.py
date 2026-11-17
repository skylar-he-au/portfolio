"""A deliberately unavailable Claude for the Gemini answer route."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk, ChatResult


class FailingChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing-chat"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("route gemini-answer: Claude is switched off")

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("route gemini-answer: Claude is switched off")

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        raise RuntimeError("route gemini-answer: Claude is switched off")
        yield  # Make this a generator, like BaseChatModel._stream.

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        raise RuntimeError("route gemini-answer: Claude is switched off")
        yield  # Make this an async generator, like BaseChatModel._astream.

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self
