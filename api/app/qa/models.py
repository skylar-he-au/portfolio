"""Chat models for the Q&A graph: building them, timeouts and the fallback between them.

Every call gets one overall deadline (`deadline_s`, from `GraphDeps.call_timeout`: 30 s in
production) that covers the SDK's own retries too. Any exception, the deadline or an
empty/invalid reply counts as the provider failing; the next provider in the order then
gets the same messages. If every provider fails the node raises `ModelsUnavailable` and
the API sends `error unavailable`.
Failures are logged by exception type only: no prompts, no answers, no keys.
"""

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from app.config import Settings
from app.qa.output import message_text
from app.rag.types import ModelsUnavailable

logger = logging.getLogger(__name__)

CALL_TIMEOUT_S = 30.0
# Explicit output caps. The longest output is a check: reasoning + quotes for a
# 3-5 sentence answer, well under this.
MAX_OUTPUT_TOKENS = 1024


@dataclass(frozen=True)
class ChatProvider:
    name: Literal["anthropic", "google"]
    model: str
    chat: BaseChatModel


def build_providers(settings: Settings) -> tuple[ChatProvider, ChatProvider]:
    """(Claude, Gemini) from settings.

    Claude: no `temperature`/`top_p`/`top_k` (Haiku 5.5 rejects them; ChatAnthropic leaves
    them out when they are None). Thinking explicitly disabled: Haiku 5.5 thinks
    (adaptively) when `thinking` is omitted, which spent ~700 of 1000 output tokens on a
    Chinese answer in a live run and cut it off; `disabled` is accepted at its default
    effort (medium).
    Gemini: thinking as low as the model allows (see `gemini_thinking`).
    Reasoning, where a node needs it, is a field of the structured output instead.
    A missing key is not an error here: that provider's calls fail and the other one is used.
    """
    anthropic_key = settings.anthropic_api_key
    gemini_key = settings.gemini_api_key
    claude = ChatAnthropic(
        model=settings.answer_model,
        api_key=anthropic_key.get_secret_value() if anthropic_key else "missing",
        max_tokens=MAX_OUTPUT_TOKENS,
        thinking={"type": "disabled"},
        timeout=CALL_TIMEOUT_S,
        max_retries=1,
    )
    gemini = ChatGoogleGenerativeAI(
        model=settings.check_model,
        google_api_key=gemini_key.get_secret_value() if gemini_key else "missing",
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=CALL_TIMEOUT_S,
        max_retries=1,
        **gemini_thinking(settings.check_model),
    )
    return (
        ChatProvider("anthropic", settings.answer_model, claude),
        ChatProvider("google", settings.check_model, gemini),
    )


def gemini_thinking(model: str) -> dict[str, object]:
    """Explicit thinking settings for a Gemini model id.

    Gemini 2.x: `thinking_budget=0` turns thinking off; temperature 0 (library default 0.7).
    Gemini 3 and later reject `thinking_budget=0` (400 INVALID_ARGUMENT, tried 2026-11-17 on
    gemini-3.5-flash-lite) and ignore temperature; `thinking_level="minimal"` is the lowest.
    """
    if model.removeprefix("models/").startswith("gemini-2."):
        return {"thinking_budget": 0, "temperature": 0}
    return {"thinking_level": "minimal"}


class EmptyReply(Exception):
    """The model answered with nothing usable."""


def _log_failure(node: str, provider: ChatProvider, error: BaseException) -> None:
    logger.warning(
        "qa model call failed: node=%s provider=%s model=%s error=%s",
        node,
        provider.name,
        provider.model,
        type(error).__name__,
    )


async def structured[T: BaseModel](
    node: str,
    order: Sequence[ChatProvider],
    schema: type[T],
    messages: list[BaseMessage],
    *,
    deadline_s: float,
) -> tuple[T, ChatProvider]:
    """Structured output from the first provider in `order` that succeeds."""
    last: BaseException | None = None
    for provider in order:
        try:
            async with asyncio.timeout(deadline_s):
                raw = await provider.chat.with_structured_output(schema).ainvoke(messages)
            if raw is None:
                raise EmptyReply
            value = raw if isinstance(raw, schema) else schema.model_validate(raw)
            return value, provider
        except Exception as error:  # noqa: BLE001 - any provider failure means: try the next one
            _log_failure(node, provider, error)
            last = error
    raise ModelsUnavailable(f"{node}: every provider failed") from last


async def stream_text(
    node: str,
    order: Sequence[ChatProvider],
    messages: list[BaseMessage],
    *,
    deadline_s: float,
    on_switch: Callable[[], None] | None = None,
) -> tuple[str, ChatProvider]:
    """Streamed plain text from the first provider in `order` that succeeds.

    LangGraph's `messages` stream picks the tokens up as they arrive. Before switching to
    the next provider `on_switch` runs (the answer node sends `reset fallback`, so the page
    clears any half answer; with nothing shown yet the reset is harmless).
    """
    last: BaseException | None = None
    for index, provider in enumerate(order):
        parts: list[str] = []
        try:
            async with asyncio.timeout(deadline_s):
                async for chunk in provider.chat.astream(messages):
                    text = message_text(chunk)
                    if text:
                        parts.append(text)
            answer = "".join(parts).strip()
            if not answer:
                raise EmptyReply
            return answer, provider
        except Exception as error:  # noqa: BLE001 - any provider failure means: try the next one
            _log_failure(node, provider, error)
            last = error
            if on_switch is not None and index + 1 < len(order):
                on_switch()
    raise ModelsUnavailable(f"{node}: every provider failed") from last
