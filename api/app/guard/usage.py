"""Meter model calls without depending on the Q&A graph or API layer."""

import logging
from collections.abc import Mapping
from types import TracebackType
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from app.guard.ledger import ProviderBudgetExhausted, current_scope
from app.guard.prices import cost_usd

logger = logging.getLogger(__name__)


def estimate_input_tokens(messages) -> int:
    characters = 0
    for message in messages:
        content = message.content
        if isinstance(content, str):
            characters += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    characters += len(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    characters += len(block["text"])
    return (characters + 2) // 3


class UsageRecorder(BaseCallbackHandler):
    run_inline = True
    raise_error = False

    def __init__(self) -> None:
        self._usage: dict[str, Any] | None = None

    def _add(self, usage: Mapping[str, Any]) -> None:
        # Build a complete copy before publishing so malformed metadata cannot partially add.
        previous = self._usage or {}
        total = {
            key: (previous.get(key) or 0) + (usage.get(key) or 0)
            for key in ("input_tokens", "output_tokens")
        }
        for key, fields in (
            ("input_token_details", ("cache_read", "cache_creation")),
            ("output_token_details", ("reasoning",)),
        ):
            before = previous.get(key) or {}
            details = usage.get(key) or {}
            total[key] = {
                name: (before.get(name) or 0) + (details.get(name) or 0) for name in fields
            }
        self._usage = total

    def on_llm_end(self, response, **kwargs: Any) -> None:
        try:
            for generations in response.generations:
                for generation in generations:
                    try:
                        message = getattr(generation, "message", None)
                        usage = getattr(message, "usage_metadata", None)
                        if usage is not None:
                            self._add(usage)
                    except Exception as error:  # noqa: BLE001 - callbacks must never fail a call
                        logger.warning("could not read model usage: %s", type(error).__name__)
        except Exception as error:  # noqa: BLE001 - malformed callback responses are harmless
            logger.warning("could not read model usage: %s", type(error).__name__)

    @property
    def usage(self) -> dict[str, Any] | None:
        return self._usage


class Metered:
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        input_tokens: int,
        max_output_tokens: int | None = None,
        embedding: bool = False,
    ) -> None:
        self.scope = current_scope.get()
        self.provider = provider
        self.model = model
        self.input_tokens = input_tokens
        self.max_output_tokens = max_output_tokens
        self.embedding = embedding
        self.callbacks: list[UsageRecorder] = []
        self.recorder = UsageRecorder()
        self.worst = None

    async def __aenter__(self) -> "Metered":
        if self.scope is not None:
            worst = cost_usd(
                self.model,
                {"input_tokens": self.input_tokens, "output_tokens": self.max_output_tokens or 0},
            )
            await self.scope.ensure(self.provider, worst)
            self.worst = worst
            if not self.embedding:
                self.callbacks = [self.recorder]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # No awaits: cancellation cannot interrupt the in-memory accounting.
        if self.scope is not None and self.worst is not None:
            actual = None
            if self.embedding:
                actual = self.worst
            elif self.recorder.usage is not None:
                actual = cost_usd(self.model, self.recorder.usage)
            self.scope.finish(self.provider, self.worst, actual)


def metered(
    provider: str,
    model: str,
    *,
    input_tokens: int,
    max_output_tokens: int | None = None,
    embedding: bool = False,
) -> Metered:
    return Metered(
        provider,
        model,
        input_tokens=input_tokens,
        max_output_tokens=max_output_tokens,
        embedding=embedding,
    )


def _status_and_text(error: BaseException) -> tuple[int | None, str]:
    response = getattr(error, "response", None)
    status = next(
        (
            value
            for value in (
                getattr(error, "status_code", None),
                getattr(error, "code", None),
                getattr(response, "status_code", None),
            )
            if isinstance(value, int)
        ),
        None,
    )
    text = str(error)
    body = getattr(error, "body", None)
    if body is not None:
        text += repr(body)
    if isinstance(response, Mapping):
        text += repr(response)
    elif response is not None:
        try:
            response_body = response.json()
        except Exception:  # noqa: BLE001 - an SDK response need not contain JSON
            response_body = None
        if response_body is not None:
            text += repr(response_body)
    return status, text.lower()


def is_budget_exhausted(provider: str, error: BaseException) -> bool:
    chain = [(error, 0)]
    seen: set[int] = set()
    while chain:
        current, depth = chain.pop()
        if id(current) in seen or depth >= 5:
            continue
        seen.add(id(current))
        status, text = _status_and_text(current)
        if provider == "anthropic":
            if status == 400 and ("usage limits" in text or "credit balance is too low" in text):
                return True
            if status == 429 and "enforced_spend_limit_reached" in text:
                return True
        elif provider == "openai":
            if any(
                code in text
                for code in (
                    "credit_balance_exhausted",
                    "organization_spend_limit_exceeded",
                    "project_spend_limit_exceeded",
                    "organization_usage_limit_exceeded",
                )
            ):
                return True
        elif provider == "google":
            if status == 402 or (status == 429 and "prepayment credits are depleted" in text):
                return True
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                chain.append((linked, depth + 1))
    return False


async def raise_if_budget_exhausted(provider: str, error: BaseException) -> None:
    scope = current_scope.get()
    if scope is not None and is_budget_exhausted(provider, error):
        try:
            await scope.ledger.pause_provider(provider)
        except Exception as pause_error:  # noqa: BLE001 - preserve the provider's budget error
            logger.warning("could not pause AI provider: %s", type(pause_error).__name__)
        raise ProviderBudgetExhausted(provider) from error
