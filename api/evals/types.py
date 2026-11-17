"""Interfaces shared by the eval runner (run.py) and scoring/cost (ragas_score.py, cost.py)."""

from dataclasses import dataclass
from typing import Any, Protocol

METRICS = ("faithfulness", "response_relevancy", "context_recall", "factual_correctness")


class CapReached(Exception):
    """The pass cap or the total cap is used up: stop before the next paid call."""


class UnknownModel(Exception):
    """A model id with no price in evals/pricing.py: never count it as $0."""


@dataclass(frozen=True)
class ScoreInput:
    user_input: str  # the question's `scored_question`
    response: str
    retrieved_contexts: list[str]
    reference: str | None  # metrics that need it return None when it is None
    metrics: tuple[str, ...]  # subset of METRICS


class MeterLike(Protocol):
    """One per run: `Meter(caps, ledger_path, run_id)` in evals/cost.py, where
    caps = {"real": (run_cap, total_cap), "credit": (run_cap, total_cap)}. A model's
    pool comes from evals/pricing.py: Anthropic models are "credit" (the dev
    organisation's granted credit), Google and OpenAI models are "real"."""

    def check(self) -> None:
        """Raise UnknownModel if any call so far used a model with no price. Then re-read
        the ledger (another process may be spending) and raise CapReached if
        min(pass cap - this pass's cost, total cap - ledger total - cost not yet
        recorded) <= 0
        for either pool."""

    def callbacks(self) -> list[Any]:
        """LangChain callback handlers to pass in the graph's config; they add usage."""

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Add a call's cost. Never raises (it runs inside LangChain callbacks, whose
        errors are swallowed): a model with no price is remembered, and the next
        check() or record() raises UnknownModel."""

    def add_embedding_chars(self, model: str, chars: int) -> None:
        """Embeddings the callbacks cannot see: cost estimated from characters, doubled."""

    def record(self, question_id: str, repeat: int) -> float:
        """Append one ledger line per pool (under a file lock) with that pool's cost
        since the last record, which then counts as recorded; return the sum. Writes
        the lines first, then raises UnknownModel if an unpriced model was seen."""

    def start_pass(self) -> None:
        """A new pass starts: the pass caps apply to each pass separately."""

    @property
    def run_cost(self) -> dict[str, float]:
        """This run's cost so far per pool: {"real": ..., "credit": ...}."""
