"""RAGAS metrics with metered clients and per-metric failure handling."""

import asyncio
import logging
import math
import os
import weakref
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms.base import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from app.config import get_settings
from evals.cost import meter_anthropic, meter_openai
from evals.types import METRICS, CapReached, MeterLike, ScoreInput, UnknownModel

logger = logging.getLogger(__name__)

# ragas posts usage analytics to its own server by default; the eval never sends anything there.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
_METRIC_CACHE: weakref.WeakKeyDictionary[MeterLike, dict[tuple[str, str], dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)


def make_llm(meter: MeterLike, judge_model: str, *, client: Any = None) -> Any:
    """Build the judge with usage recorded at the SDK boundary."""
    if client is None:
        key = get_settings().anthropic_api_key
        if key is None or not key.get_secret_value():
            raise RuntimeError("ANTHROPIC_API_KEY is required for eval scoring")
        client = AsyncAnthropic(api_key=key.get_secret_value())
    return llm_factory(
        judge_model,
        provider="anthropic",
        client=meter_anthropic(client, meter),
        output_config={"effort": "low"},
        max_tokens=4096,
    )


def make_embeddings(meter: MeterLike, embed_model: str, *, client: Any = None) -> Any:
    """Build metered OpenAI embeddings."""
    if client is None:
        key = get_settings().openai_api_key
        if key is None or not key.get_secret_value():
            raise RuntimeError("OPENAI_API_KEY is required for eval embeddings")
        client = AsyncOpenAI(api_key=key.get_secret_value())
    return embedding_factory("openai", embed_model, client=meter_openai(client, meter))


def _build_metrics(meter: MeterLike, judge_model: str, embed_model: str) -> dict[str, Any]:
    # Cached clients must not retain the weak cache key.
    meter = weakref.proxy(meter)
    llm = make_llm(meter, judge_model)
    embeddings = make_embeddings(meter, embed_model)
    return {
        "faithfulness": Faithfulness(llm),
        "response_relevancy": AnswerRelevancy(llm, embeddings),
        "context_recall": ContextRecall(llm),
        "factual_correctness": FactualCorrectness(llm),
    }


async def score(
    item: ScoreInput,
    meter: MeterLike,
    judge_model: str = "claude-sonnet-5-5",
    embed_model: str = "text-embedding-3-small",
) -> dict[str, float | None]:
    """Score requested metrics in order, skipping missing references."""
    scores: dict[str, float | None] = dict.fromkeys(METRICS)
    for name in METRICS:
        if name not in item.metrics:
            continue
        if name in ("context_recall", "factual_correctness") and item.reference is None:
            continue
        # off the loop: another process holding the ledger must not stall it (evals/cost.py)
        await asyncio.to_thread(meter.check)
        try:
            cached = _METRIC_CACHE.setdefault(meter, {})
            models = (judge_model, embed_model)
            if models not in cached:
                cached[models] = _build_metrics(meter, judge_model, embed_model)
            kwargs: dict[str, Any]
            if name == "faithfulness":
                kwargs = {
                    "user_input": item.user_input,
                    "response": item.response,
                    "retrieved_contexts": item.retrieved_contexts,
                }
            elif name == "response_relevancy":
                kwargs = {"user_input": item.user_input, "response": item.response}
            elif name == "context_recall":
                kwargs = {
                    "user_input": item.user_input,
                    "retrieved_contexts": item.retrieved_contexts,
                    "reference": item.reference,
                }
            else:
                kwargs = {"response": item.response, "reference": item.reference}
            result = await cached[models][name].ascore(**kwargs)
            value = float(result.value)
            scores[name] = None if math.isnan(value) else value
        except (CapReached, UnknownModel):
            raise
        except Exception as error:  # noqa: BLE001 - isolate failures to the current metric
            logger.warning("eval metric %s failed: %s", name, type(error).__name__)
    return scores
