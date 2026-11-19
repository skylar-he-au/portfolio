"""Token prices for the evals: the app's price table (`app.guard.prices`), as floats.

There is one price table, in the app; this module only adds what the evals need on top of
it: the spending pool of each model (the eval caps are per pool) and float costs. The app
counts in Decimal and adds the prompt-cache write premium (`input_token_details`); the eval
callbacks see only input and output tokens and none of the eval prompts write a cache.
A model the app has no price for raises `evals.types.UnknownModel` (never $0), so every
runner keeps catching the eval's own exception.
"""

from dataclasses import dataclass
from decimal import Decimal

from app.guard import prices as app_prices
from evals.types import UnknownModel

POOLS = ("real", "credit")

# Which pool pays: Anthropic models use the dev organisation's granted credit; Google and
# OpenAI are real money. A priced model with no pool here is unknown too.
_POOL_PREFIXES = (
    ("claude-", "credit"),
    ("gemini-", "real"),
    ("gpt-", "real"),
    ("text-embedding-", "real"),
)
_MILLION = 1_000_000


@dataclass(frozen=True)
class Price:
    pool: str
    input_per_m: float
    output_per_m: float


def _app_cost(model: str, input_tokens: float, output_tokens: float) -> Decimal:
    try:
        return app_prices.cost_usd(
            model, {"input_tokens": input_tokens, "output_tokens": output_tokens}
        )
    except app_prices.UnknownModel:
        raise UnknownModel(model) from None


def _pool(model: str) -> str:
    name = model.removeprefix("models/")
    for prefix, pool in _POOL_PREFIXES:
        if name.startswith(prefix):
            return pool
    raise UnknownModel(model)


def price_for(model: str) -> Price:
    """The app's price of `model` (longest matching prefix there) and its pool."""
    input_per_m = _app_cost(model, _MILLION, 0)
    output_per_m = _app_cost(model, 0, _MILLION)
    return Price(_pool(model), float(input_per_m), float(output_per_m))


def cost_usd(model: str, input_tokens: float, output_tokens: float) -> float:
    """Price token usage, including fractional embedding estimates."""
    _pool(model)
    return float(_app_cost(model, input_tokens, output_tokens))
