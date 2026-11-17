"""Token prices in USD per million tokens."""

from dataclasses import dataclass

from evals.types import UnknownModel

POOLS = ("real", "credit")


@dataclass(frozen=True)
class Price:
    pool: str
    input_per_m: float
    output_per_m: float


_PRICES = {
    # Anthropic pricing: prompts below 100k tokens (ours are far smaller).
    "claude-haiku-5-5": Price("credit", 0.10, 0.50),
    # Anthropic pricing.
    "claude-sonnet-5-5": Price("credit", 2.00, 10.00),
    # Google Gemini API pricing; output tokens already include thinking tokens.
    "gemini-3.5-flash-lite": Price("real", 0.30, 2.50),
    # https://developers.openai.com/api/docs/pricing (checked, Standard tier).
    "text-embedding-3-small": Price("real", 0.02, 0.00),
}
_PREFIXES = tuple(sorted(_PRICES, key=len, reverse=True))


def price_for(model: str) -> Price:
    """Find the longest matching model prefix."""
    name = model.removeprefix("models/")
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return _PRICES[prefix]
    raise UnknownModel(model)


def cost_usd(model: str, input_tokens: float, output_tokens: float) -> float:
    """Price token usage, including fractional embedding estimates."""
    price = price_for(model)
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1e6
