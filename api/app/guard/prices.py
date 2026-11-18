"""Conservative, exact USD prices for the models used by the application."""

from collections.abc import Mapping
from decimal import Decimal
from typing import Any


class UnknownModel(Exception):
    """A model cannot be charged without a known price."""


_PRICES = {
    "claude-haiku-5-5": (Decimal("0.10"), Decimal("0.50")),
    "claude-sonnet-5-5": (Decimal("2.00"), Decimal("10.00")),
    "gemini-3.5-flash-lite": (Decimal("0.30"), Decimal("2.50")),
    "text-embedding-3-small": (Decimal("0.02"), Decimal("0")),
}
_MILLION = Decimal("1000000")


def cost_usd(model: str, usage: Mapping[str, Any]) -> Decimal:
    model = model.removeprefix("models/")
    for prefix in sorted(_PRICES, key=len, reverse=True):
        if model.startswith(prefix):
            input_price, output_price = _PRICES[prefix]
            break
    else:
        raise UnknownModel(model)

    input_tokens = Decimal(str(usage.get("input_tokens") or 0))
    output_tokens = Decimal(str(usage.get("output_tokens") or 0))
    details = usage.get("input_token_details") or {}
    cache_creation = Decimal(str(details.get("cache_creation") or 0))
    # Cache tokens are already included in input_tokens; reasoning is in output_tokens.
    return (
        input_tokens * input_price
        + cache_creation * input_price * Decimal("0.25")
        + output_tokens * output_price
    ) / _MILLION
