from decimal import Decimal

import pytest

from app.guard.prices import UnknownModel, cost_usd


def test_haiku_price_is_exact() -> None:
    actual = cost_usd("claude-haiku-5-5", {"input_tokens": 1000, "output_tokens": 1000})
    assert actual == Decimal("0.0006")


def test_cache_creation_only_adds_the_write_premium() -> None:
    usage = {
        "input_tokens": 1000,
        "input_token_details": {"cache_creation": 200, "cache_read": 300},
    }
    expected = (
        Decimal(1000) * Decimal("0.10") + Decimal(200) * Decimal("0.10") * Decimal("0.25")
    ) / Decimal(1000000)
    assert cost_usd("claude-haiku-5-5", usage) == expected


def test_reasoning_is_already_in_output_tokens() -> None:
    usage = {"input_tokens": 1000, "output_tokens": 1000}
    assert cost_usd("gemini-3.5-flash-lite", usage) == cost_usd(
        "gemini-3.5-flash-lite", {**usage, "output_token_details": {"reasoning": 700}}
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("models/gemini-3.5-flash-lite", Decimal("0.0028")),
        ("claude-haiku-5-5-20261001", Decimal("0.0006")),
        ("claude-sonnet-5-5", Decimal("0.012")),
        ("text-embedding-3-small", Decimal("0.000024")),
    ],
)
def test_model_names(model: str, expected: Decimal) -> None:
    assert cost_usd(model, {"input_tokens": 1000, "output_tokens": 1000}) == expected


def test_missing_and_none_counts_are_zero() -> None:
    assert cost_usd("claude-haiku-5-5", {}) == 0
    actual = cost_usd(
        "claude-haiku-5-5",
        {"input_tokens": None, "output_tokens": None, "input_token_details": None},
    )
    assert actual == 0
    actual = cost_usd("claude-haiku-5-5", {"input_token_details": {"cache_creation": None}})
    assert actual == 0


def test_unknown_model_is_not_free() -> None:
    with pytest.raises(UnknownModel):
        cost_usd("unknown", {})
