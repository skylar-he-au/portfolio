import importlib
import os
from decimal import Decimal

import pytest
from guard_helpers import guard_settings
from pydantic import ValidationError

import app.config


def test_importing_config_turns_langchain_tracing_off(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    importlib.reload(app.config)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


@pytest.mark.parametrize(
    "field",
    [
        "origin_secret",
        "session_secret",
        "turnstile_secret",
        *[
            f"spend_{period}_usd_{provider}"
            for period in ("daily", "monthly")
            for provider in ("anthropic", "google", "openai")
        ],
    ],
)
def test_empty_guard_settings_are_none(field):
    settings = guard_settings("postgresql://unused", **{field: ""})
    assert getattr(settings, field) is None


def test_comma_separated_hostnames_from_environment(monkeypatch):
    monkeypatch.setenv("TURNSTILE_HOSTNAMES", " a.com, b.com, ,")
    settings = app.config.Settings(_env_file=None, database_url="postgresql://unused")
    assert settings.turnstile_hostnames == ["a.com", "b.com"]


@pytest.mark.parametrize("kind", ["ask", "match", "tts"])
@pytest.mark.parametrize("scope", ["token", "ip", "site"])
def test_daily_limits_must_be_positive(kind, scope):
    with pytest.raises(ValidationError):
        guard_settings("postgresql://unused", **{f"{kind}_limit_{scope}": 0})


def test_spend_settings_use_decimals():
    settings = guard_settings("postgresql://unused", spend_daily_usd_openai="0.10")
    assert settings.spend_daily_usd_openai == Decimal("0.10")
    assert isinstance(settings.spend_daily_usd_openai, Decimal)
    assert settings.spend_input_tokens_estimate == 12000
