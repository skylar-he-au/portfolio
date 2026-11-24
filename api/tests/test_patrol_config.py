"""Production checks every required variable; development requires none."""

import pytest
from fastapi.testclient import TestClient
from guard_helpers import guard_settings
from pydantic import ValidationError
from test_api import ExplodingPool

from app.main import create_app

REQUIRED = (
    "origin_secret",
    "session_secret",
    "turnstile_secret",
    "turnstile_hostnames",
    "anthropic_api_key",
    "gemini_api_key",
    "azure_openai_api_key",
    "azure_openai_endpoint",
    "azure_openai_deployment",
    "patrol_secret",
    *(
        f"spend_{period}_usd_{provider}"
        for period in ("daily", "monthly")
        for provider in ("anthropic", "google", "openai")
    ),
)


def complete_settings(url, **overrides):
    values = {
        "app_env": "prod",
        "turnstile_hostnames": ["example.com"],
        "anthropic_api_key": "test-anthropic",
        "gemini_api_key": "test-gemini",
        "azure_openai_api_key": "test-azure",
        "azure_openai_endpoint": "https://test.openai.azure.com/",
        "azure_openai_deployment": "text-embedding-3-small",
        "patrol_secret": "test-patrol",
    }
    return guard_settings(url, **(values | overrides))


@pytest.mark.parametrize("name", REQUIRED)
def test_production_requires_each_variable(refused_url, name):
    missing = [] if name == "turnstile_hostnames" else None
    settings = complete_settings(refused_url, **{name: missing})
    with pytest.raises(RuntimeError) as caught:
        create_app(settings, pool=ExplodingPool())
    assert str(caught.value) == f"Missing required variables: {name.upper()}"


@pytest.mark.parametrize("env", ["dev", "prod"])
def test_all_missing_variables_are_reported_only_outside_dev(refused_url, env):
    values = {name: [] if name == "turnstile_hostnames" else None for name in REQUIRED}
    settings = guard_settings(refused_url, app_env=env, **values)
    pool = ExplodingPool()
    if env == "prod":
        with pytest.raises(RuntimeError) as caught:
            create_app(settings, pool=pool)
        prefix, names = str(caught.value).split(": ", 1)
        assert prefix == "Missing required variables"
        assert set(names.split(", ")) == {name.upper() for name in REQUIRED}
    else:
        with TestClient(create_app(settings, pool=pool)) as client:
            assert client.get("/api/ping").status_code == 200
    assert pool.touched == []


def test_patrol_settings_defaults_and_blank_secret(refused_url):
    assert guard_settings(refused_url, patrol_secret="").patrol_secret is None
    assert guard_settings(refused_url).patrol_limit_day == 10
    with pytest.raises(ValidationError):
        guard_settings(refused_url, patrol_limit_day=0)


@pytest.mark.parametrize(
    "name",
    [
        "anthropic_api_key",
        "gemini_api_key",
        "azure_openai_api_key",
        "azure_openai_endpoint",
        "azure_openai_deployment",
        "patrol_secret",
    ],
)
def test_blank_production_keys_are_missing(refused_url, name):
    settings = complete_settings(refused_url, **{name: ""})
    with pytest.raises(RuntimeError, match=name.upper()):
        create_app(settings, pool=ExplodingPool())
