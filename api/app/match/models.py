"""Which model each job-match step uses, with which thinking level and output cap.

Every step (read, match, check, write) has its own ordered list "first -> fallbacks" of
`ModelChoice(provider, model, effort)` and its own output cap (`StepModels`). The graph
(`app.match.graph`) and the eval replays build the same chat objects from it
(`build_step_providers`), so a replay of one step uses exactly what the product would.

`effort` per provider (official pages, read 2026-11-19):
- anthropic: `off` = `thinking: disabled` (Haiku 5.5 accepts it at its default effort).
  Sonnet 5.5 rejects `disabled` (400); its lowest setting is `thinking: between_tools`, which
  without tools means no up-front thinking, so `off` sends that for the models in
  `ANTHROPIC_NO_DISABLE`. `low` .. `max` = `thinking: adaptive` + `output_config.effort`.
  Thinking counts toward `max_tokens` and is billed as output.
- google: `minimal` is the lowest (`app.qa.models.gemini_thinking`: Gemini 3 rejects
  `thinking_budget=0`, Gemini 2.x turns thinking off with it); `low` .. `high` set
  `thinking_level`.
- openai: `reasoning_effort` as written; `none` turns reasoning off (Luna defaults to
  `medium`, so it is always sent). `none` and `off` are different words on purpose.
  Reasoning counts toward `max_completion_tokens` and is billed as output.

Defaults equal the product before round 2: read, match and write on Claude Haiku 5.5 with
thinking off, Gemini 3.5 Flash-Lite as fallback; check on Gemini (minimal) with Haiku as
fallback; every step capped at 1024 output tokens. The Q&A (`app.qa`) does not use this.

Command-line form (eval tools parse it with `apply_overrides`):
`--model check=openai:gpt-6-luna:none`, `--fallback check=` (no fallback) or
`--fallback check=anthropic:claude-haiku-5-5:off,google:gemini-3.5-flash-lite:minimal`,
`--max-output read=4096`.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.qa.models import CALL_TIMEOUT_S, MAX_OUTPUT_TOKENS, ChatProvider, gemini_thinking

Step = Literal["read", "match", "check", "write"]
STEPS: tuple[Step, ...] = get_args(Step)
ProviderName = Literal["anthropic", "google", "openai"]
PROVIDERS: tuple[ProviderName, ...] = get_args(ProviderName)

EFFORTS: Mapping[ProviderName, tuple[str, ...]] = {
    "anthropic": ("off", "low", "medium", "high", "xhigh", "max"),
    "google": ("minimal", "low", "medium", "high"),
    "openai": ("none", "minimal", "low", "medium", "high", "xhigh", "max"),
}
# The effort that means "no thinking" (or the lowest there is) per provider.
LOWEST: Mapping[ProviderName, str] = {"anthropic": "off", "google": "minimal", "openai": "none"}
# Anthropic models that reject `thinking: disabled` (400); `off` sends `between_tools`.
ANTHROPIC_NO_DISABLE = frozenset({"claude-sonnet-5-5"})

# Output caps. Without thinking the longest reply (a check of 24 pairs) is far below 1024.
# With thinking the cap also holds the thinking tokens (all three providers count them in
# the cap), so a thinking choice raises its step's default cap.
THINKING_MAX_OUTPUT_TOKENS = 4096


@dataclass(frozen=True)
class ModelChoice:
    provider: ProviderName
    model: str
    effort: str

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}")
        if not self.model.strip():
            raise ValueError("empty model id")
        if self.effort not in EFFORTS[self.provider]:
            allowed = ", ".join(EFFORTS[self.provider])
            raise ValueError(f"{self.provider} effort {self.effort!r}: use one of {allowed}")

    @property
    def thinks(self) -> bool:
        return self.effort != LOWEST[self.provider]

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}:{self.effort}"


@dataclass(frozen=True)
class StepModels:
    """One step: the choices in order (first, then fallbacks) and the step's output cap."""

    choices: tuple[ModelChoice, ...]
    max_output_tokens: int

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("a step needs at least one model")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

    @property
    def first(self) -> ModelChoice:
        return self.choices[0]


type MatchModels = Mapping[Step, StepModels]


def default_cap(choices: Iterable[ModelChoice]) -> int:
    return THINKING_MAX_OUTPUT_TOKENS if any(c.thinks for c in choices) else MAX_OUTPUT_TOKENS


def step_models(*choices: ModelChoice, max_output_tokens: int | None = None) -> StepModels:
    return StepModels(tuple(choices), max_output_tokens or default_cap(choices))


def default_models(
    answer_model: str = "claude-haiku-5-5",
    check_model: str = "gemini-3.5-flash-lite",
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[Step, StepModels]:
    """The product's models (see the module docstring). `answer_model` and `check_model`
    come from `Settings`, so their environment overrides keep working."""
    claude = ModelChoice("anthropic", answer_model, "off")
    gemini = ModelChoice("google", check_model, "minimal")
    answer = StepModels((claude, gemini), max_output_tokens)
    return {
        "read": answer,
        "match": answer,
        "check": StepModels((gemini, claude), max_output_tokens),
        "write": answer,
    }


def models_for(settings: Settings) -> dict[Step, StepModels]:
    return default_models(settings.answer_model, settings.check_model)


# --- command-line overrides -----------------------------------------------------------------


def parse_choice(text: str) -> ModelChoice:
    """`provider:model:effort`, e.g. `openai:gpt-6-luna:none`."""
    parts = text.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"{text!r}: expected provider:model:effort")
    provider, model, effort = (p.strip() for p in parts)
    if provider not in PROVIDERS:
        raise ValueError(f"{text!r}: provider must be one of {', '.join(PROVIDERS)}")
    return ModelChoice(cast(ProviderName, provider), model, effort)


def _split_step(text: str) -> tuple[Step, str]:
    step, sep, value = text.partition("=")
    step = step.strip()
    if not sep or step not in STEPS:
        raise ValueError(f"{text!r}: expected <step>=..., step one of {', '.join(STEPS)}")
    return cast(Step, step), value.strip()


def apply_overrides(
    base: MatchModels,
    models: Sequence[str] = (),
    fallbacks: Sequence[str] = (),
    max_output: Sequence[str] = (),
) -> dict[Step, StepModels]:
    """`base` with `--model step=p:m:e`, `--fallback step=[p:m:e,...]` and
    `--max-output step=N` applied.

    `--model` replaces the first choice and keeps the base fallbacks (minus the new first),
    unless `--fallback` is given for that step: then the fallbacks are exactly that list
    (empty = none). A step whose models changed gets `default_cap` unless `--max-output`
    sets its cap."""
    out = dict(base)
    firsts: dict[Step, ModelChoice] = {}
    rests: dict[Step, tuple[ModelChoice, ...]] = {}
    caps: dict[Step, int] = {}
    for text in models:
        step, value = _split_step(text)
        firsts[step] = parse_choice(value)
    for text in fallbacks:
        step, value = _split_step(text)
        rests[step] = tuple(parse_choice(v) for v in value.split(",") if v.strip())
    for text in max_output:
        step, value = _split_step(text)
        try:
            caps[step] = int(value)
        except ValueError:
            raise ValueError(f"{text!r}: expected <step>=<tokens>") from None
    for step in STEPS:
        if step not in firsts and step not in rests and step not in caps:
            continue
        current = out[step]
        first = firsts.get(step, current.first)
        rest = rests.get(step, tuple(c for c in current.choices[1:] if c != first))
        choices = (first, *rest)
        changed = choices != current.choices
        cap = caps.get(step) or (default_cap(choices) if changed else current.max_output_tokens)
        out[step] = StepModels(choices, cap)
    return out


def describe(models: MatchModels) -> list[str]:
    """One line per step, for `--dry-run`: `read: a -> b (max 1024)`."""
    return [
        f"{step}: {' -> '.join(str(c) for c in models[step].choices)}"
        f" (max {models[step].max_output_tokens})"
        for step in STEPS
    ]


# --- chat objects ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepProvider(ChatProvider):
    """A `ChatProvider` built for one step: also carries its effort and output cap, which
    `graph.ask_json` reserves and sends."""

    effort: str = "off"
    max_output_tokens: int = MAX_OUTPUT_TOKENS


def _key(settings: Settings, provider: ProviderName) -> str:
    secret = {
        "anthropic": settings.anthropic_api_key,
        "google": settings.gemini_api_key,
        "openai": settings.openai_api_key,
    }[provider]
    # A missing key is not an error here: that provider's calls fail, the next one is used.
    return secret.get_secret_value() if secret else "missing"


def anthropic_thinking(choice: ModelChoice) -> dict[str, Any]:
    if choice.effort == "off":
        kind = "between_tools" if choice.model in ANTHROPIC_NO_DISABLE else "disabled"
        return {"thinking": {"type": kind}}
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": choice.effort}}


def google_thinking(choice: ModelChoice) -> dict[str, Any]:
    if choice.effort == "minimal":
        return gemini_thinking(choice.model)
    return {"thinking_level": choice.effort}


def build_chat(choice: ModelChoice, settings: Settings, max_output_tokens: int) -> BaseChatModel:
    """The chat object for one choice. No `temperature`/`top_p` anywhere: Haiku and Sonnet
    5.5 reject non-default values, and reasoning models ignore or reject them."""
    key = _key(settings, choice.provider)
    if choice.provider == "anthropic":
        return ChatAnthropic(
            model=choice.model,
            api_key=key,
            max_tokens=max_output_tokens,
            timeout=CALL_TIMEOUT_S,
            max_retries=1,
            **anthropic_thinking(choice),
        )
    if choice.provider == "google":
        return ChatGoogleGenerativeAI(
            model=choice.model,
            google_api_key=key,
            max_output_tokens=max_output_tokens,
            timeout=CALL_TIMEOUT_S,
            max_retries=1,
            **google_thinking(choice),
        )
    # Chat Completions (not the Responses API) with `store=False`: nothing kept by OpenAI
    # beyond its abuse-monitoring logs (PRODUCT-PRINCIPLES "数据不外流").
    return ChatOpenAI(
        model=choice.model,
        api_key=key,
        max_completion_tokens=max_output_tokens,
        reasoning_effort=choice.effort,
        store=False,
        timeout=CALL_TIMEOUT_S,
        max_retries=1,
    )


type ChatBuilder = Callable[[ModelChoice, Settings, int], BaseChatModel]


def build_step_providers(
    models: MatchModels, settings: Settings, build: ChatBuilder = build_chat
) -> dict[Step, tuple[StepProvider, ...]]:
    """Each step's providers in order. One chat object per (choice, cap)."""
    built: dict[tuple[ModelChoice, int], StepProvider] = {}
    out: dict[Step, tuple[StepProvider, ...]] = {}
    for step in STEPS:
        config = models[step]
        providers: list[StepProvider] = []
        for choice in config.choices:
            key = (choice, config.max_output_tokens)
            if key not in built:
                built[key] = StepProvider(
                    cast(Any, choice.provider),
                    choice.model,
                    build(choice, settings, config.max_output_tokens),
                    effort=choice.effort,
                    max_output_tokens=config.max_output_tokens,
                )
            providers.append(built[key])
        out[step] = tuple(providers)
    return out
