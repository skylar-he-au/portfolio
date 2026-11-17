"""Meter usage against per-pass and shared ledger caps."""

import csv
import fcntl
import logging
import os
import threading
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from evals.pricing import POOLS, cost_usd, price_for
from evals.types import CapReached, MeterLike, UnknownModel

logger = logging.getLogger(__name__)

_HEADER = ("run_id", "question_id", "repeat", "pool", "cost_usd", "at")


class Meter:
    """Track both spending pools and persist each question's usage."""

    def __init__(
        self, caps: dict[str, tuple[float, float]], ledger_path: str | Path, run_id: str
    ) -> None:
        missing = set(POOLS) - caps.keys()
        if missing:
            raise ValueError(f"Missing caps for pools: {', '.join(sorted(missing))}")
        self._caps = {pool: caps[pool] for pool in POOLS}
        self._ledger_path = Path(ledger_path)
        self._run_id = run_id
        self._lock = threading.Lock()
        self._pass_cost = dict.fromkeys(POOLS, 0.0)
        self._unrecorded = dict.fromkeys(POOLS, 0.0)
        self._run_total = dict.fromkeys(POOLS, 0.0)
        self._unpriced: set[str] = set()

    def _add(self, model: str, input_tokens: float, output_tokens: float) -> None:
        with self._lock:
            try:
                price = price_for(model)
                cost = cost_usd(model, input_tokens, output_tokens)
            except UnknownModel:
                self._unpriced.add(model)
                return
            self._pass_cost[price.pool] += cost
            self._unrecorded[price.pool] += cost
            self._run_total[price.pool] += cost

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Accumulate usage, deferring unknown-model errors."""
        self._add(model, input_tokens, output_tokens)

    def add_embedding_chars(self, model: str, chars: int) -> None:
        """Estimate embedding tokens from characters with a twofold margin."""
        self._add(model, chars / 4 * 2, 0)

    def callbacks(self) -> list[Any]:
        return [_UsageHandler(self)]

    def _raise_unpriced(self) -> None:
        if self._unpriced:
            names = ", ".join(repr(name) for name in sorted(self._unpriced))
            raise UnknownModel(f"Unpriced models: {names}")

    def _ledger_totals(self) -> dict[str, float]:
        totals = dict.fromkeys(POOLS, 0.0)
        try:
            with self._ledger_path.open(newline="", encoding="utf-8") as ledger:
                fcntl.flock(ledger.fileno(), fcntl.LOCK_SH)
                try:
                    for row in csv.DictReader(ledger):
                        pool = row.get("pool")
                        if pool not in totals:
                            continue
                        try:
                            cost = float(row.get("cost_usd", ""))
                        except (TypeError, ValueError):
                            continue
                        totals[pool] += cost
                finally:
                    fcntl.flock(ledger.fileno(), fcntl.LOCK_UN)
        except FileNotFoundError:
            return totals
        return totals

    def check(self) -> None:
        """Stop on unknown prices or an exhausted cap in either pool."""
        with self._lock:
            self._raise_unpriced()
            totals = self._ledger_totals()
            for pool in POOLS:
                pass_cap, total_cap = self._caps[pool]
                pass_remaining = pass_cap - self._pass_cost[pool]
                total_spent = totals[pool] + self._unrecorded[pool]
                remaining = min(pass_remaining, total_cap - total_spent)
                if remaining <= 0:
                    reasons = []
                    if pass_remaining <= 0:
                        reasons.append(f"pass cap: cost={self._pass_cost[pool]}, cap={pass_cap}")
                    if total_spent >= total_cap:
                        reasons.append(
                            f"total cap: ledger={totals[pool]}, "
                            f"unrecorded={self._unrecorded[pool]}, cap={total_cap}"
                        )
                    raise CapReached(f"{pool}: {'; '.join(reasons)}")

    def record(self, question_id: str, repeat: int) -> float:
        """Append both pools atomically, then report any unknown prices."""
        with self._lock:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self._ledger_path.open("a", newline="", encoding="utf-8") as ledger:
                fcntl.flock(ledger.fileno(), fcntl.LOCK_EX)
                try:
                    ledger.seek(0, os.SEEK_END)
                    writer = csv.writer(ledger)
                    if ledger.tell() == 0:
                        writer.writerow(_HEADER)
                    at = datetime.now(UTC).isoformat()
                    for pool in POOLS:
                        writer.writerow(
                            (
                                self._run_id,
                                question_id,
                                repeat,
                                pool,
                                self._unrecorded[pool],
                                at,
                            )
                        )
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    recorded = sum(self._unrecorded.values())
                    self._unrecorded = dict.fromkeys(POOLS, 0.0)
                finally:
                    fcntl.flock(ledger.fileno(), fcntl.LOCK_UN)
            self._raise_unpriced()
            return recorded

    def start_pass(self) -> None:
        with self._lock:
            self._pass_cost = dict.fromkeys(POOLS, 0.0)

    @property
    def run_cost(self) -> dict[str, float]:
        with self._lock:
            return self._run_total.copy()


class _UsageHandler(BaseCallbackHandler):
    run_inline = True
    raise_error = False

    def __init__(self, meter: MeterLike) -> None:
        self._meter = meter

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            for generations in response.generations:
                for generation in generations:
                    try:
                        message = getattr(generation, "message", None)
                        usage = getattr(message, "usage_metadata", None)
                        if usage is None:
                            continue
                        metadata = message.response_metadata
                        model = metadata.get("model_name") or metadata.get("model_version") or ""
                        self._meter.add_usage(model, usage["input_tokens"], usage["output_tokens"])
                    except Exception:
                        logger.warning("could not read usage from a generation", exc_info=True)
        except Exception:
            logger.warning("could not read usage from an LLM response", exc_info=True)


def meter_anthropic(client: Any, meter: MeterLike) -> Any:
    """Meter Anthropic message responses without changing their behavior."""
    original = client.messages.create

    @wraps(original)
    async def create(*args: Any, **kwargs: Any) -> Any:
        response = await original(*args, **kwargs)
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                meter.add_usage(response.model, usage.input_tokens, usage.output_tokens)
        except Exception:  # noqa: BLE001 - metering must not hide a successful response
            return response
        return response

    client.messages.create = create
    return client


def meter_openai(client: Any, meter: MeterLike) -> Any:
    """Meter OpenAI embedding responses without changing their behavior."""
    original = client.embeddings.create

    @wraps(original)
    async def create(*args: Any, **kwargs: Any) -> Any:
        response = await original(*args, **kwargs)
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                model = getattr(response, "model", None) or kwargs.get("model", "")
                meter.add_usage(model, usage.prompt_tokens, 0)
        except Exception:  # noqa: BLE001 - metering must not hide a successful response
            return response
        return response

    client.embeddings.create = create
    return client
