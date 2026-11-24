"""Ingest stays under an Azure deployment's tokens-per-minute limit: token-sized batches,
one per window, and a 429 waits and tries again. Waiting is injected, so nothing sleeps."""

import asyncio
import logging
from pathlib import Path

import httpx
import openai
import pytest
from fakes import FakeEmbedder

import app.ingest as ingest_module
from app.db import Database
from app.ingest import BATCH_SIZE, IngestError, Pacing, ingest, load_content, plan_batches
from app.rag.embed import estimate_tokens

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "content"


def rate_limit_error(headers: dict[str, str] | None = None) -> openai.RateLimitError:
    response = httpx.Response(
        429, headers=headers or {}, request=httpx.Request("POST", "https://x/embeddings")
    )
    body = {"error": {"code": "429", "message": "exceeded call rate limit"}}
    return openai.RateLimitError("rate limited", response=response, body=body)


class Clock:
    """A fake clock: `sleep` only moves time forward and records the wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


class LimitedEmbedder(FakeEmbedder):
    """Answers 429 when a request would take the last minute's tokens past the limit,
    the way a deployment's TPM limit does; can also fail the first `fail_first` calls."""

    def __init__(self, clock: Clock, tokens_per_minute: int, *, fail_first: int = 0) -> None:
        super().__init__()
        self.clock = clock
        self.tokens_per_minute = tokens_per_minute
        self.fail_first = fail_first
        self.attempts = 0
        self.accepted: list[tuple[float, int]] = []
        self.rejected = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.attempts += 1
        tokens = estimate_tokens(texts)
        recent = sum(t for at, t in self.accepted if self.clock.now - at < 60)
        if self.attempts <= self.fail_first or recent + tokens > self.tokens_per_minute:
            self.rejected += 1
            raise rate_limit_error()
        self.accepted.append((self.clock.now, tokens))
        return await super().embed(texts)


class NoWriteDb(Database):
    """Embedding happens before the transaction; stop there."""

    def __init__(self) -> None:
        super().__init__(object())

    def connection(self):  # type: ignore[override]
        raise RuntimeError("stop before writing")


def write_page(tmp_path: Path, sections: int, words: int = 200) -> None:
    body = "".join(f"## Section {n}\n{' '.join(['word'] * words)}\n" for n in range(sections))
    (tmp_path / "page.md").write_text(f"---\ntitle: T\nrag: true\n---\n{body}", encoding="utf-8")


def test_batches_stay_under_half_the_tpm() -> None:
    texts = ["x" * 300] * 50  # 100 estimated tokens each
    batches = plan_batches(texts, Pacing(5000))
    assert [len(batch) for batch in batches] == [25, 25]
    assert all(estimate_tokens(batch) <= 2500 for batch in batches)
    assert [text for batch in batches for text in batch] == texts


def test_a_text_over_the_batch_limit_goes_alone() -> None:
    texts = ["a" * 30, "b" * 9000, "c" * 30]
    assert plan_batches(texts, Pacing(4000)) == [[texts[0]], [texts[1]], [texts[2]]]


def test_without_pacing_only_the_count_limit_applies() -> None:
    texts = ["x" * 9000] * (BATCH_SIZE + 1)
    assert [len(batch) for batch in plan_batches(texts, None)] == [BATCH_SIZE, 1]
    assert [len(batch) for batch in plan_batches(texts[:3], Pacing(1000))] == [1, 1, 1]


def test_real_content_never_trips_a_5k_deployment() -> None:
    """The site's own pages, against a fake deployment that enforces 5K TPM."""
    clock = Clock()
    embedder = LimitedEmbedder(clock, 5000)
    pages = load_content(CONTENT)
    total = sum(len(page.sections) for page in pages)

    with pytest.raises(RuntimeError, match="stop before writing"):
        asyncio.run(ingest(NoWriteDb(), embedder, CONTENT, pacing=Pacing(5000), sleep=clock.sleep))

    assert embedder.rejected == 0
    assert sum(len(call) for call in embedder.calls) == total
    assert len(embedder.calls) > 1  # the site is bigger than one 2.5K-token batch
    assert all(estimate_tokens(call) <= 2500 for call in embedder.calls)
    # One wait of a whole window between batches, none before the first.
    assert clock.waits == [60.0] * (len(embedder.calls) - 1)


def test_429_waits_and_retries(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    write_page(tmp_path, sections=2)
    clock = Clock()
    embedder = LimitedEmbedder(clock, 5000, fail_first=2)
    caplog.set_level(logging.WARNING, logger="app.ingest")

    with pytest.raises(RuntimeError, match="stop before writing"):
        asyncio.run(ingest(NoWriteDb(), embedder, tmp_path, pacing=Pacing(5000), sleep=clock.sleep))

    assert embedder.attempts == 3
    assert len(embedder.calls) == 1
    assert clock.waits == [60.0, 60.0]
    assert caplog.text.count("rate limited; retry") == 2


def test_429_uses_a_longer_retry_after(tmp_path: Path) -> None:
    write_page(tmp_path, sections=1)
    clock = Clock()
    errors = [rate_limit_error({"retry-after": "90"}), rate_limit_error({"retry-after-ms": "5"})]

    class Embedder(FakeEmbedder):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            if errors:
                raise errors.pop(0)
            return await super().embed(texts)

    with pytest.raises(RuntimeError, match="stop before writing"):
        asyncio.run(
            ingest(NoWriteDb(), Embedder(), tmp_path, pacing=Pacing(5000), sleep=clock.sleep)
        )
    # The server's 90 s is longer than a window; 5 ms is not, so a window is waited.
    assert clock.waits == [90.0, 60.0]


def test_429_gives_up_after_the_retries(tmp_path: Path) -> None:
    write_page(tmp_path, sections=1)
    clock = Clock()
    embedder = LimitedEmbedder(clock, 5000, fail_first=100)

    with pytest.raises(openai.RateLimitError):
        asyncio.run(
            ingest(
                NoWriteDb(), embedder, tmp_path, pacing=Pacing(5000, retries=3), sleep=clock.sleep
            )
        )
    assert embedder.attempts == 4
    assert clock.waits == [60.0] * 3


def test_without_pacing_a_429_is_not_retried(tmp_path: Path) -> None:
    write_page(tmp_path, sections=1)
    clock = Clock()
    embedder = LimitedEmbedder(clock, 5000, fail_first=1)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(ingest(NoWriteDb(), embedder, tmp_path, sleep=clock.sleep))
    assert embedder.attempts == 1 and clock.waits == []


def test_wrong_vector_count_is_still_an_error(tmp_path: Path) -> None:
    write_page(tmp_path, sections=2, words=5)

    class Short(FakeEmbedder):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return (await super().embed(texts))[:-1]

    with pytest.raises(IngestError, match="wrong number of vectors"):
        asyncio.run(ingest(NoWriteDb(), Short(), tmp_path, pacing=Pacing(5000)))


def test_cli_ingest_failure_after_retries_logs_only_the_error_type(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def always_limited(*_args: object, **_kwargs: object) -> None:
        raise rate_limit_error()

    monkeypatch.setattr(ingest_module, "_run", always_limited)
    assert ingest_module.main([]) == 1
    assert "ingest failed: RateLimitError" in caplog.text
