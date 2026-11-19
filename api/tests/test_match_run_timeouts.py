"""How long the match eval waits for one item (`evals.match_run`): one deadline for both
tries of the graph, an item past it is an error row and the run goes on, and a blocked
event loop is stopped by the `faulthandler` watchdog instead of stalling the run.
"""

import asyncio
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
from test_evals_match import JOB_FOR_GRAPH, EvalScript, _evaluate, eval_deps

from evals import match_run
from evals.match_checks import MatchJob

pytestmark = pytest.mark.anyio

SECOND = MatchJob(**{**JOB_FOR_GRAPH.__dict__, "id": "real-02"})


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Graph:
    """The real graph, except that the first `hangs` calls of `astream` never end (or end
    in an error after `fail_after` seconds)."""

    def __init__(self, real: Any, hangs: int = 1, fail_after: float | None = None) -> None:
        self.real = real
        self.hangs = hangs
        self.fail_after = fail_after
        self.calls = 0

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self.hangs:
            if self.fail_after is None:
                await asyncio.Event().wait()
            await asyncio.sleep(self.fail_after)
            raise RuntimeError("down")
        async for item in self.real.astream(*args, **kwargs):
            yield item


def real_graph() -> Any:
    deps = eval_deps(EvalScript())
    return match_run.make_graph(deps), deps


def test_one_item_waits_a_little_longer_than_the_api_does() -> None:
    from app.api.match import MATCH_TIMEOUT_S

    assert MATCH_TIMEOUT_S < match_run.SAMPLE_TIMEOUT_S <= MATCH_TIMEOUT_S + 15
    assert match_run.WATCHDOG_S > match_run.SAMPLE_TIMEOUT_S


async def test_an_item_that_hangs_is_an_error_row_and_the_next_item_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(match_run, "SAMPLE_TIMEOUT_S", 0.3)
    real, deps = real_graph()
    graph = Graph(real)
    started = time.monotonic()
    outcome = await match_run.evaluate_match(
        [JOB_FOR_GRAPH, SECOND],
        **_kwargs(tmp_path, deps, graph),
    )
    took = time.monotonic() - started
    first, second = outcome.rows
    assert first["status"] == "error" and first["error"] == "TimeoutError"
    assert first["ms"] < 600  # one deadline: no second try after the time ran out
    assert second["status"] == "ok" and second["passed"]
    assert graph.calls == 2  # real-01 once (timed out, not retried), real-02 once
    assert took < 2.0


async def test_a_quick_failure_is_retried_within_the_same_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(match_run, "SAMPLE_TIMEOUT_S", 1.0)
    real, deps = real_graph()
    graph = Graph(real, hangs=1, fail_after=0.0)
    outcome = await match_run.evaluate_match([JOB_FOR_GRAPH], **_kwargs(tmp_path, deps, graph))
    assert outcome.rows[0]["status"] == "ok" and graph.calls == 2


async def test_the_retry_gets_only_what_is_left_of_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(match_run, "SAMPLE_TIMEOUT_S", 0.5)
    real, deps = real_graph()
    graph = Graph(real, hangs=2, fail_after=0.3)  # fails at 0.3 s, the retry would at 0.6
    outcome = await match_run.evaluate_match([JOB_FOR_GRAPH], **_kwargs(tmp_path, deps, graph))
    row = outcome.rows[0]
    assert row["status"] == "error" and row["error"] == "TimeoutError"
    assert graph.calls == 2 and row["ms"] < 800


async def test_the_watchdog_is_armed_for_the_graph_and_cleared_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(
        match_run.faulthandler,
        "dump_traceback_later",
        lambda timeout, **kw: seen.append(("arm", timeout, kw.get("exit"))),
    )
    monkeypatch.setattr(
        match_run.faulthandler, "cancel_dump_traceback_later", lambda: seen.append("clear")
    )
    _, deps = real_graph()
    await _evaluate(tmp_path, deps)
    assert seen[0] == ("arm", match_run.WATCHDOG_S, True)
    assert seen[1:] and set(seen[1:]) == {"clear"}


def test_a_blocked_event_loop_is_stopped_with_every_stack(tmp_path: Path) -> None:
    """An asyncio deadline cannot fire while a blocking call holds the loop (the r2g1
    stall had 0% CPU): the watchdog prints the stacks and ends the process."""
    script = textwrap.dedent(
        """
        import asyncio, sys, time
        sys.path.insert(0, "tests")
        from test_evals_match import _evaluate, eval_deps, EvalScript
        from evals import match_run
        from pathlib import Path

        match_run.WATCHDOG_S = 0.5

        class Blocking:
            async def astream(self, *a, **k):
                time.sleep(30)  # holds the event loop: no await, no timeout fires
                yield None

        deps = eval_deps(EvalScript())
        asyncio.run(_evaluate(Path(sys.argv[1]), deps, graph=Blocking()))
        print("finished")
        """
    )
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=Path(__file__).parent.parent,
    )
    took = time.monotonic() - started
    assert done.returncode == 1 and "finished" not in done.stdout
    assert "Timeout (" in done.stderr and "in astream" in done.stderr
    assert took < 10


def _kwargs(tmp_path: Path, deps: Any, graph: Any) -> dict[str, Any]:
    from test_evals_match import CAPS, FakeAligner, FakeScorer

    return {
        "graph": graph,
        "expected": match_run.expected_providers(deps, "default"),
        "route": "default",
        "meter": match_run.MatchMeter(CAPS, tmp_path / "ledger.csv", "run-1"),
        "aligner": FakeAligner(),
        "scorer": FakeScorer(),
        "repeats": 1,
        "run_id": "run-1",
        "out_path": tmp_path / "runs" / "run-1.jsonl",
        "worst": {"real": 0.01, "credit": 0.3},
        "summary_out": tmp_path / "summary",
    }
