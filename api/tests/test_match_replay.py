"""Replaying one job-match step on a stored run (evals.match_replay): no network, fake chats."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from test_evals_match import RR, ProviderChat
from test_match_graph import Script, kind, section

from app.api.match import match_worst_steps
from app.match import models as match_models
from app.match.graph import MatchDeps
from app.match.state import MAX_REQUIREMENTS
from app.qa.models import ChatProvider
from evals.grading import read_jsonl
from evals.match_checks import GoldRequirement, MatchJob
from evals.match_replay import (
    V4_RUN,
    Option,
    SourceError,
    batch_options,
    budget_check,
    checked_findings_of,
    later_per_item,
    ledger_totals,
    main,
    make_option,
    needed_urls,
    preflight,
    replay,
    replay_run_id,
    source_rows,
    step_worst,
)
from evals.match_run import STEPS, MatchMeter, _add, _pools
from evals.pricing import POOLS

pytestmark = pytest.mark.anyio

CAPS = {"real": (1.0, 5.0), "credit": (1.0, 5.0)}
DEFAULTS = match_models.default_models()
HAIKU_OFF = "anthropic:claude-haiku-5-5:off"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- a stored run ---------------------------------------------------------------------------

SECTIONS = {
    s.url: s
    for s in (
        section("roleradar", "python", "Skylar built it in Python."),
        section("roleradar", "sql", "Postgres queries."),
        section("roleradar", "lambda", "Three Lambda functions."),
    )
}
PY, SQL, LAMBDA = (RR + "python", RR + "sql", RR + "lambda")

JOBS = [
    MatchJob(
        id="real-01",
        kind="real",
        holdout=False,
        title="t",
        jd="We need Python, SQL and Kubernetes.",
        requirements=(
            GoldRequirement("real-01.r1", "Python", (PY,)),
            GoldRequirement("real-01.r2", "SQL", (SQL,)),
            GoldRequirement("real-01.r3", "Kubernetes", (), True),
        ),
    ),
    MatchJob(
        id="real-02",
        kind="real",
        holdout=True,
        title="t",
        jd="We need Python, SQL and Kubernetes.",
        requirements=(GoldRequirement("real-02.r1", "Python", (PY,)),),
    ),
]


def _ref(url: str) -> dict[str, str]:
    return {"url": url, "title": f"Heading {url.rsplit('#', 1)[1]}"}


def stored_row(job_id: str, repeat: int = 1) -> dict[str, Any]:
    """A row as runs before 6 r2 stored it (no checked findings, verdicts, section texts):
    the check removed SQL's link, so r2 is `unchecked` in the guide."""
    return {
        "run_id": V4_RUN,
        "id": job_id,
        "repeat": repeat,
        "status": "ok",
        "requirements": [
            {"id": "r1", "text": "Python", "query": "q Python"},
            {"id": "r2", "text": "SQL", "query": "q SQL"},
            {"id": "r3", "text": "Kubernetes", "query": "q Kubernetes"},
        ],
        "retrieved": {"r1": [PY, LAMBDA], "r2": [SQL], "r3": []},
        "findings": [
            {
                "requirement_id": "r1",
                "status": "found",
                "sections": [_ref(PY)],
                "provider": "a",
                "ms": 5,
            },
            {
                "requirement_id": "r2",
                "status": "found",
                "sections": [_ref(SQL)],
                "provider": "a",
                "ms": 5,
            },
            {
                "requirement_id": "r3",
                "status": "not_found",
                "sections": [],
                "provider": None,
                "ms": 0,
            },
        ],
        "result": {
            "items": [{"requirement_id": "r1", "text": "Old text.", "sections": [_ref(PY)]}],
            "not_covered": ["Kubernetes"],
            "unchecked": 1,
            "top8": False,
            "language": "en",
        },
    }


SOURCES = {job.id: stored_row(job.id) for job in JOBS}


class Recorder(Script):
    """Replies like the graph tests' script; records which step was asked."""

    def __init__(self, requirements: list[str] | None = None) -> None:
        names = requirements or ["Python", "SQL", "Kubernetes"]
        super().__init__([{"text": n, "query": f"q {n}"} for n in names])
        self.asked: list[str] = []

    async def __call__(self, messages: list[BaseMessage]) -> Any:
        self.asked.append(kind(messages))
        return await super().__call__(messages)


class NoSearch:
    """The stored run's sections are used: retrieval must never run again."""

    async def __call__(self, queries: Sequence[str], *, page_prefix: str) -> Any:
        raise AssertionError("search ran in a replay")


def deps_for(script: Recorder) -> MatchDeps:
    chat = ProviderChat(reply=script, provider="anthropic", model="claude-haiku-5-5")
    provider = ChatProvider("anthropic", "claude-haiku-5-5", chat)
    return MatchDeps(
        provider,
        provider,
        NoSearch(),
        stage_timeouts={"reading": 5.0, "matching": 5.0, "checking": 5.0, "writing": 5.0},
        call_timeout=5.0,
        steps={step: (provider,) for step in STEPS},
    )


def option_for(step: str, **kwargs: Any) -> Option:
    return make_option(step, DEFAULTS, [f"{step}={HAIKU_OFF}"], [f"{step}="], **kwargs)


async def run_replay(
    tmp_path: Path, step: str, script: Recorder | None = None, **kwargs: Any
) -> tuple[Any, Recorder, Path]:
    script = script or Recorder()
    option = kwargs.pop("option", None) or option_for(step)
    run_id = replay_run_id(datetime(2026, 11, 20, tzinfo=UTC), option)
    outcome = await replay(
        option,
        JOBS,
        SOURCES,
        deps=deps_for(script),
        # what `main` fetches: `needed_urls` (none for read)
        sections=kwargs.pop("sections", SECTIONS),
        meter=MatchMeter(CAPS, tmp_path / "ledger.csv", run_id),
        run_id=run_id,
        out_dir=tmp_path / "runs",
        tasks_out=tmp_path / "grading",
        meta={"source_run": V4_RUN, "source_pass": 1, "content_hash": "sha256:x"},
    )
    return outcome, script, tmp_path / "runs" / f"{run_id}.jsonl"


def _ledger(tmp_path: Path) -> list[list[str]]:
    lines = (tmp_path / "ledger.csv").read_text(encoding="utf-8").splitlines()
    return [line.split(",") for line in lines[1:]]


def _metered_every_call(tmp_path: Path, outcome: Any) -> None:
    """Every call has its tokens and cost, and the ledger holds exactly that spend."""
    calls = [c for row in outcome.rows for c in row["calls"]]
    assert calls and all(c["cost_usd"] is not None and c["input_tokens"] == 1000 for c in calls)
    ledger = _ledger(tmp_path)
    assert {(line[1], line[3]) for line in ledger} == {
        (job.id, pool) for job in JOBS for pool in POOLS
    }
    credit = sum(float(line[4]) for line in ledger if line[3] == "credit")
    assert credit == pytest.approx(sum(c["cost_usd"] for c in calls))
    assert credit == pytest.approx(sum(row["cost_usd"] for row in outcome.rows))


# --- each step ------------------------------------------------------------------------------


async def test_read_replay_calls_read_only_and_exports_align_tasks(tmp_path: Path) -> None:
    # no sections, as `main` gives a read replay (`needed_urls`): it once raised KeyError
    outcome, script, path = await run_replay(tmp_path, "read", sections={})
    assert outcome.status == "complete"
    assert script.asked == ["read", "read"]  # one per item, nothing downstream
    row = outcome.rows[0]
    assert [r["id"] for r in row["requirements"]] == ["r1", "r2", "r3"]
    assert "retrieved" not in row and "result" not in row and "findings" not in row
    assert row["step"] == "read" and row["candidate"] == "haiku-5-5-off"
    assert {c["node"] for c in row["calls"]} == {"read"}
    _metered_every_call(tmp_path, outcome)
    run_id = path.stem
    tasks = read_jsonl(tmp_path / "grading" / run_id / "tasks.jsonl")
    assert [t["task_id"] for t in tasks] == [f"{run_id}/{j.id}/1/align" for j in JOBS]
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "replay" and meta["step"] == "read" and meta["status"] == "complete"
    assert meta["source_run"] == V4_RUN and meta["content_hash"] == "sha256:x"
    assert meta["models"] == [HAIKU_OFF] and meta["grader"] == "subscription"


async def test_read_replay_can_read_twelve(tmp_path: Path) -> None:
    names = [f"Skill {n}" for n in range(1, 14)]
    option = option_for("read", max_requirements=12)
    outcome, _, path = await run_replay(tmp_path, "read", Recorder(names), option=option)
    row = outcome.rows[0]
    assert [r["id"] for r in row["requirements"]] == [f"r{n}" for n in range(1, 13)]
    assert row["top8"] is True  # the ad had more than 12
    assert path.stem.endswith("-replay-read-haiku-5-5-off-max12")
    tasks = read_jsonl(path.parent.parent / "grading" / path.stem / "tasks.jsonl")
    assert len(tasks[0]["payload"]["read"]) == 12


async def test_match_replay_uses_the_stored_requirements_and_sections(tmp_path: Path) -> None:
    outcome, script, path = await run_replay(tmp_path, "match")
    assert outcome.status == "complete"
    # r3 had nothing retrieved: no call, as in the graph; no read, search, check or write
    assert script.asked == ["match"] * 4
    row = outcome.rows[0]
    assert row["requirements"] == SOURCES["real-01"]["requirements"]
    assert row["retrieved"] == SOURCES["real-01"]["retrieved"]
    assert [(f["requirement_id"], f["status"]) for f in row["findings"]] == [
        ("r1", "found"), ("r2", "found"), ("r3", "not_found"),
    ]  # fmt: skip
    assert row["findings"][0]["sections"] == [{"url": PY, "title": "Heading python"}]
    assert "result" not in row and "checked_findings" not in row
    assert {c["node"] for c in row["calls"]} == {"match"}
    _metered_every_call(tmp_path, outcome)
    assert not (tmp_path / "grading" / path.stem).exists()  # scored with the stored alignment


async def test_check_replay_judges_the_stored_pairs(tmp_path: Path) -> None:
    class SaysNoToSql(Recorder):
        async def check(self, messages: list[BaseMessage]) -> Any:
            return {"verdicts": [{"n": 1, "shows": True}, {"n": 2, "shows": False}]}

    outcome, script, _ = await run_replay(tmp_path, "check", SaysNoToSql())
    assert script.asked == ["check", "check"]
    row = outcome.rows[0]
    assert row["findings"] == SOURCES["real-01"]["findings"]  # the stored ones, untouched
    assert row["check_verdicts"] == [
        {"requirement_id": "r1", "url": PY, "shows": True},
        {"requirement_id": "r2", "url": SQL, "shows": False},
    ]
    assert [(f["requirement_id"], f["status"]) for f in row["checked_findings"]] == [
        ("r1", "found"), ("r2", "unchecked"), ("r3", "not_found"),
    ]  # fmt: skip
    _metered_every_call(tmp_path, outcome)


async def test_write_replay_writes_from_the_stored_checked_findings(tmp_path: Path) -> None:
    outcome, script, path = await run_replay(tmp_path, "write")
    assert script.asked == ["write", "write"]
    row = outcome.rows[0]
    # rebuilt from the stored guide: r1 found (its guide link), r2 unchecked, r3 not found
    assert [(f["requirement_id"], f["status"]) for f in row["checked_findings"]] == [
        ("r1", "found"), ("r2", "unchecked"), ("r3", "not_found"),
    ]  # fmt: skip
    assert row["result"]["items"] == [
        {"requirement_id": "r1", "text": "guide text 1", "sections": [_ref(PY)]}
    ]
    assert row["result"]["not_covered"] == ["Kubernetes"] and row["result"]["unchecked"] == 1
    assert set(row["section_texts"]) == {PY}
    assert row["code_checks"] == dict.fromkeys(row["code_checks"], True)
    _metered_every_call(tmp_path, outcome)
    tasks = read_jsonl(tmp_path / "grading" / path.stem / "tasks.jsonl")
    assert [t["kind"] for t in tasks] == ["faith", "faith"]
    assert tasks[0]["payload"]["response"] == "guide text 1"
    assert tasks[0]["payload"]["contexts"][0].endswith("Skylar built it in Python.")


def test_checked_findings_are_the_stored_ones_when_the_row_has_them() -> None:
    row = {**stored_row("real-01"), "checked_findings": [{"requirement_id": "r1"}]}
    assert checked_findings_of(row) == [{"requirement_id": "r1"}]


async def test_an_item_failing_twice_is_an_error_row_and_three_stop_the_run(
    tmp_path: Path,
) -> None:
    class Down(Recorder):
        async def write(self, messages: list[BaseMessage]) -> Any:
            raise RuntimeError("down")

    jobs = [*JOBS, *(MatchJob(f"real-0{n}", "real", False, "t", "jd", JOBS[0].requirements)
                     for n in (3, 4))]  # fmt: skip
    option = option_for("write")
    script = Down()
    outcome = await replay(
        option,
        jobs,
        {job.id: stored_row(job.id) for job in jobs},
        deps=deps_for(script),
        sections=SECTIONS,
        meter=MatchMeter(CAPS, tmp_path / "ledger.csv", "r"),
        run_id="r",
        out_dir=tmp_path / "runs",
        tasks_out=tmp_path / "grading",
        meta={},
    )
    assert outcome.status == "incomplete" and "three consecutive errors" in str(outcome.reason)
    assert [r["status"] for r in outcome.rows] == ["error"] * 3
    assert script.asked == ["write"] * 6  # each item tried twice; the fourth never started


async def test_no_room_for_the_worst_case_stops_before_any_call(tmp_path: Path) -> None:
    script = Recorder()
    option = option_for("read")
    tiny = {"real": (1.0, 5.0), "credit": (1e-9, 5.0)}
    outcome = await replay(
        option, JOBS, SOURCES, deps=deps_for(script), sections=SECTIONS,
        meter=MatchMeter(tiny, tmp_path / "ledger.csv", "r"), run_id="r",
        out_dir=tmp_path / "runs", tasks_out=tmp_path / "grading", meta={},
    )  # fmt: skip
    assert outcome.status == "incomplete" and "CapReached" in str(outcome.reason)
    assert script.asked == []


# --- options --------------------------------------------------------------------------------


def test_options_set_only_the_replayed_step() -> None:
    option = make_option(
        "check", DEFAULTS, ["check=openai:gpt-6-luna:none"], ["check=" + HAIKU_OFF]
    )
    assert [str(c) for c in option.models.choices] == ["openai:gpt-6-luna:none", HAIKU_OFF]
    assert option.slug == "luna-none"
    with pytest.raises(ValueError, match="only the check step"):
        make_option("check", DEFAULTS, ["read=" + HAIKU_OFF])
    with pytest.raises(ValueError, match="read step only"):
        make_option("write", DEFAULTS, max_requirements=12)
    with pytest.raises(ValueError, match="one of"):
        make_option("read", DEFAULTS, max_requirements=10)
    sonnet = make_option("read", DEFAULTS, ["read=anthropic:claude-sonnet-5-5:off"])
    assert sonnet.slug == "sonnet-5-5-off"
    gemini = make_option("check", DEFAULTS)
    assert gemini.slug == "35-flash-lite-minimal"
    run_id = replay_run_id(datetime(2026, 11, 20, 1, 2, 3, tzinfo=UTC), option)
    assert run_id == "20261120T010203-match-v5+check-v2-replay-check-luna-none"


def test_step_worst_cases_add_up_to_the_apps_reservation() -> None:
    """The first models' worst cases of the four steps are the app's reservation for a
    match (`match_worst_steps`) without the embedding."""

    def one(text: str) -> match_models.StepModels:
        return match_models.step_models(match_models.parse_choice(text))

    models = {
        **DEFAULTS,
        "read": one("anthropic:claude-haiku-5-5:low"),
        "check": one("openai:gpt-6-luna:none"),
    }
    ours = _add(*(step_worst(Option(step, models[step]), first_only=True) for step in STEPS))
    app = match_worst_steps(models, "text-embedding-3-small")
    embed = match_worst_steps(DEFAULTS, "text-embedding-3-small")["openai"]
    app_chat = {
        "credit": float(app["anthropic"]),
        "real": float(app["google"] + app["openai"] - embed),
    }
    assert ours == {pool: pytest.approx(app_chat[pool]) for pool in POOLS}
    assert app["openai"] > embed  # Luna's check is in it
    assert isinstance(app["anthropic"], Decimal)


def test_step_worst_counts_every_model_in_the_order() -> None:
    option = make_option(
        "check", DEFAULTS, ["check=openai:gpt-6-luna:none"], ["check=" + HAIKU_OFF]
    )
    first = step_worst(option, first_only=True)
    both = step_worst(option)
    assert first["credit"] == 0 and both["credit"] > 0 and both["real"] == first["real"]


def test_dry_run_batch_reports_order_worst_and_contract_12(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = tmp_path / "batch.txt"
    batch.write_text(
        "# read\n"
        "--step read --model read=anthropic:claude-haiku-5-5:low --fallback read=" + HAIKU_OFF
        + "\n--step read --model read=" + HAIKU_OFF + " --fallback read= --max-requirements 12\n"
        "--step check --model check=openai:gpt-6-luna:none --fallback check=" + HAIKU_OFF + "\n",
        encoding="utf-8",
    )  # fmt: skip
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "run_id,question_id,repeat,pool,cost_usd,at\nx,a,1,credit,0.5,t\nx,a,1,real,0.25,t\n",
        encoding="utf-8",
    )
    match_set = Path("/Users/skylar/workspace/portfolio/docs/evals/match-set.yaml")
    if not match_set.exists():
        pytest.skip("the match set lives in the docs repo")
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / f"{V4_RUN}.meta.json").write_text(
        json.dumps({"run_id": V4_RUN, "samples": 78, "run_cost": {"real": 0.78, "credit": 7.8}}),
        encoding="utf-8",
    )

    def args(total_real: str) -> list[str]:
        return [
            "--set", str(match_set), "--out", str(runs), "--ledger", str(ledger),
            "--pass-cap", "0.15", "--total-cap", total_real, "--credit-pass-cap", "3",
            "--credit-total-cap", "60", "--dry-run",
        ]  # fmt: skip

    assert main([*args("1"), "--batch", str(batch)]) == 1  # does not fit: exit 1
    report = json.loads(capsys.readouterr().out)
    assert [o["option"] for o in report["options"]] == [
        "haiku-5-5-low", "haiku-5-5-off-max12", "luna-none",
    ]  # fmt: skip
    assert report["options"][0]["order"] == (
        "anthropic:claude-haiku-5-5:low -> anthropic:claude-haiku-5-5:off (max 4096),"
        " up to 8 requirements"
    )
    assert not any(o["uses_google"] for o in report["options"])
    assert report["jobs"] == 26 and report["spent"] == {"real": 0.25, "credit": 0.5}
    per_pass = report["options"][2]["worst_per_pass"]
    assert per_pass["real"] > 0 and per_pass["credit"] > 0  # Luna, then Haiku
    # later: 4 passes x 26 items at the stored run's measured $0.01 / $0.1 per item
    assert report["later_estimate"] == {"real": 1.04, "credit": 10.4}
    assert report["later_per_item"] == {"real": 0.01, "credit": 0.1}
    assert report["fits"] is False and report["problems"][0].startswith("real: spent 0.250000")
    assert main([*args("2"), "--batch", str(batch)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["fits"] is True and report["problems"] == []
    assert report["batch_worst"]["real"] == pytest.approx(per_pass["real"], abs=1e-6)
    # an estimate given by hand replaces the measured one
    assert main([*args("1"), "--batch", str(batch), "--later-per-item", "real=0,credit=0"]) == 0
    capsys.readouterr()
    # a single option, as the real run takes it
    google = "check=google:gemini-3.5-flash-lite:low"
    assert main([*args("2"), "--step", "check", "--model", google]) == 0
    single = json.loads(capsys.readouterr().out)
    assert single["options"][0]["uses_google"] is True


def test_budget_check_warns_on_a_pass_and_stops_on_an_item_over_the_pass_cap() -> None:
    option = option_for("read")
    item = step_worst(option)["credit"]

    def check(pass_cap: float) -> dict[str, Any]:
        return budget_check(
            [option],
            26,
            spent=dict.fromkeys(POOLS, 0.0),
            caps={"real": (1.0, 1.0), "credit": (pass_cap, 6.0)},
            future=dict.fromkeys(POOLS, 0.0),
        )

    report = check(item * 2)
    assert report["fits"] is True and "worst pass" in report["warnings"][0]
    report = check(item / 2)
    assert report["fits"] is False and "worst item" in report["problems"][0]


def test_later_estimate_falls_back_to_the_worst_case(tmp_path: Path) -> None:
    worst = {"real": 0.5, "credit": 0.25}
    assert later_per_item(None, tmp_path / "none.meta.json", lambda: worst)[0] == worst
    given, basis = later_per_item("real=0.1, credit=0.2", tmp_path / "x", lambda: worst)
    assert given == {"real": 0.1, "credit": 0.2} and basis == "the given estimate per item"
    with pytest.raises(ValueError, match="expected real"):
        later_per_item("gold=1", tmp_path / "x", lambda: worst)


def test_batch_lines_are_checked() -> None:
    with pytest.raises(ValueError, match="batch line 1"):
        batch_options("--step read --model check=" + HAIKU_OFF, DEFAULTS)
    with pytest.raises(ValueError, match="no options"):
        batch_options("# nothing\n", DEFAULTS)


def test_ledger_totals_add_each_pool(tmp_path: Path) -> None:
    path = tmp_path / "l.csv"
    assert ledger_totals(path) == {"real": 0.0, "credit": 0.0}
    path.write_text(
        "run_id,question_id,repeat,pool,cost_usd,at\na,b,1,real,0.1,t\na,b,1,real,x,t\n"
        "a,b,1,credit,0.2,t\n",
        encoding="utf-8",
    )
    assert ledger_totals(path) == {"real": pytest.approx(0.1), "credit": pytest.approx(0.2)}


# --- contract 7 -----------------------------------------------------------------------------


def test_source_rows_need_an_ok_row_per_job() -> None:
    rows = [stored_row("real-01"), {**stored_row("real-02"), "status": "error"}]
    with pytest.raises(SourceError, match="real-02"):
        source_rows(rows, JOBS, 1)
    assert source_rows([stored_row(j.id, 2) for j in JOBS] + [stored_row("real-01")], JOBS[:1], 1)


def test_needed_urls_are_every_retrieved_section_except_for_read() -> None:
    assert needed_urls(SOURCES, "read") == set()
    assert needed_urls(SOURCES, "match") == {PY, SQL, LAMBDA}


async def test_preflight_stops_on_changed_sources_or_missing_sections() -> None:
    async def fetch(urls: set[str]) -> dict[str, Any]:
        return {u: SECTIONS[u] for u in urls if u != LAMBDA}

    calls: list[Any] = []

    def unchanged(since: datetime) -> list[str]:
        calls.append(since)
        return []

    with pytest.raises(SourceError, match="changed"):
        await preflight(V4_RUN, {PY}, fetch, changes=lambda since: ["abc content"])
    with pytest.raises(SourceError, match=r"1 sections not in the database: .*lambda"):
        await preflight(V4_RUN, {PY, LAMBDA}, fetch, changes=unchanged)
    sections, checked = await preflight(
        V4_RUN, {PY, SQL}, fetch, changes=unchanged, content=lambda: "sha256:abc"
    )
    assert set(sections) == {PY, SQL}
    assert checked == {"sources_changed": [], "sections_checked": 2, "content_hash": "sha256:abc"}
    assert calls[0] == datetime(2026, 11, 19, 3, 4, 33, tzinfo=UTC)


def test_main_stops_before_any_call_when_the_stored_run_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    match_set = Path("/Users/skylar/workspace/portfolio/docs/evals/match-set.yaml")
    if not match_set.exists():
        pytest.skip("the match set lives in the docs repo")
    code = main(
        [
            "--set", str(match_set), "--out", str(tmp_path), "--ledger", str(tmp_path / "l.csv"),
            "--pass-cap", "1", "--total-cap", "1", "--credit-pass-cap", "1",
            "--credit-total-cap", "1", "--step", "read",
        ]
    )  # fmt: skip
    assert code == 2 and "cannot use the stored run" in capsys.readouterr().err
    assert not (tmp_path / "l.csv").exists()


def test_default_max_requirements_is_the_graphs() -> None:
    assert option_for("read").max_requirements == MAX_REQUIREMENTS == 8
    assert _pools("claude-haiku-5-5", 1.0) == {"real": 0.0, "credit": 1.0}
