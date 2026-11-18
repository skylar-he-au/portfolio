"""Offline tests for the CoT comparison (evals/cot.py): scripted models, no API calls."""

import asyncio
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import SecretStr, ValidationError

from app.qa import prompts
from app.rag.types import Section
from evals import cot
from evals.cost import Meter
from evals.types import CapReached

CAPS = {"real": (1.0, 5.0), "credit": (1.0, 5.0)}
MODEL = "gemini-3.5-flash-lite"

ANSWER = (
    "RoleRadar ranks jobs by keyword overlap [1]. It was built in 2024 [1]. "
    "Tests run on every push [2]."
)
S1 = "RoleRadar ranks jobs by keyword overlap [1]."
S2 = "It was built in 2024 [1]."
S3 = "Tests run on every push [2]."


def _sections() -> list[dict[str, Any]]:
    return [
        {"url": "/projects/roleradar#matching", "heading": "Matching", "text": "Keyword overlap."},
        {"url": "/projects/roleradar#tests", "heading": "Tests", "text": "CI on every push."},
    ]


def _segment() -> cot.Segment:
    return cot.Segment(
        id="s001",
        question_id="rr-match",
        question="How does RoleRadar match jobs?",
        answer=ANSWER,
        sections=tuple(cot._section_from_row(s) for s in _sections()),
        sentences=(
            cot.Sentence("s001", S1, "supported", False),
            cot.Sentence("s002", S2, "unsupported", True),
            cot.Sentence("s003", S3, "supported", False),
        ),
    )


class FakeRunnable:
    def __init__(self, chat: "FakeChat", schema: type, include_raw: bool) -> None:
        self.chat, self.schema, self.include_raw = chat, schema, include_raw

    async def ainvoke(self, messages: list[BaseMessage], config: dict[str, Any]) -> Any:
        self.chat.calls.append(
            {"schema": self.schema, "include_raw": self.include_raw, "messages": messages}
        )
        reply = self.chat.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeChat:
    def __init__(self, script: list[Any], thinking_level: str = "minimal") -> None:
        self.script = script
        self.thinking_level = thinking_level
        self.calls: list[dict[str, Any]] = []

    def with_structured_output(self, schema: type, include_raw: bool = False) -> FakeRunnable:
        return FakeRunnable(self, schema, include_raw)


def _reply(parsed: Any, thinking: int = 0, error: Exception | None = None) -> dict[str, Any]:
    usage = {"input_tokens": 100, "output_tokens": 20 + thinking, "total_tokens": 120 + thinking}
    if thinking:
        usage["output_token_details"] = {"reasoning": thinking}
    raw = AIMessage(content="{}", usage_metadata=usage, response_metadata={"finish_reason": "STOP"})
    return {"raw": raw, "parsed": parsed, "parsing_error": error}


# --- the three arms --------------------------------------------------------------------


def test_arms_use_their_prompt_schema_and_thinking_level() -> None:
    by_name = {arm.name: arm for arm in cot.arms(MODEL)}
    assert list(by_name) == ["reasoning_field", "thinking_medium", "no_reasoning"]
    a, b, c = by_name["reasoning_field"], by_name["thinking_medium"], by_name["no_reasoning"]
    assert (a.schema, a.messages, a.thinking_level) == (
        prompts.CheckOut,
        prompts.check_messages,
        "minimal",
    )
    assert (b.schema, b.thinking_level) == (cot.VerdictOut, "medium")
    assert (c.schema, c.thinking_level) == (cot.VerdictOut, "minimal")
    assert b.messages is c.messages is cot.no_reasoning_messages


def test_b_and_c_prompt_and_schema_have_no_reasoning() -> None:
    assert "reasoning" not in cot.VerdictOut.model_fields
    assert set(cot.VerdictOut.model_fields) == {"unsupported", "passed"}
    sections = [cot._section_from_row(s) for s in _sections()]
    system, human = cot.no_reasoning_messages("Q?", ANSWER, sections)
    assert "reasoning" not in system.content.lower()
    assert "in this order" not in system.content
    # Everything else is the product prompt, word for word.
    product = prompts._CHECK_SYSTEM.replace("Fill in, in this order:", "Fill in:")
    assert cot.NO_REASONING_SYSTEM == "\n".join(
        line for line in product.split("\n") if not line.startswith("- reasoning:")
    )
    assert "<answer-" in human.content and ANSWER in human.content


def test_changed_product_prompt_is_caught() -> None:
    with pytest.raises(RuntimeError):
        cot._without_reasoning("Fill in:\n- unsupported: x\n")


def test_make_chat_sets_thinking_level_and_product_limits() -> None:
    settings = SimpleNamespace(check_model=MODEL, gemini_api_key=SecretStr("test-key"))
    medium = cot.make_chat(settings, "medium")
    minimal = cot.make_chat(settings, "minimal")
    assert medium.thinking_level == "medium"
    assert minimal.thinking_level == "minimal"
    assert medium.max_output_tokens == minimal.max_output_tokens == 1024
    assert medium.model.removeprefix("models/") == MODEL


def test_judge_passes_each_arm_its_messages_schema_and_raw() -> None:
    segment = _segment()
    meter = SimpleNamespace(callbacks=lambda: [])
    for arm in cot.arms(MODEL):
        if arm.schema is prompts.CheckOut:
            parsed = prompts.CheckOut(reasoning="r", unsupported=[S2], passed=False)
        else:
            parsed = cot.VerdictOut(unsupported=[S2], passed=False)
        chat = FakeChat([_reply(parsed)])
        result = asyncio.run(cot.judge(chat, arm, segment, meter))
        [call] = chat.calls
        assert call["schema"] is arm.schema and call["include_raw"] is True
        system = call["messages"][0].content
        if arm.name == "reasoning_field":
            assert "- reasoning:" in system and "in this order" in system
        else:
            assert "reasoning" not in system.lower()
        assert result.status == "ok" and result.unsupported == [S2]


# --- invalid, overlap and retries -------------------------------------------------------


def test_unparsed_output_is_invalid_not_supported() -> None:
    arm = cot.arms(MODEL)[2]
    chat = FakeChat([_reply(None, error=ValueError("bad json"))])
    result = asyncio.run(cot.judge(chat, arm, _segment(), SimpleNamespace(callbacks=lambda: [])))
    assert result.status == "invalid" and result.error == "ValueError"
    assert set(cot.predictions(_segment(), result).values()) == {"invalid"}


def test_quotes_are_verified_then_matched_by_overlap() -> None:
    # Without "[1]" and with a final period that is not in the answer: verify_unsupported
    # trims it, and the kept quote overlaps S2.
    assert cot.judged_unsupported(S2, ANSWER, ["It was built in 2024"])
    assert not cot.judged_unsupported(S1, ANSWER, ["It was built in 2024"])
    # Touching only punctuation or whitespace does not count.
    assert not cot.judged_unsupported(S3, ANSWER, ["[1]. "])
    segment = _segment()
    result = cot.CallResult("ok", None, [], ["It was built in 2024"], False, None, 1, 0, 0, 0, None)
    assert cot.predictions(segment, result) == {
        "s001": "supported",
        "s002": "unsupported",
        "s003": "supported",
    }
    assert cot.exact_match(segment, result)["s002"] is False


def test_hand_worked_metrics() -> None:
    labels = {
        "a": "unsupported",
        "b": "supported",
        "c": "unsupported",
        "d": "unsupported",
        "e": "supported",
    }
    preds = {
        "a": "unsupported",  # hit
        "b": "unsupported",  # false alarm
        "c": "invalid",  # counts as wrong and as a miss
        "d": "supported",  # miss
        "e": "supported",  # right
    }
    metrics = cot.pass_metrics(labels, preds)
    assert metrics["accuracy"] == pytest.approx(2 / 5)
    assert metrics["unsupported_precision"] == pytest.approx(1 / 2)
    assert metrics["unsupported_recall"] == pytest.approx(1 / 3)
    assert cot.pass_metrics({"a": "supported"}, {"a": "supported"})["unsupported_precision"] is None
    assert cot.spread([0.5, None, 1.0]) == {"mean": 0.75, "min": 0.5, "max": 1.0}


def _compare(tmp_path: Path, chats: dict[str, Any], **kwargs: Any) -> cot.Outcome:
    meter = Meter(CAPS, tmp_path / "spend.csv", "run-1")
    return asyncio.run(
        cot.compare(
            [_segment()],
            arm_list=cot.arms(MODEL),
            chats=chats,
            meter=meter,
            repeats=kwargs.pop("repeats", 2),
            run_id="run-1",
            out_path=tmp_path / "runs" / "run-1.jsonl",
            model=MODEL,
            date="2026-11-18",
            summary_out=tmp_path / "content",
            **kwargs,
        )
    )


def test_compare_writes_rows_ledger_and_public_summary(tmp_path: Path) -> None:
    verdict = cot.VerdictOut(unsupported=[S2], passed=False)
    chats = {
        "reasoning_field": FakeChat(
            [
                _reply(prompts.CheckOut(reasoning="r", unsupported=[], passed=True)),
                _reply(None, error=ValueError("bad")),
            ]
        ),
        "thinking_medium": FakeChat([_reply(verdict, thinking=50), _reply(verdict, thinking=40)]),
        "no_reasoning": FakeChat([RuntimeError("503"), _reply(verdict), _reply(verdict)]),
    }
    outcome = _compare(tmp_path, chats)
    assert outcome.status == "complete"
    rows = [json.loads(line) for line in (tmp_path / "runs" / "run-1.jsonl").open()]
    assert [(r["repeat"], r["arm"]) for r in rows] == [
        (1, "reasoning_field"),
        (1, "thinking_medium"),
        (1, "no_reasoning"),
        (2, "reasoning_field"),
        (2, "thinking_medium"),
        (2, "no_reasoning"),
    ]
    ledger = list(csv.DictReader((tmp_path / "spend.csv").open()))
    assert {r["question_id"] for r in ledger} == {
        "reasoning_field:s001",
        "thinking_medium:s001",
        "no_reasoning:s001",
    }
    summary = json.loads((tmp_path / "content" / "run-1.json").read_text())
    assert summary["kind"] == "cot" and summary["sentences"] == 3 and summary["repeats"] == 2
    arms = {a["name"]: a for a in summary["arms"]}
    # Pass 1: everything judged supported -> 2/3; pass 2: invalid -> 0/3.
    assert arms["reasoning_field"]["accuracy"] == {
        "mean": pytest.approx(1 / 3),
        "min": 0.0,
        "max": pytest.approx(2 / 3),
    }
    assert arms["reasoning_field"]["invalid"] == 1
    assert arms["reasoning_field"]["unsupported_recall"]["mean"] == 0.0
    assert arms["reasoning_field"]["unsupported_precision"]["mean"] is None
    assert arms["thinking_medium"]["accuracy"]["mean"] == 1.0
    assert arms["no_reasoning"]["invalid"] == 0  # a model error is retried, not "invalid"
    # No answer, question or section text in the public summary.
    text = json.dumps(summary)
    assert "RoleRadar" not in text and "2024" not in text
    meta = json.loads((tmp_path / "runs" / "run-1.meta.json").read_text())
    assert meta["thinking_tokens_median"]["thinking_medium"] == 45


def test_second_model_error_stops_the_run_without_summary(tmp_path: Path) -> None:
    chats = {
        "reasoning_field": FakeChat([RuntimeError("503"), RuntimeError("503")]),
        "thinking_medium": FakeChat([]),
        "no_reasoning": FakeChat([]),
    }
    outcome = _compare(tmp_path, chats)
    assert outcome.status == "incomplete" and outcome.exit_code == 1
    assert "ModelFailed" in (outcome.reason or "")
    assert not (tmp_path / "content").exists()
    ledger = list(csv.DictReader((tmp_path / "spend.csv").open()))
    assert {r["question_id"] for r in ledger} == {"reasoning_field:s001"}  # finally records


def test_cap_stops_before_the_next_call(tmp_path: Path) -> None:
    (tmp_path / "spend.csv").write_text(
        "run_id,question_id,repeat,pool,cost_usd,at\nold,q,1,real,5.0,2026-01-01\n"
    )
    chats = {name: FakeChat([]) for name in ("reasoning_field", "thinking_medium", "no_reasoning")}
    outcome = _compare(tmp_path, chats)
    assert outcome.status == "incomplete" and CapReached.__name__ in (outcome.reason or "")
    assert all(not chat.calls for chat in chats.values())


# --- inputs ----------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def test_load_segments_groups_by_answer_and_takes_sections_from_source(tmp_path: Path) -> None:
    label_sections = [
        {"n": i, "text": f"[{s['heading']}] {s['text']}"} for i, s in enumerate(_sections(), 1)
    ]
    planted = ANSWER.replace("keyword overlap", "a neural ranker")
    labels = [
        {"sid": "s001", "question_id": "rr-match", "answer": ANSWER, "sentence": S1,
         "sections": label_sections, "label": "supported", "planted": False, "note": ""},
        {"sid": "s002", "question_id": "rr-match", "answer": ANSWER, "sentence": S2,
         "sections": label_sections, "label": "unsupported", "planted": False, "note": ""},
        {"sid": "s003", "question_id": "rr-match", "answer": planted,
         "sentence": "RoleRadar ranks jobs by a neural ranker [1].", "sections": label_sections,
         "label": "unsupported", "planted": True, "note": ""},
    ]  # fmt: skip
    source = [
        {"id": "rr-match", "turns": ["First?", "How does it match?"], "sections": _sections()},
    ]
    segments = cot.load_segments(
        _write_jsonl(tmp_path / "labels.jsonl", labels),
        _write_jsonl(tmp_path / "run.jsonl", source),
    )
    assert [s.id for s in segments] == ["s001", "s003"]
    assert [len(s.sentences) for s in segments] == [2, 1]
    first = segments[0]
    assert first.question == "How does it match?"
    assert first.sections[0] == Section(
        page="/projects/roleradar", section_id="matching", heading="Matching",
        text="Keyword overlap.",
    )  # fmt: skip


def test_load_segments_rejects_a_sentence_not_in_its_answer(tmp_path: Path) -> None:
    labels = [
        {"sid": "s001", "question_id": "q", "answer": ANSWER, "sentence": "Not there.",
         "sections": [], "label": "supported", "planted": False, "note": ""},
    ]  # fmt: skip
    with pytest.raises(ValueError, match="exactly once"):
        cot.load_segments(
            _write_jsonl(tmp_path / "labels.jsonl", labels),
            _write_jsonl(tmp_path / "run.jsonl", []),
        )


def test_caps_are_required() -> None:
    with pytest.raises(SystemExit):
        cot._parser().parse_args(["--labels", "l", "--source-run", "r", "--out", "o",
                                  "--ledger", "s"])  # fmt: skip
    args = cot._parser().parse_args(
        ["--labels", "l", "--source-run", "r", "--out", "o", "--ledger", "s",
         "--pass-cap", "1", "--total-cap", "2", "--credit-pass-cap", "1",
         "--credit-total-cap", "2"]
    )  # fmt: skip
    assert args.repeats == 3 and args.probe is False


def test_verdict_schema_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        cot.VerdictOut.model_validate({"passed": True})


def test_side_run_options_never_publish() -> None:
    base = ["--labels", "l", "--source-run", "r", "--out", "o", "--ledger", "s",
            "--pass-cap", "1", "--total-cap", "2", "--credit-pass-cap", "1",
            "--credit-total-cap", "2"]  # fmt: skip
    with pytest.raises(SystemExit):
        cot.main([*base, "--max-output-tokens", "2048", "--summary-out", "x"])
    with pytest.raises(SystemExit):
        cot.main([*base, "--arms", "thinking_medium", "--summary-out", "x"])
    settings = SimpleNamespace(check_model=MODEL, gemini_api_key=SecretStr("test-key"))
    assert cot.make_chat(settings, "medium", 2048).max_output_tokens == 2048
