"""Compare three ways of letting the answer check's Gemini model reason (step 4d2).

Each distinct labelled answer is checked once per arm and pass, with the same sections,
question and output cap the product uses. A labelled sentence counts as judged
unsupported when a quote kept by `verify_unsupported` overlaps it in the answer.

Arms:
- reasoning_field: the product's `check_messages` and `CheckOut` (reasoning written
  before the verdict), thinking "minimal" (what `gemini_thinking` gives the product).
- thinking_medium: the product prompt without the reasoning item and without "in this
  order", an output schema without `reasoning`, thinking "medium".
- no_reasoning: same prompt and schema as thinking_medium, thinking "minimal".

Usage (in site/api, caps are passed in, never written here):
    uv run --no-default-groups --group eval python -m evals.cot --labels ... \
        --source-run ... --out ... --ledger ... --pass-cap N --total-cap N \
        --credit-pass-cap N --credit-total-cap N [--repeats 3] [--summary-out ...] [--probe]
"""

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from app.qa import prompts
from app.qa.models import CALL_TIMEOUT_S, MAX_OUTPUT_TOKENS, gemini_thinking
from app.qa.output import verify_unsupported
from app.rag.types import Section
from evals.summary import write_summary
from evals.types import CapReached, MeterLike, UnknownModel

LABELS = ("supported", "unsupported")


# --- the prompt and schema for the two arms without a reasoning field -----------------


def _without_reasoning(system: str) -> str:
    """The product's check prompt with the reasoning item and "in this order" removed."""
    text, ordered = re.subn(r"Fill in, in this order:", "Fill in:", system)
    text, items = re.subn(r"- reasoning: [^\n]*\n", "", text)
    if ordered != 1 or items != 1 or "reasoning" in text or "in this order" in text:
        raise RuntimeError("the product check prompt changed; update evals/cot.py")
    return text


NO_REASONING_SYSTEM = _without_reasoning(prompts._CHECK_SYSTEM)


class VerdictOut(BaseModel):
    """`CheckOut` without `reasoning`: same field descriptions."""

    unsupported: list[str] = Field(
        description=prompts.CheckOut.model_fields["unsupported"].description
    )
    passed: bool = Field(description=prompts.CheckOut.model_fields["passed"].description)


def no_reasoning_messages(
    question: str, answer: str, sections: Sequence[Section]
) -> list[BaseMessage]:
    marker = prompts._marker()
    return [
        SystemMessage(NO_REASONING_SYSTEM.format(data_rule=prompts._data_rule(marker))),
        prompts._check_human(question, answer, sections, marker),
    ]


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    thinking_level: str
    schema: type[BaseModel]
    messages: Callable[[str, str, Sequence[Section]], list[BaseMessage]]


def arms(check_model: str) -> list[Arm]:
    product_level = str(gemini_thinking(check_model).get("thinking_level", "none"))
    return [
        Arm(
            "reasoning_field",
            "Writes reasoning, then verdict",
            product_level,
            prompts.CheckOut,
            prompts.check_messages,
        ),
        Arm(
            "thinking_medium",
            "Model thinking (medium), verdict only",
            "medium",
            VerdictOut,
            no_reasoning_messages,
        ),
        Arm(
            "no_reasoning",
            "Verdict only",
            product_level,
            VerdictOut,
            no_reasoning_messages,
        ),
    ]


def make_chat(settings: Any, thinking_level: str) -> Any:
    """The product's Gemini check model (`build_providers`) with one thinking level."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = settings.gemini_api_key
    return ChatGoogleGenerativeAI(
        model=settings.check_model,
        google_api_key=key.get_secret_value() if key else "missing",
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=CALL_TIMEOUT_S,
        max_retries=1,
        thinking_level=thinking_level,
    )


# --- inputs ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Sentence:
    sid: str
    text: str
    label: str
    planted: bool


@dataclass(frozen=True)
class Segment:
    """One distinct answer, checked once per arm and pass."""

    id: str  # the first sid in it
    question_id: str
    question: str
    answer: str
    sections: tuple[Section, ...]
    sentences: tuple[Sentence, ...]


def _section_from_row(raw: dict[str, Any]) -> Section:
    page, _, section_id = raw["url"].partition("#")
    return Section(page=page, section_id=section_id, heading=raw["heading"], text=raw["text"])


def load_segments(labels_path: Path, source_run: Path) -> list[Segment]:
    """Group labelled sentences by answer; take question and sections from the source run.

    The labels carry sections as "[heading] text" for reading; the source run (the run the
    answers came from) has the page and section id the product prompt shows the model.
    """
    labels = [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runs = [
        json.loads(line)
        for line in source_run.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in labels:
        if row["label"] not in LABELS:
            raise ValueError(f"{row['sid']}: unknown label {row['label']!r}")
        if row["answer"].count(row["sentence"]) != 1:
            raise ValueError(f"{row['sid']}: sentence must appear exactly once in its answer")
        groups.setdefault(row["answer"], []).append(row)
    segments = []
    for answer, rows in groups.items():
        first = rows[0]
        wanted = [section["text"] for section in first["sections"]]
        matches = [
            run
            for run in runs
            if run["id"] == first["question_id"]
            and [f"[{s['heading']}] {s['text']}" for s in run["sections"]] == wanted
        ]
        if not matches:
            raise ValueError(f"{first['sid']}: no source-run row with the same sections")
        source = matches[0]
        segments.append(
            Segment(
                id=first["sid"],
                question_id=first["question_id"],
                question=source["turns"][-1],
                answer=answer,
                sections=tuple(_section_from_row(s) for s in source["sections"]),
                sentences=tuple(
                    Sentence(r["sid"], r["sentence"], r["label"], bool(r["planted"])) for r in rows
                ),
            )
        )
    return segments


# --- one call --------------------------------------------------------------------------


class ModelFailed(Exception):
    """The model call failed twice: stop the run (no provider switch here)."""


def _alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def judged_unsupported(sentence: str, answer: str, quotes: Sequence[str]) -> bool:
    """True if a verified quote overlaps `sentence` in `answer` (on more than punctuation)."""
    start = answer.index(sentence)
    end = start + len(sentence)
    for quote in quotes:
        at = answer.find(quote)
        while at != -1:
            lo, hi = max(start, at), min(end, at + len(quote))
            if lo < hi and _alnum(answer[lo:hi]):
                return True
            at = answer.find(quote, at + 1)
    return False


@dataclass
class CallResult:
    status: str  # "ok" | "invalid"
    error: str | None
    raw_unsupported: list[str]
    unsupported: list[str]
    passed: bool | None
    reasoning: str | None
    ms: int
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    finish_reason: str | None


async def judge(chat: Any, arm: Arm, segment: Segment, meter: MeterLike) -> CallResult:
    """One check call. Model errors raise; output that does not parse is "invalid"."""
    messages = arm.messages(segment.question, segment.answer, segment.sections)
    runnable = chat.with_structured_output(arm.schema, include_raw=True)
    started = time.perf_counter()
    async with asyncio.timeout(CALL_TIMEOUT_S):
        out = await runnable.ainvoke(messages, config={"callbacks": meter.callbacks()})
    ms = round((time.perf_counter() - started) * 1000)
    raw = out.get("raw")
    usage = getattr(raw, "usage_metadata", None) or {}
    details = usage.get("output_token_details") or {}
    metadata = getattr(raw, "response_metadata", None) or {}
    finish = metadata.get("finish_reason")
    parsed = out.get("parsed")
    error = out.get("parsing_error")
    common = {
        "ms": ms,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "thinking_tokens": details.get("reasoning", 0) if usage else None,
        "finish_reason": str(finish) if finish is not None else None,
    }
    if parsed is not None and not isinstance(parsed, arm.schema):
        try:
            parsed = arm.schema.model_validate(parsed)
        except ValueError as invalid:
            parsed, error = None, invalid
    if parsed is None:
        return CallResult(
            status="invalid",
            error=type(error).__name__ if error else "EmptyReply",
            raw_unsupported=[],
            unsupported=[],
            passed=None,
            reasoning=None,
            **common,
        )
    quotes = list(parsed.unsupported)  # type: ignore[attr-defined]
    return CallResult(
        status="ok",
        error=None,
        raw_unsupported=quotes,
        unsupported=verify_unsupported(quotes, segment.answer),
        passed=bool(parsed.passed),  # type: ignore[attr-defined]
        reasoning=getattr(parsed, "reasoning", None),
        **common,
    )


def predictions(segment: Segment, result: CallResult) -> dict[str, str]:
    """sid -> "unsupported" | "supported" | "invalid"."""
    if result.status != "ok":
        return {s.sid: "invalid" for s in segment.sentences}
    return {
        s.sid: (
            "unsupported"
            if judged_unsupported(s.text, segment.answer, result.unsupported)
            else "supported"
        )
        for s in segment.sentences
    }


def exact_match(segment: Segment, result: CallResult) -> dict[str, bool]:
    """The stricter reading (the sentence itself is one of the quotes), for the record."""
    return {s.sid: s.text.strip() in result.unsupported for s in segment.sentences}


# --- metrics ---------------------------------------------------------------------------


def pass_metrics(labels: dict[str, str], preds: dict[str, str]) -> dict[str, float | None]:
    """Accuracy over all sentences (invalid counts as wrong); precision and recall of
    "unsupported" (invalid is never a hit, so it lowers recall, not precision)."""
    total = len(labels)
    correct = sum(preds.get(sid) == label for sid, label in labels.items())
    flagged = [sid for sid in labels if preds.get(sid) == "unsupported"]
    actual = [sid for sid, label in labels.items() if label == "unsupported"]
    hits = sum(labels[sid] == "unsupported" for sid in flagged)
    return {
        "accuracy": correct / total if total else None,
        "unsupported_precision": hits / len(flagged) if flagged else None,
        "unsupported_recall": hits / len(actual) if actual else None,
    }


def spread(values: Sequence[float | None]) -> dict[str, float | None]:
    known = [v for v in values if v is not None]
    if not known:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(known), 6),
        "min": round(min(known), 6),
        "max": round(max(known), 6),
    }


@dataclass
class ArmTally:
    passes: dict[int, dict[str, str]] = field(default_factory=dict)  # repeat -> sid -> pred
    exact_differs: int = 0
    invalid: int = 0
    ms: list[int] = field(default_factory=list)
    cost: float = 0.0
    thinking_tokens: list[int] = field(default_factory=list)


def arm_summary(arm: Arm, tally: ArmTally, labels: dict[str, str]) -> dict[str, Any]:
    per_pass = [pass_metrics(labels, preds) for _, preds in sorted(tally.passes.items())]
    return {
        "name": arm.name,
        "label": arm.label,
        "thinking_level": arm.thinking_level,
        "accuracy": spread([m["accuracy"] for m in per_pass]),
        "unsupported_precision": spread([m["unsupported_precision"] for m in per_pass]),
        "unsupported_recall": spread([m["unsupported_recall"] for m in per_pass]),
        "invalid": tally.invalid,
        "ms_median": round(statistics.median(tally.ms)) if tally.ms else None,
        "cost_usd": round(tally.cost, 6),
    }


def build_summary(
    *,
    run_id: str,
    date: str,
    model: str,
    repeats: int,
    labels: dict[str, str],
    arm_list: Sequence[Arm],
    tallies: dict[str, ArmTally],
) -> dict[str, Any]:
    """The public `kind: "cot"` summary (convention C4): numbers only, no text."""
    return {
        "kind": "cot",
        "run_id": run_id,
        "date": date,
        "model": model,
        "sentences": len(labels),
        "repeats": repeats,
        "arms": [arm_summary(arm, tallies[arm.name], labels) for arm in arm_list],
    }


# --- the run ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    status: str
    reason: str | None
    summary: dict[str, Any] | None

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "complete" else 1


Judge = Callable[[Any, Arm, Segment, MeterLike], Awaitable[CallResult]]


async def compare(
    segments: Sequence[Segment],
    *,
    arm_list: Sequence[Arm],
    chats: dict[str, Any],
    meter: MeterLike,
    repeats: int,
    run_id: str,
    out_path: Path,
    model: str,
    date: str,
    summary_out: Path | None = None,
    call: Judge = judge,
) -> Outcome:
    """Passes outermost, then arms, then segments; no concurrency."""
    labels = {s.sid: s.label for seg in segments for s in seg.sentences}
    tallies = {arm.name: ArmTally() for arm in arm_list}
    status, reason = "complete", None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("a", encoding="utf-8") as output:
            for repeat in range(1, repeats + 1):
                for arm in arm_list:
                    meter.start_pass()
                    tally = tallies[arm.name]
                    preds_this_pass = tally.passes.setdefault(repeat, {})
                    for segment in segments:
                        key = f"{arm.name}:{segment.id}"
                        recorded = False
                        try:
                            result: CallResult | None = None
                            for attempt in range(2):
                                meter.check()
                                try:
                                    result = await call(chats[arm.name], arm, segment, meter)
                                    break
                                except (CapReached, UnknownModel):
                                    raise
                                except Exception as error:
                                    if attempt:
                                        raise ModelFailed(type(error).__name__) from error
                            if result is None:  # unreachable: the loop breaks or raises
                                raise ModelFailed("no result")
                            cost = meter.record(key, repeat)
                            recorded = True
                            preds = predictions(segment, result)
                            exact = exact_match(segment, result)
                            preds_this_pass.update(preds)
                            tally.exact_differs += sum(
                                exact[sid] != (pred == "unsupported") for sid, pred in preds.items()
                            )
                            tally.invalid += result.status == "invalid"
                            tally.ms.append(result.ms)
                            tally.cost += cost
                            if result.thinking_tokens is not None:
                                tally.thinking_tokens.append(result.thinking_tokens)
                            row = {
                                "run_id": run_id,
                                "arm": arm.name,
                                "repeat": repeat,
                                "segment": segment.id,
                                "question_id": segment.question_id,
                                "sids": [s.sid for s in segment.sentences],
                                **result.__dict__,
                                "predictions": preds,
                                "cost_usd": cost,
                            }
                            output.write(json.dumps(row, ensure_ascii=False) + "\n")
                            output.flush()
                            print(
                                f"{repeat} {arm.name} {segment.id} {result.status} "
                                f"{result.ms}ms think={result.thinking_tokens} {cost:.6f}",
                                flush=True,
                            )
                        finally:
                            if not recorded:
                                with suppress(CapReached, UnknownModel):
                                    meter.record(key, repeat)
    except (CapReached, UnknownModel, ModelFailed) as error:
        status, reason = "incomplete", f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001 - keep the rows written so far
        status, reason = "incomplete", type(error).__name__
    summary = build_summary(
        run_id=run_id,
        date=date,
        model=model,
        repeats=repeats,
        labels=labels,
        arm_list=arm_list,
        tallies=tallies,
    )
    meta = {
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "segments": len(segments),
        "summary": summary,
        "exact_vs_overlap_differs": {n: t.exact_differs for n, t in tallies.items()},
        "thinking_tokens_median": {
            n: statistics.median(t.thinking_tokens) if t.thinking_tokens else None
            for n, t in tallies.items()
        },
    }
    if status == "incomplete":
        print(f"cot incomplete: {reason}", file=sys.stderr)
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if status == "complete" and summary_out is not None:
        write_summary(summary, summary_out)
    return Outcome(status, reason, summary if status == "complete" else None)


# --- CLI -------------------------------------------------------------------------------


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite float > 0")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    for flag in ("pass-cap", "total-cap", "credit-pass-cap", "credit-total-cap"):
        parser.add_argument(f"--{flag}", type=_positive_float, required=True)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="one call per arm on the first segment, to check thinking settings; no summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        segments = load_segments(args.labels, args.source_run)
    except (OSError, ValueError, KeyError) as error:
        print(f"invalid labels or source run: {error}", file=sys.stderr)
        return 2
    from app.config import get_settings
    from evals.cost import Meter

    settings = get_settings()
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%dT%H%M%S}-{prompts.PROMPT_VERSION}-cot" + (
        "-probe" if args.probe else ""
    )
    caps = {
        "real": (args.pass_cap, args.total_cap),
        "credit": (args.credit_pass_cap, args.credit_total_cap),
    }
    arm_list = arms(settings.check_model)
    meter = Meter(caps, args.ledger, run_id)
    chats = {arm.name: make_chat(settings, arm.thinking_level) for arm in arm_list}
    if args.probe:
        segments, repeats, summary_out = segments[:1], 1, None
    else:
        repeats, summary_out = args.repeats, args.summary_out
    try:
        outcome = asyncio.run(
            compare(
                segments,
                arm_list=arm_list,
                chats=chats,
                meter=meter,
                repeats=repeats,
                run_id=run_id,
                out_path=args.out / f"{run_id}.jsonl",
                model=settings.check_model,
                date=started.date().isoformat(),
                summary_out=summary_out,
            )
        )
    except Exception as error:  # noqa: BLE001 - never print config or prompts
        print(f"cot setup failed: {type(error).__name__}", file=sys.stderr)
        return 1
    print(f"run {run_id}: {outcome.status}")
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
