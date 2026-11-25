# ruff: noqa: RUF001 - Chinese patterns use full-width punctuation
"""Code backstop for general answers (step 10, 10u): `answer_problem`.

The general check is a small model that looks at whether each sentence is supported, so it
passes answers whose sentences are all true but that a visitor should never see:

- "self_edit": the answer model rewrote itself mid-reply, "... [6].\\n\\nWait, the last
  sentence needs checking: ... Corrected answer below.\\n\\nI have merged ..."
  (`oc-numbers`, 20261125T031802-v12-default pass 3).
- "meta": the answer talks about the retrieved sections instead of the site, "The sections
  don't say why I avoided a model judge beyond cost." (`rr-no-api-grader`,
  20261125T031301-v12-default run 6). The prompt forbids it; v2-v12 runs have it in most
  passes of the "not on the site" questions.

A hit counts as a failed check (see `graph.check`): the answer is written once more, and if
that one has a hit too it stays, with the hit sentences marked like unsupported ones.

Blank lines are not a rule: real answers use them ("My project pages don't give ...\\n\\n
The closest figures ..."). The bare word "section" is not either: OnCallOps pages have
sections that report for themselves. A hit is only counted when the matched words are not
in the retrieved sections, so a blog post that quotes such a phrase can still be cited.
"""

import re
from collections.abc import Sequence

# --- self_edit: the model correcting its own reply --------------------------------------

# At the start of a sentence or line: an interjection a reply only uses to correct itself.
_SELF_EDIT_START = re.compile(
    r"(?:^|(?<=[.!?。！？\]])\s+|\n)\s*"
    r"(?:wait|hmm+|oops|on second thought|scratch that|correction|let me (?:re-?|double-)?check"
    r"|let me (?:correct|fix|rewrite|redo))\b",
    re.IGNORECASE,
)
# Anywhere: the reply talking about its own draft.
_SELF_EDIT_ANY = re.compile(
    # Not "version" or "draft": RoleRadar's resume pages talk about a rewritten version.
    r"\b(?:corrected|revised|rewritten|fixed|updated) (?:answer|reply|response)\b"
    r"|\b(?:needs|need) (?:re-?|double-)?checking\b"
    r"|\b(?:the|my) (?:last|previous|above|first|second|third|earlier) (?:sentence|answer|draft"
    r"|reply|response) (?:is|was|needs|should)\b",
    re.IGNORECASE,
)
_SELF_EDIT_ZH = re.compile(r"等等[，,]|更正[:：]|修正后的(?:回答|答案)|上一句(?:不对|有误|需要)")

# --- meta: the reply talking about the retrieved sections -------------------------------

# Verbs a reply uses when it reports what its sources say.
_TELL = (
    r"(?:say|says|said|state|states|stated|mention|mentions|describe|describes|note|notes"
    r"|give|gives|show|shows|cover|covers|list|lists|include|includes|tell|tells|explain"
    r"|explains|provide|provides|contain|contains|name|names|break|breaks|specify"
    r"|specifies|detail|details|confirm|confirms|answer|answers|address|addresses"
    r"|go into|goes into|define|defines|spell out|spells out|discuss|discusses|clarify"
    r"|clarifies|outline|outlines|quantify|quantifies|touch on|touches on|elaborate on"
    r"|elaborates on)"
)
# Not "report", "indicate", "offer" or "reference": OnCallOps pages have sections that do
# those things themselves ("each section reports its own status", v11 oc-failure pass 3).
_META_EN = re.compile(
    # "Section 6 says", "section 6 gives", "Section 1 and Section 3 describe"
    r"(?<![-\w])sections? \d+\b"
    # "The sections don't say", "the provided sections do not state", "The sections I have
    # don't give", "What the sections do say", "the cloud-hosting and MCP section notes"
    r"|(?<![-\w])sections?(?: (?:I have|I was given|on (?:this|the|my) site|here|above"
    r"|below|provided))?(?: (?:also|only))?"
    r"(?: (?:do|does|did)(?: not|n['’]t)?| (?:don|doesn|didn)['’]t| never| not)?"
    rf" {_TELL}\b"
    # "The provided text on my site does not state ..." (v2, Gemini answering)
    r"|\bprovided (?:text|context|passages?|excerpts?|snippets?)\b"
    r"|\b(?:the|these|those) (?:passages?|excerpts?|snippets?)\b",
    re.IGNORECASE,
)
_META_ZH = re.compile(
    r"(?:这些|上述|以上|提供的|给出的|所给的?|检索到的)(?:段落|小节|章节|资料|材料)"
    r"|第\s*\d+\s*(?:段|节|个段落|个小节)|段落\s*\d+"
)

# A sentence: from after the previous end mark (or line break) to its own end mark, with
# the citations that follow it.
_CITES = r"(?:\s*\[[\d,\s]+\])*"
# Chinese end marks need no space after them; the citations after one stay with it.
_SENTENCE_START = re.compile(rf"[。！？]+{_CITES}\s*|(?<=[.!?])\s+|(?<=[.!?]\])\s+|\n+")
# Not the dot in "gpt-5.6" or "roleradar.org": an English end mark is followed by space.
_SENTENCE_END = re.compile(rf"[。！？]+{_CITES}|[.!?]+{_CITES}(?=\s|$)|\n")


def _sentence_at(text: str, start: int, end: int) -> str:
    """The sentence of `text` that holds `text[start:end]`."""
    begin = 0
    for match in _SENTENCE_START.finditer(text, 0, start):
        begin = match.end()
    stop = _SENTENCE_END.search(text, end)
    finish = stop.end() if stop else len(text)
    return text[begin:finish].strip()


def answer_problem(answer: str, section_texts: Sequence[str]) -> tuple[str, list[str]] | None:
    """(kind, sentences) if a general answer must not be shown as it is, else None.

    `kind` is "self_edit" or "meta" (the first one found, in that order); `sentences` are
    the sentences that hold a hit, each a substring of `answer`, for marking.
    """
    source = " ".join(section_texts).casefold()
    for kind, patterns in (
        ("self_edit", (_SELF_EDIT_START, _SELF_EDIT_ANY, _SELF_EDIT_ZH)),
        ("meta", (_META_EN, _META_ZH)),
    ):
        sentences: list[str] = []
        for pattern in patterns:
            for match in pattern.finditer(answer):
                words = match.group(0).strip()
                if words.casefold() in source:
                    continue
                start = match.start() + match.group(0).index(words)  # past the space before it
                sentence = _sentence_at(answer, start, match.end())
                if sentence and sentence not in sentences:
                    sentences.append(sentence)
        if sentences:
            return kind, sentences
    return None
