"""Code backstop for spotting personal questions about Skylar.

The rewrite model flags personal questions; this keyword list catches the ones it misses.
Either one sends the question down the private path (answer from /about only, shown only
after the check passes), so a miss here costs nothing and a false hit only means a
cautious answer from the About page.

Words that also mean something in the projects (OnCallOps is an on-call dashboard:
"availability", "contact", "phone", "location", "email" all appear there) only count
when they are about "you"/"your".
"""

import re
from collections.abc import Sequence

KEYWORDS_VERSION = "v1"  # bump when the lists change, like PROMPT_VERSION

_EN_PATTERNS = [
    # work rights
    r"\bvisas?\b",
    r"\bwork(?:ing)?[- ]rights?\b",
    r"\bright to work\b",
    r"\b(?:allowed|eligible|able) to work\b",
    r"\bsponsor(?:ship|ed)?\b",
    r"\bcitizen(?:ship)?\b",
    r"\bpermanent residen(?:t|ce|cy)\b",
    r"\bnationality\b",
    # pay
    r"\bsalary\b",
    r"\bwages?\b",
    r"\bhourly rate\b",
    r"\b(?:your|expected) (?:pay|rate|compensation)\b",
    r"\bpay expectations?\b",
    # kind of work and when
    r"\bfull[- ]?time\b",
    r"\bpart[- ]?time\b",
    r"\binternships?\b",
    r"\bgraduate (?:program|programme|role|job)s?\b",
    r"\bwhen (?:can|could) you start\b",
    r"\b(?:are|is) (?:you|skylar) (?:available|free|looking|open to)\b",
    r"\byour availability\b",
    r"\bnotice period\b",
    r"\b(?:you|skylar) graduate\b",
    r"\bhire (?:you|skylar|him|her|them)\b",
    # where
    r"\brelocat(?:e|ion|ing)\b",
    r"\bwhere (?:do|are|is) (?:you|skylar)\b",
    r"\b(?:you|skylar) (?:live|based|located|from)\b",
    r"\byour (?:location|city|address|hometown)\b",
    # contact
    r"\b(?:contact|reach|email|call|phone) (?:you|skylar)\b",
    r"\byour (?:e-?mail|phone|number|linkedin|contact)\b",
    # person
    r"\bhow old\b",
    r"\byour age\b",
    r"\byour (?:university|degree|studies|background|family|hobbies)\b",
    r"\bwho (?:are you|is skylar)\b",
    r"\babout yourself\b",
    r"\btell me about you\b",
]

# Chinese has no word boundaries; plain substrings, anchored on 你 where the bare word also
# appears in project talk (身份验证, 电话告警, 数据有多大).
_ZH_TERMS = [
    "签证",
    "工签",
    "工作权",
    "工作许可",
    "公民",
    "永居",
    "绿卡",
    "担保",
    "工资",
    "薪资",
    "薪水",
    "待遇",
    "时薪",
    "全职",
    "兼职",
    "实习",
    "毕业",
    "到岗",
    "入职",
    "你在哪个城市",
    "你人在哪",
    "你住",
    "搬到",
    "联系你",
    "联系方式",
    "你的邮箱",
    "你的电话",
    "年龄",
    "你多大",
    "国籍",
    "你是谁",
    "自我介绍",
]

_EN_RE = re.compile("|".join(_EN_PATTERNS), re.IGNORECASE)


def looks_private(question: str) -> bool:
    """True if the question matches a personal-topic keyword (English or Chinese)."""
    if _EN_RE.search(question):
        return True
    return any(term in question for term in _ZH_TERMS)


# --- unstated conclusions in a private draft (step 10, 10s) ----------------------------
#
# The private check is a small model and once passed "Yes, I can work in Australia" as the
# answer to "Can you work full-time in Australia?": every word is on the About page, but
# the "Yes" answers the full-time part, which the page never states. These rules catch,
# by code, a draft that settles work rights, visas or hours where the sections do not.
# A hit sends the fixed sentence (same as a failed check); a false hit only costs a
# cautious answer, which the eval counts as a format miss, never as a safety one.

# Conditions a visitor may ask about that the About page does not settle.
_CONDITION_EN = re.compile(
    r"\b(?:full|part)[- ]?time\b|\bhours?\b|\bvisas?\b|\bsponsor\w*|\bcitizen\w*"
    r"|\bpermanent(?:ly)? resid\w*|\bresidency\b|\bunrestricted\b|\bunlimited\b"
    r"|\brestrict\w*|\bsalary\b|\bwages?\b",
    re.IGNORECASE,
)
_CONDITION_ZH = (
    "全职",
    "兼职",
    "小时",
    "签证",
    "担保",
    "公民",
    "永居",
    "绿卡",
    "限制",
    "工资",
    "薪",
)

# Asking whether something is allowed or possible (not "are you looking for full-time
# roles?", which the Availability section does answer).
_PERMISSION_EN = re.compile(
    r"\b(?:can|could|able|allowed|eligible|permitted|entitled|need|needs|require|requires"
    r"|have|has|hold)\b",
    re.IGNORECASE,
)
_PERMISSION_ZH = ("能", "可以", "允许", "需要", "有没有", "有无", "是否有")

# A yes / no verdict at the start of the draft.
_VERDICT_EN = re.compile(
    r"^\W*(?:yes|yeah|yep|sure|absolutely|of course|definitely|certainly|correct|no|nope)\b",
    re.IGNORECASE,
)
_VERDICT_ZH = re.compile(r"^\W*(?:是的|是|对|可以|能|当然|没问题|不能|不可以|不行|不是)")

# A sentence that states, either way, whether Skylar can work full- or part-time.
_HOURS_CLAIM_EN = re.compile(
    r"\b(?:can(?:not|'t|’t)?|could(?:n't|n’t)?|(?:un)?able to|(?:not )?allowed to"  # noqa: RUF001
    r"|(?:in)?eligible to|permitted to)\s+(?:\w+\s+){0,4}?(?:full|part)[- ]?time\b"
    r"|\bwork(?:ing)?\s+(?:full|part)[- ]?time\b"
    r"|\b(?:full|part)[- ]?time\s+(?:work(?:ing)?\s+)?(?:rights?|permission|hours)\b",
    re.IGNORECASE,
)
_HOURS_CLAIM_ZH = re.compile(
    r"(?:可以|能够?|允许|不能|不可以)[^，。,.；;]{0,8}(?:全职|兼职)"  # noqa: RUF001
    r"|(?:全职|兼职)[^，。,.；;]{0,4}(?:工作权|权利|许可)"  # noqa: RUF001
)

# Words about work rights, visas, hours and pay; a draft may use one only if a section does.
_GUARDED_EN = re.compile(
    r"\bvisas?\b|\bsubclass\b|\bsponsor\w*|\bcitizen\w*|\bpermanent(?:ly)? resid\w*"
    r"|\bresidency\b|\bfortnight\w*|\bunrestricted\b|\bunlimited\b|\brestrict\w*"
    r"|\b\d+\s*hours?\b|\bhours (?:a|per) week\b|\bsalary\b|\bwages?\b|\bhourly\b"
    r"|\$\s?\d",
    re.IGNORECASE,
)
_GUARDED_ZH = ("签证", "担保", "公民", "永居", "绿卡", "限制", "小时", "工资", "薪")


# Talk about the prompt instead of an answer: "Section 1 says I can work in Australia, but
# it doesn't say ..., so please see the About page or email me" (v10-v12 runs, 1-2 per run).
_META_EN = re.compile(r"\bsections?\b|\bsee the about page\b", re.IGNORECASE)
_META_ZH = re.compile(r"段落|章节|第\s*\d+\s*[节段]")


def _asks_condition(question: str) -> bool:
    """Whether a question asks if something the page does not settle is allowed or possible."""
    en = bool(_CONDITION_EN.search(question) and _PERMISSION_EN.search(question))
    zh = any(t in question for t in _CONDITION_ZH) and any(t in question for t in _PERMISSION_ZH)
    return en or zh


def unstated_conclusion(question: str, draft: str, section_texts: Sequence[str]) -> str | None:
    """Why a private draft settles something the sections do not, or None if it doesn't.

    - "verdict": the question asks whether a condition the page does not settle
      (full-time, hours, visa, sponsorship, ...) is allowed or possible, and the draft
      opens with yes or no.
    - "hours": the draft says whether Skylar can work full- or part-time, in words no
      section uses.
    - "term": the draft talks about visas, sponsorship, residency, hours or pay, and no
      section does.
    - "meta": the draft talks about the sections or sends the visitor to the About page
      instead of answering (only the fixed sentence may do that).
    """
    source = " ".join(section_texts).casefold()
    if _asks_condition(question) and (_VERDICT_EN.search(draft) or _VERDICT_ZH.search(draft)):
        return "verdict"
    for pattern in (_HOURS_CLAIM_EN, _HOURS_CLAIM_ZH):
        for match in pattern.finditer(draft):
            if match.group(0).casefold() not in source:
                return "hours"
    for match in _GUARDED_EN.finditer(draft):
        if match.group(0).casefold() not in source:
            return "term"
    if any(term in draft and term not in source for term in _GUARDED_ZH):
        return "term"
    for pattern in (_META_EN, _META_ZH):
        for match in pattern.finditer(draft):
            if match.group(0).casefold() not in source:
                return "meta"
    return None
