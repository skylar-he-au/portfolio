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
