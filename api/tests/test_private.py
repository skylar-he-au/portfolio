"""The code backstop for private drafts (10s): `private.unstated_conclusion`."""

import pytest

from app.qa.private import unstated_conclusion

# The About page as it is (content/about.md, 2026-11-17).
ABOUT = [
    "This is the About page of Skylar He's portfolio. I'm studying a Master of Computer "
    "Science at the University of Wollongong (July 2025 to June 2027, expected) and also "
    "doing the JR Academy Full-Stack Development Program. I'm looking for AI engineer and "
    "full stack roles, and backend, frontend and software engineer roles too, in the tech "
    "industry only. I'm based in Sydney.",
    "About Skylar's work rights: I'm studying in Sydney and can work in Australia; happy to "
    "discuss details — email me.",
    "I'm Skylar, based in Sydney, NSW, and I'm looking for work in Sydney or Wollongong only.",
    "I'm Skylar, and I'm looking for internships and part-time work now, and graduate "
    "programs and full-time roles from early 2027 (I graduate in June 2027).",
    "Here's how to contact me (Skylar):\n\n- LinkedIn: https://www.linkedin.com/in/skylar-he",
]

FULLTIME = "Can you work full-time in Australia?"


@pytest.mark.parametrize(
    ("question", "draft", "rule"),
    [
        # the 10p answer (20261125T023509-v11-default, pa-fulltime, pass 6)
        (
            FULLTIME,
            "Yes, I can work in Australia, and I'm happy to discuss the details, so please "
            "email me [1]. I'm looking for internships and part-time work now, and graduate "
            "programs and full-time roles from early 2027, when I graduate in June 2027 [4][3].",
            "verdict",
        ),
        (FULLTIME, "No, not yet [1].", "verdict"),
        ("Do you need visa sponsorship?", "No, I can work in Australia [1].", "verdict"),
        ("Are you allowed to work part-time?", "Sure, I can work in Australia [1].", "verdict"),
        ("你能在澳大利亚全职工作吗？", "可以，我在悉尼读书，可以在澳大利亚工作 [1]。", "verdict"),  # noqa: RUF001
        # a conclusion on hours, without a yes / no
        (FULLTIME, "I can work full-time in Australia [1].", "hours"),
        (FULLTIME, "I can't work full-time until I graduate [4].", "hours"),
        ("What are your work rights?", "I'm able to work full-time [1].", "hours"),
        ("Work rights?", "I have full-time work rights [1].", "hours"),
        ("工作权利？", "我可以在澳大利亚全职工作 [1]。", "hours"),  # noqa: RUF001
        # words about visas, hours or pay that no section uses
        ("What are your work rights?", "I'm on a student visa [1].", "term"),
        ("What are your work rights?", "I can work 48 hours a fortnight [1].", "term"),
        ("What are your work rights?", "I have unrestricted work rights [1].", "term"),
        ("What are your work rights?", "I don't need sponsorship [1].", "term"),
        ("你的工作权利？", "我持学生签证，可以在澳大利亚工作 [1]。", "term"),  # noqa: RUF001
        # talk about the sections instead of an answer (v12 run 20261125T025517, pass 13)
        (
            FULLTIME,
            "Section 1 says I can work in Australia, but it doesn't say whether that covers "
            "full-time work, so please see the About page or email me.",
            "meta",
        ),
        (
            "Where are you based?",
            "I'm based in Sydney [1], but the sections don't say more [1].",
            "meta",
        ),
        ("Where are you based?", "I'm based in Sydney [1]. For more, see the About page.", "meta"),
        ("你在哪？", "第1段说我在悉尼 [1]。", "meta"),  # noqa: RUF001
    ],
)
def test_unstated_conclusions_are_caught(question: str, draft: str, rule: str) -> None:
    assert unstated_conclusion(question, draft, ABOUT) == rule


@pytest.mark.parametrize(
    ("question", "draft"),
    [
        # what the About page says, in its own words or close to them
        (
            "What are your work rights in Australia?",
            "I'm studying in Sydney and can work in Australia; happy to discuss details — "
            "email me [1].",
        ),
        (
            FULLTIME,
            "I'm studying in Sydney and can work in Australia [1]. I'm looking for "
            "internships and part-time work now, and graduate programs and full-time roles "
            "from early 2027 [2].",
        ),
        (
            "Are you looking for full-time roles?",
            "Yes, I'm looking for graduate programs and full-time roles from early 2027 [1].",
        ),
        ("Are you looking for internships right now?", "Yes, I'm looking for internships now [1]."),
        ("Can you work in Australia?", "Yes, I can work in Australia [1]."),
        (
            "Where are you based and where do you want to work?",
            "I'm based in Sydney, NSW, and I'm looking for work in Sydney or Wollongong only [1].",
        ),
        ("你现在找实习吗？", "是的，我现在在找实习和兼职工作 [1]。"),  # noqa: RUF001
    ],
)
def test_answers_from_the_page_pass(question: str, draft: str) -> None:
    assert unstated_conclusion(question, draft, ABOUT) is None


def test_a_word_the_sections_use_is_allowed() -> None:
    sections = ["I hold a graduate visa and can work full-time in Australia."]
    question = "Which visa are you on?"
    assert unstated_conclusion(question, "I hold a graduate visa [1].", sections) is None
    assert unstated_conclusion(question, "I can work full-time in Australia [1].", sections) is None


# --- page labels in the answer prompt (v12, 10s) -----------------------------------------


def test_sections_carry_their_page_kind_and_the_answer_rule_uses_it() -> None:
    from app.qa import prompts
    from app.rag.types import Section

    post = Section(
        "/blog/how-i-evaluate-my-portfolio-ai",
        "can-i-trust-the-judge",
        "Can I trust the judge?",
        "An LLM judge can simply be wrong.",
    )
    project = Section("/projects/roleradar", "evals", "Evals", "RoleRadar has evals.")
    about = Section("/about", "work-rights", "Work rights", "I can work in Australia.")
    privacy = Section("/privacy", "who-sees", "Who sees", "Nobody.")
    assert [prompts.page_kind(s.page) for s in (post, project, about, privacy)] == [
        "blog post",
        "project page",
        "About page",
        "site page",
    ]
    system, human = prompts.answer_messages("Why no model grader?", "en", [project, post], [])
    assert 'page="/blog/how-i-evaluate-my-portfolio-ai" kind="blog post"' in str(human.content)
    assert 'page="/projects/roleradar" kind="project page"' in str(human.content)
    assert "Keep every fact with its own page" in str(system.content)
    assert "name that page in the sentence" in str(system.content)
    assert prompts.PROMPT_VERSION == "v12"
