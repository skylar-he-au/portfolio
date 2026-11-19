"""The match-v5+check-v2 rules are in the prompts the agents get (step 6 round 2).

One test per new rule: each asserts the rule's sentence reaches the system prompt, so a
later edit that drops it fails here and has to bump `PROMPT_VERSION` on purpose.
"""

from app.match import prompts
from app.match.state import Requirement
from app.rag.types import Section

MARKER = "0123456789abcdef"
REQUIREMENT = Requirement(id="r1", text="TypeScript, React and Node.js", query="TypeScript")
SECTION = Section("/projects/oncallops", "dashboard", "Dashboard", "Next.js pages.", 0.01)


def _system(messages: list) -> str:
    return " ".join(str(messages[0].content).split())


def _read() -> str:
    return _system(prompts.read_messages("A job ad.", MARKER))


def _write(language: str = "en") -> str:
    return _system(prompts.write_messages([(1, REQUIREMENT, [SECTION])], language, MARKER))


def _check() -> str:
    return _system(prompts.check_messages([(1, REQUIREMENT, SECTION)], MARKER))


def test_version_names_both_parts() -> None:
    assert prompts.PROMPT_VERSION == "match-v5+check-v2"


# --- read -----------------------------------------------------------------------------------


def test_read_puts_skills_of_one_kind_in_one_requirement() -> None:
    assert (
        "Put skills of one kind in one requirement, even when the ad names them in more "
        "than one place: the languages and frameworks of one stack are one requirement "
        '("TypeScript, React and Node.js"), not one per tool.'
    ) in _read()


def test_read_keeps_its_earlier_rules() -> None:
    system = _read()
    assert "One line or bullet of the ad is one requirement." in system
    assert "Keep an amount of experience with its skill" in system
    assert "Merge requirements that say the same thing." in system


# --- write ----------------------------------------------------------------------------------


def test_write_states_only_what_the_section_says() -> None:
    system = _write()
    assert "Every sentence states only what the section text itself says." in system
    assert (
        "The requirement's words are what the job asks for, not facts about the project: "
        "never write that a project shows or uses something the requirement names "
        '(such as "Agile delivery") unless a section says so.'
    ) in system


def test_write_keeps_to_the_part_that_matches() -> None:
    assert (
        "Write only the part of the sections that matches this requirement; leave out "
        "tools and work it does not ask about"
    ) in _write()


def test_write_uses_the_ad_language_with_names_and_numbers_kept() -> None:
    system = _write("zh")
    assert (
        "Write in the job ad's language, given below. Keep only proper names (products, "
        "tools, projects, pages) and numbers as the sections write them, and translate "
        "everything else"
    ) in system
    assert "Dashboard 页面用 keyset 分页\uff0c后端有 2,576 个测试" in system
    assert system.endswith("Write the text in Chinese (language code 'zh').")


def test_write_keeps_statements_and_its_own_sections() -> None:
    system = _write()
    assert "Write it as a plain statement about the project, not an instruction" in system
    assert (
        "Use only the sections listed under this finding, never a section listed under "
        "another finding."
    ) in system


# --- check ----------------------------------------------------------------------------------


def test_check_opens_with_what_it_decides() -> None:
    system = _check()
    assert system.startswith(
        "You decide whether one section of Skylar He's project pages shows one job requirement."
    )
    assert "guide" not in system.lower()


def test_check_and_match_still_share_the_shows_rule() -> None:
    rule = " ".join(prompts._SHOWS_RULE.split())
    assert rule in _check()
    match = _system(prompts.match_messages(REQUIREMENT, [SECTION], MARKER))
    assert rule in match


def test_probes_follow_the_new_check_opening() -> None:
    assert any(_check().startswith(probe) for probe in prompts.SYSTEM_PROMPT_PROBES)
    assert "You check links in a reading guide" not in prompts.SYSTEM_PROMPT_PROBES
