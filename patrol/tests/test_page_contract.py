"""The patrol's page selectors and task text must match the home page's one assistant box
(`section#ask`, design step 9). No browser here: the strings are checked, and the selectors are
tried against the real site by running the patrol (README)."""

from patrol.run import SNAPSHOT_JS
from patrol.task import JOB_AD, task_text

TASK = task_text("http://localhost:4321")


def test_job_ad_is_long_enough_to_be_matched_not_asked():
    # The box treats more than 500 characters as a job ad (web/src/lib/detect.ts, ASK_MAX_CHARS).
    assert len(JOB_AD) > 500


def test_snapshot_reads_the_one_assistant_box():
    assert "#match" not in SNAPSHOT_JS  # the home page has no #match any more
    assert "querySelector('#ask')" in SNAPSHOT_JS
    assert ".match-result" in SNAPSHOT_JS
    assert '[aria-label="This guide is wrong"]' in SNAPSHOT_JS
    assert '[aria-label="This answer is wrong"]' in SNAPSHOT_JS


def test_snapshot_does_not_count_restored_turns_as_the_new_answer():
    # After going back to the home page, earlier turns are read back from sessionStorage and
    # carry data-restored: they must not count as an answer finished in this run.
    assert ".ask-turn:not([data-restored])" in SNAPSHOT_JS
    assert "askAnswers" in SNAPSHOT_JS and "data-restored" in SNAPSHOT_JS


def test_task_uses_the_new_page_wording():
    assert "#match" not in TASK
    assert '"Send"' not in TASK  # an example question is sent when clicked
    assert '"Match job"' in TASK
    assert "Ask a question, or paste a job ad" in TASK
    assert "#ask .match-result" in TASK
    assert "data-restored" in TASK


def test_task_waits_for_the_new_answer_not_a_restored_one():
    assert '.ask-turn:not([data-restored]) [aria-label="This answer is wrong"]' in TASK
