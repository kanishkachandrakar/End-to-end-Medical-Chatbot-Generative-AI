"""The reply the user sees is whatever comes out of strip_reasoning()."""

import pytest

from src.text import strip_reasoning


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "<think>weighing it up</think>Acne is a skin condition.",
            "Acne is a skin condition.",
        ),
        ("Acne is a skin condition.", "Acne is a skin condition."),
        ("<think>only reasoning</think>", ""),
        ("before<think>middle</think>after", "beforeafter"),
        ("  surrounded by space  ", "surrounded by space"),
        ("", ""),
    ],
)
def test_removes_think_blocks(raw, expected):
    assert strip_reasoning(raw) == expected


def test_spans_newlines():
    """re.DOTALL matters: the reasoning is almost always multi-line."""
    raw = "<think>line one\nline two\nline three</think>The answer."
    assert strip_reasoning(raw) == "The answer."


def test_removes_every_block_not_just_the_first():
    raw = "<think>a</think>one<think>b</think>two"
    assert strip_reasoning(raw) == "onetwo"


def test_an_unterminated_block_is_removed_not_published():
    """This used to assert the opposite, and the opposite was the leak.

    The tag is unclosed because generation stopped -- the token ceiling or
    the 60s Groq timeout, partway through reasoning -- so there is no answer
    after it to protect. What was published instead was the model's chain of
    thought with a `<think>` on the front, in a chat bubble, from a module
    whose first comment says it should never reach the user.
    """
    assert strip_reasoning("<think>unclosed reasoning") == ""
    assert strip_reasoning("<think>line one\nline two, cut off here") == ""


def test_an_empty_result_is_what_the_route_turns_into_no_answer():
    """Stripping everything is the intended outcome, not a broken one."""
    from src.messages import NO_ANSWER
    from src.webapp import create_app

    class Truncated:
        def invoke(self, payload):
            return {"answer": "<think>reasoning the model never finished"}

    response = create_app(Truncated()).test_client().post("/get", data={"msg": "hi"})
    assert response.status_code == 200
    assert response.data.decode() == NO_ANSWER
    assert b"<think>" not in response.data


def test_text_before_an_orphan_closing_tag_goes_too():
    """A reply whose opening was lost: what precedes </think> is reasoning."""
    assert strip_reasoning("tail of the reasoning</think>The answer.") == "The answer."


def test_an_answer_following_an_unclosed_tag_is_sacrificed_knowingly():
    """Recorded because it is the cost, not an oversight.

    If a reply ever did carry an answer after an unclosed opener, it is
    discarded along with the reasoning. The visitor gets NO_ANSWER, which is
    a worse answer and a better failure than the alternative.
    """
    assert strip_reasoning("<think>reasoning\nAcne is a skin condition.") == ""


def test_is_not_greedy_across_separate_blocks():
    """A greedy pattern would delete 'kept' along with both blocks."""
    assert strip_reasoning("<think>a</think>kept<think>b</think>") == "kept"


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "1 second"),
        (1, "1 second"),
        (2, "2 seconds"),
        (59, "59 seconds"),
        (60, "about a minute"),
        (89, "about a minute"),
        (120, "about 2 minutes"),
        (600, "about 10 minutes"),
    ],
)
def test_a_wait_is_worded_for_a_sentence(seconds, expected):
    """"1 seconds" in a refusal makes the rest of it less believable."""
    from src.text import seconds_in_words

    assert seconds_in_words(seconds) == expected


def test_no_wait_renders_as_a_bare_number():
    """So the message always reads as English rather than as a field."""
    from src.text import seconds_in_words

    for seconds in range(0, 300):
        rendered = seconds_in_words(seconds)
        assert rendered.endswith(("second", "seconds", "minute", "minutes")), rendered
