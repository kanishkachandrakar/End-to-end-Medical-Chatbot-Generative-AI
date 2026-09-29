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


def test_leaves_unpaired_tags_alone():
    """A lone opening tag must not swallow the answer."""
    assert strip_reasoning("<think>unclosed reasoning") == "<think>unclosed reasoning"


def test_is_not_greedy_across_separate_blocks():
    """A greedy pattern would delete 'kept' along with both blocks."""
    assert strip_reasoning("<think>a</think>kept<think>b</think>") == "kept"
