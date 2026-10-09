"""static/chat.js builds the chat log, and once did so by concatenating HTML.

That was an injection: a question, or a passage retrieved from the PDF, could
contain markup and it was parsed. The fix was to build DOM nodes and set their
text. Nothing stops it coming back except this.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = (ROOT / "static" / "chat.js").read_text()


def test_the_script_is_not_empty():
    """The assertions below all pass trivially against an empty file."""
    assert len(SCRIPT) > 500


def test_message_text_is_set_as_text_not_markup():
    assert ".text(" in SCRIPT


def test_nothing_assigns_innerhtml():
    assert "innerHTML" not in SCRIPT
    assert "outerHTML" not in SCRIPT


def test_jquery_html_is_not_used():
    """$el.html(x) is the jQuery spelling of the same mistake."""
    assert not re.search(r"\.html\s*\(", SCRIPT)


def test_html_is_not_parsed_from_a_string():
    """$.parseHTML on a reply is how the original injection worked."""
    assert "parseHTML" not in SCRIPT


def test_nothing_is_evaluated():
    assert not re.search(r"\beval\s*\(", SCRIPT)
    assert not re.search(r"new\s+Function\s*\(", SCRIPT)


def test_no_timer_is_given_a_string_to_run():
    """setTimeout('...') evaluates its argument, like eval with a delay."""
    for match in re.finditer(r"set(?:Timeout|Interval)\s*\(\s*(.)", SCRIPT):
        assert match.group(1) not in "\"'", "a timer was handed a string"


def test_the_reply_goes_through_the_same_builder_as_the_question():
    """One code path means the escaping cannot be right for only one of them."""
    assert SCRIPT.count("appendMessage(") >= 3
