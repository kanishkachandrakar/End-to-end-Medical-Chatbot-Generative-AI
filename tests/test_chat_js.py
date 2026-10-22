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


def test_the_failure_path_does_not_display_the_body_directly():
    """A body is chosen by failureText, not handed straight to the log.

    The old spelling was `jqXHR.responseText || "..."`, which displays
    whatever came back -- including the HTML page the Space's router serves
    when the container is restarting.
    """
    assert "appendMessage(failureText(jqXHR)" in SCRIPT
    assert not re.search(r"appendMessage\(\s*\n?\s*jqXHR\.responseText", SCRIPT)


# The rest of this file reads the script as text, because that is all the
# injection checks need. These run it: failureText is a pure function of a
# jqXHR's content type and body, so node can be given a fake one and asked
# what the bubble would say. CI has node already, for `node --check`.
_HARNESS = """
%s
function jqXHR(type, body) {
    return {
        getResponseHeader: function () { return type; },
        responseText: body
    };
}
const cases = %s;
console.log(JSON.stringify(cases.map(c => failureText(jqXHR(c[0], c[1])))));
"""


def _ask_node(cases):
    """Run failureText over `cases` and return what each would display."""
    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:  # pragma: no cover - CI and the Makefile both have node
        pytest.skip("node is not installed")

    source = re.search(
        r"^    function failureText\(jqXHR\) \{\n(?:.*\n)*?^    \}$",
        SCRIPT,
        re.MULTILINE,
    )
    assert source, "failureText is no longer where this test looks for it"

    script = _HARNESS % (source.group(0), json.dumps(cases))
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


FALLBACK = "Sorry, the server is not reachable right now."


def test_a_plain_text_refusal_is_shown_as_sent():
    """The server's own messages are written for a reader; show them."""
    sent = "This demo answers a limited number of questions a minute."
    assert _ask_node([["text/plain; charset=utf-8", sent]]) == [sent]


def test_an_html_error_page_is_not_shown():
    """What a router in front of the app serves when the container is down."""
    page = "<!doctype html><html><head><title>502 Bad Gateway</title></head></html>"
    assert _ask_node([["text/html; charset=utf-8", page]]) == [FALLBACK]


def test_a_long_body_is_not_shown_even_if_it_claims_to_be_text():
    """Every message the server can send here is one sentence."""
    assert _ask_node([["text/plain", "x" * 4000]]) == [FALLBACK]


def test_an_empty_or_missing_body_falls_back():
    """status 0: the request never arrived, so there is no body at all."""
    assert _ask_node([["", ""], ["text/plain", "   "]]) == [FALLBACK, FALLBACK]
