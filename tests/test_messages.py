"""The text a visitor can be shown, held to being plain.

Every string here reaches a chat bubble through .text(), so the page cannot
render markup in it -- but it can render internal vocabulary, and that is the
failure worth guarding. A refusal naming a component tells the reader nothing
and invites them to debug a system they cannot see.
"""

import pathlib
import re

from src import messages

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Words that mean something here and nothing to a visitor.
INTERNAL = (
    "chain",
    "pinecone",
    "groq",
    "index",
    "embedding",
    "vector",
    "gunicorn",
    "traceback",
    "exception",
    "token bucket",
)


def test_there_are_messages_to_check():
    """Everything below iterates ALL; an empty tuple would pass it all."""
    assert len(messages.ALL) >= 6


def test_all_lists_every_message_defined():
    """A string left out of ALL is a string nothing below checks."""
    defined = {
        value
        for name, value in vars(messages).items()
        if name.isupper() and name != "ALL" and isinstance(value, str)
    }
    assert defined == set(messages.ALL), defined ^ set(messages.ALL)


def test_no_message_names_an_internal_component():
    offenders = [
        (text, word)
        for text in messages.ALL
        for word in INTERNAL
        if word in text.lower()
    ]
    assert not offenders, f"visitor-facing text naming internals: {offenders}"


def test_no_message_contains_markup():
    """The front end renders these as text; markup would appear literally."""
    for text in messages.ALL:
        assert "<" not in text and ">" not in text, text


def test_every_message_is_a_finished_sentence():
    for text in messages.ALL:
        assert text[0].isupper(), text
        assert text.rstrip().endswith("."), text


def test_the_only_placeholder_is_the_documented_one():
    """A stray brace would render literally or raise on .format()."""
    for text in messages.ALL:
        for field in re.findall(r"\{(\w*)\}", text):
            assert field == "limit", f"{text!r} has an unexpected {{{field}}}"


def test_the_banner_text_follows_the_same_rule():
    """static/chat.js has its own strings, shown in the same place."""
    script = (ROOT / "static" / "chat.js").read_text()
    shown = re.findall(r'showBanner\(\s*((?:"[^"]*"\s*\+?\s*)+)', script)
    assert shown, "no banner strings found"
    for literal in shown:
        text = " ".join(re.findall(r'"([^"]*)"', literal))
        for word in ("chain", "pinecone", "groq", "gunicorn"):
            assert word not in text.lower(), f"banner names {word!r}: {text}"
