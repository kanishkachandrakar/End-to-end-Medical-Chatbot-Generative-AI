"""static/style.css and the markup that uses it have to agree.

The page applies classes from two places -- `class="..."` in the template and
.addClass(...) in chat.js -- and style.css is the only thing that gives the
local ones any appearance. Neither side knows about the other, and a
disagreement is silent: the element renders, unstyled, which on a dark page
tends to mean invisible rather than ugly.

The live example is a misspelling. The bubbles are `msg_cotainer` and
`msg_cotainer_send`, missing the 'n', in chat.js and in style.css both. It
works because it is wrong in the same way twice, so correcting the spelling
in one file is a change that looks like tidying and removes the styling from
every message in the chat log.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
STYLESHEET = (ROOT / "static" / "style.css").read_text()
TEMPLATE = (ROOT / "templates" / "chat.html").read_text()
SCRIPT = (ROOT / "static" / "chat.js").read_text()

# Bootstrap and Font Awesome own the hyphenated names and ship their own
# rules; style.css redefines a few of them (.card, .card-header) on purpose.
# The local names are the ones with an underscore, plus .chat.
LOCAL_EXCEPTIONS = frozenset({"chat"})


def _defined():
    """Every class style.css has a rule for."""
    return set(re.findall(r"\.([A-Za-z_][A-Za-z0-9_-]*)", STYLESHEET))


def _applied():
    """Every class the page puts on an element, from either source."""
    classes = set()
    for attribute in re.findall(r'class="([^"]+)"', TEMPLATE):
        classes.update(attribute.split())
    # .addClass("a b c"), and the "a b " + (cond ? "x" : "y") form
    for literal in re.findall(r'addClass\(\s*\n?\s*"([^"]+)"', SCRIPT):
        classes.update(literal.split())
    for first, second in re.findall(
        r'\?\s*"([A-Za-z0-9_-]+)"\s*:\s*"([A-Za-z0-9_-]+)"', SCRIPT
    ):
        classes.update((first, second))
    return classes


def _is_local(name):
    return "_" in name or name in LOCAL_EXCEPTIONS


def test_both_sides_were_read():
    """Each check below is a set comparison; two empty sets agree."""
    assert len(_defined()) > 10, _defined()
    assert len(_applied()) > 15, _applied()
    # The misspelling this file exists for, named so that correcting it has
    # to come past this test.
    assert "msg_cotainer" in _defined() and "msg_cotainer" in _applied()


def test_every_rule_in_the_stylesheet_is_used():
    """A rule for a class nothing applies is dead weight that reads as live."""
    unused = sorted(name for name in _defined() if name not in _applied())
    assert not unused, f"style.css styles nothing: {unused}"


def test_every_local_class_the_page_applies_is_styled():
    """The other direction: an element with no rule renders unstyled."""
    unstyled = sorted(
        name for name in _applied() if _is_local(name) and name not in _defined()
    )
    assert not unstyled, f"applied but not in style.css: {unstyled}"


def test_the_ids_the_script_reaches_for_exist_in_the_template():
    """chat.js addresses five elements by id; a rename breaks it silently.

    $("#serviceBanner") on a missing element is an empty jQuery set, and
    every method on it succeeds and does nothing.
    """
    template_ids = set(re.findall(r'id="([^"]+)"', TEMPLATE))
    for wanted in re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', SCRIPT):
        assert wanted in template_ids, f"chat.js wants #{wanted}, the page has none"


def test_the_script_reaches_for_the_ids_we_think_it_does():
    """So the check above is comparing a real list rather than an empty one."""
    wanted = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', SCRIPT))
    assert wanted == {
        "messageFormeight",
        "text",
        "send",
        "serviceBanner",
        "messageArea",
    }
