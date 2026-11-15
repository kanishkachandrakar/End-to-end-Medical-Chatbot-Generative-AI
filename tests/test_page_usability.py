"""Usability and accessibility properties of the chat page's stylesheet and script.

test_stylesheet_contract.py checks that style.css and the markup agree on
class names. This checks what some of those rules actually do, for the cases
where a missing declaration is invisible to a mouse user on a large screen and
so would not be noticed in a manual look at the page.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
STYLESHEET = (ROOT / "static" / "style.css").read_text()
SCRIPT = (ROOT / "static" / "chat.js").read_text()


def _rule(selector):
    """The declarations of the first rule whose selector list names `selector`."""
    for selectors, body in re.findall(r"([^{}]+)\{([^}]*)\}", STYLESHEET):
        uncommented = re.sub(r"/\*.*?\*/", "", selectors, flags=re.S)
        names = [s.strip() for s in uncommented.split(",")]
        if selector in names:
            return body
    raise AssertionError(f"style.css has no rule for {selector}")


def test_focus_is_never_hidden():
    """Removing the outline with nothing in its place hides keyboard focus."""
    assert not re.search(r"outline\s*:\s*0", STYLESHEET)
    for selector in (".type_msg:focus", ".send_btn:focus-visible"):
        assert re.search(r"outline\s*:\s*2px solid", _rule(selector)), selector


def test_the_busy_state_is_visible():
    """setBusy disables the controls; the stylesheet has to show that it did."""
    assert '$("#send").prop("disabled", busy)' in SCRIPT
    for selector in (".type_msg:disabled", ".send_btn:disabled"):
        body = _rule(selector)
        assert "opacity" in body and "not-allowed" in body, selector
