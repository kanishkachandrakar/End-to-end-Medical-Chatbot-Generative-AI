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


def test_line_breaks_in_an_answer_survive():
    """An LLM answer with a list in it must not render as one run-on line."""
    for selector in (".msg_cotainer", ".msg_cotainer_send"):
        assert "white-space: pre-wrap" in _rule(selector), selector


def test_a_long_word_cannot_overflow_its_bubble():
    """Both halves are needed: a flex item will not shrink without min-width."""
    for selector in (".msg_cotainer", ".msg_cotainer_send"):
        body = _rule(selector)
        assert "overflow-wrap: anywhere" in body, selector
        assert "min-width: 0" in body, selector


def test_the_placeholder_is_light_on_the_dark_field():
    """Bootstrap's default placeholder colour is a mid grey meant for white."""
    assert "rgba(255,255,255," in _rule(".type_msg::placeholder")


def test_the_card_fits_a_short_viewport():
    """A fixed 500px card pushes the input off a landscape phone screen."""
    assert re.search(r"max-height:\s*calc\(100vh", _rule(".card"))


def test_the_log_is_marked_busy_while_waiting():
    """Otherwise the live region announces the placeholder before the answer."""
    set_busy = re.search(r"function setBusy\(busy\) \{(.*?)\n    \}", SCRIPT, re.S)
    assert set_busy, "setBusy is no longer where this test looks for it"
    assert '$log.attr("aria-busy"' in set_busy.group(1)


def test_the_status_dot_follows_the_banner():
    """A green "online" dot under a red outage banner contradicts it."""
    show_banner = re.search(
        r"function showBanner\(text\) \{(.*?)\n    \}", SCRIPT, re.S
    )
    assert show_banner, "showBanner is no longer where this test looks for it"
    assert 'addClass("online_icon_down")' in show_banner.group(1)
    # Declared after .online_icon, so it wins at equal specificity.
    assert STYLESHEET.index(".online_icon_down{") > STYLESHEET.index(".online_icon{")
    assert "background-color" in _rule(".online_icon_down")


def test_the_disclaimer_is_not_fine_print():
    """The header subtitle is where the page says it is not medical advice."""
    template = (ROOT / "templates" / "chat.html").read_text()
    assert "not medical advice" in template
    size = re.search(r"font-size:\s*(\d+)px", _rule(".user_info p"))
    assert size and int(size.group(1)) >= 12, "the disclaimer is too small to read"
