"""The template and the Flask routes have to agree on names.

Nothing fails loudly when they stop agreeing: the page renders, the button
does nothing, and the only clue is a console error nobody is watching.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = (ROOT / "templates" / "chat.html").read_text()
# The script block contains jQuery element constructors such as $("<img>"),
# which are not markup and must not be scanned as if they were.
MARKUP = re.sub(r"<script>.*?</script>", "", HTML, flags=re.DOTALL)


def test_the_elements_the_script_drives_exist():
    for element_id in ("text", "send", "messageArea", "messageFormeight"):
        assert f'id="{element_id}"' in HTML, element_id


def test_every_queried_id_is_defined_in_the_markup():
    """Catches a renamed id that the script still looks for."""
    queried = set(re.findall(r'\$\("#([A-Za-z_][\w-]*)"\)', HTML))
    defined = set(re.findall(r'id="([^"]+)"', HTML))
    assert queried <= defined, (
        f"script queries ids that do not exist: {queried - defined}"
    )


def test_posts_to_the_route_the_app_serves():
    assert 'url: "/get"' in HTML
    app_py = (ROOT / "app.py").read_text()
    webapp = ROOT / "src" / "webapp.py"
    routes = app_py + (webapp.read_text() if webapp.exists() else "")
    assert '"/get"' in routes


def test_sends_the_field_name_the_route_reads():
    assert re.search(r"data:\s*\{\s*msg:", HTML)


def test_the_css_classes_the_script_applies_are_styled():
    """Renaming a class in the stylesheet silently unstyles every bubble."""
    css = (ROOT / "static" / "style.css").read_text()
    applied = set(re.findall(r'addClass\("([a-z_]+)"\)', HTML))
    # classes chosen by a ternary, e.g. fromUser ? "msg_time_send" : "msg_time"
    for pair in re.findall(r'\?\s*"([a-z_]+)"\s*:\s*"([a-z_]+)"', HTML):
        applied.update(pair)
    project_classes = {c for c in applied if "_" in c}
    missing = {c for c in project_classes if f".{c}" not in css}
    assert not missing, (
        f"applied by the script but absent from style.css: {missing}"
    )


def test_static_assets_are_referenced_through_url_for():
    """A hardcoded /static path breaks as soon as the app is mounted elsewhere."""
    assert "url_for('static'" in HTML
    assert 'src="/static' not in HTML and 'href="/static' not in HTML


def test_the_avatar_file_exists():
    referenced = re.findall(r"filename='([^']+)'", HTML)
    for name in referenced:
        assert (ROOT / "static" / name).exists(), name


def test_the_icon_only_button_has_an_accessible_name():
    """A <button> containing only an <i> announces as 'button' and nothing else."""
    assert 'aria-label="Send question"' in HTML


def test_the_free_text_input_is_labelled():
    """There is no visible <label>, so the input needs one of its own."""
    assert 'aria-label="Your question"' in HTML


def test_replies_are_announced():
    """The log is appended to by script, which a screen reader otherwise misses."""
    assert 'aria-live="polite"' in HTML
    assert 'role="log"' in HTML


def test_every_image_is_labelled_or_marked_decorative():
    """An <img> with no alt at all is read out as its filename."""
    tags = re.findall(r"<img[^>]*>", MARKUP)
    assert tags, "expected at least one image in the markup"
    for tag in tags:
        decorative = 'aria-hidden="true"' in tag or 'alt=""' in tag
        labelled = re.search(r'alt="[^"]+"', tag)
        assert decorative or labelled, tag


def test_images_built_by_the_script_also_set_alt():
    """The avatars in each bubble are created in JS, not in the markup."""
    for constructor in re.findall(r'\$\("<img>"\)((?:\s*\.\w+\([^)]*\))+)', HTML):
        assert '"alt"' in constructor, constructor


def test_a_favicon_is_declared():
    """Browsers request /favicon.ico regardless; declaring one avoids a 404."""
    assert 'rel="icon"' in MARKUP
