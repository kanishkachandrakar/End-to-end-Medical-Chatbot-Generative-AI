"""The template and the Flask routes have to agree on names.

Nothing fails loudly when they stop agreeing: the page renders, the button
does nothing, and the only clue is a console error nobody is watching.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = (ROOT / "templates" / "chat.html").read_text()


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
