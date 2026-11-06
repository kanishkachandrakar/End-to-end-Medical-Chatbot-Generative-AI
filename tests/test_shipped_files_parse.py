"""The non-Python files the app serves must be well formed.

Every one of these has been checked by hand after each change for weeks -- an
HTML parse, a json.load, a brace count -- because a script editing them as text
can leave them plausibly broken rather than obviously so. A check run by hand is
a check that stops being run.
"""

import json
import pathlib
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parent.parent

VOID_ELEMENTS = {"img", "link", "meta", "br", "hr", "input", "source", "area"}


class _Balance(HTMLParser):
    """Records tag nesting so an unclosed or crossed element is visible."""

    def __init__(self):
        super().__init__()
        self.open_tags: list[str] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID_ELEMENTS:
            self.open_tags.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID_ELEMENTS:
            return
        if not self.open_tags:
            self.problems.append(f"</{tag}> with nothing open")
        elif self.open_tags[-1] != tag:
            self.problems.append(f"</{tag}> closes <{self.open_tags[-1]}>")
            self.open_tags.pop()
        else:
            self.open_tags.pop()


def test_the_chat_template_is_balanced():
    parser = _Balance()
    parser.feed((ROOT / "templates" / "chat.html").read_text())
    assert not parser.problems, parser.problems
    assert not parser.open_tags, f"never closed: {parser.open_tags}"


def test_the_chat_template_opens_with_the_doctype():
    """Anything above it puts the browser into quirks mode, which this page
    has already shipped once -- the Bootstrap layout silently misbehaves."""
    text = (ROOT / "templates" / "chat.html").read_text().lstrip()
    assert text.startswith("<!DOCTYPE html>"), text[:60]


def test_the_notebook_is_valid_json():
    """Every edit to it here has been made by a script, and more than one of
    those got it wrong. The pre-commit hook checks this too, for anyone who
    has installed it."""
    notebook = json.loads((ROOT / "research" / "trials.ipynb").read_text())
    assert notebook["nbformat"] >= 4
    assert notebook["cells"], "the notebook has no cells"


def test_the_notebook_carries_no_outputs():
    """Committed outputs were a dump of Documents with ids from a dead index,
    and they make every diff of the notebook noisy."""
    notebook = json.loads((ROOT / "research" / "trials.ipynb").read_text())
    with_output = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("outputs") or cell.get("execution_count") is not None
    ]
    assert not with_output, (
        f"cells carrying output or an execution count: {with_output}"
    )


def test_the_stylesheet_braces_balance():
    """A regex rewrite of this file once dropped the closing brace of a nested
    @media block, which neither the page nor any test noticed."""
    css = (ROOT / "static" / "style.css").read_text()
    assert css.count("{") == css.count("}"), (
        f"{css.count('{')} opening braces, {css.count('}')} closing"
    )


def test_the_stylesheet_is_not_empty():
    """The brace check above passes trivially against an empty file."""
    assert len((ROOT / "static" / "style.css").read_text()) > 500


def test_every_static_file_the_template_names_exists():
    import re

    template = (ROOT / "templates" / "chat.html").read_text()
    named = re.findall(r"filename='([^']+)'", template)
    assert named, "the template names no static files"
    for name in named:
        assert (ROOT / "static" / name).is_file(), name


def test_the_files_editorconfig_declares_tab_indented_are():
    """.editorconfig says tabs for the template and the stylesheet.

    Nothing checked it, and template.py was found three spaces deep the same
    way -- against the four its own section declares. The stylesheet had two
    space-indented lines and one with no indentation at all, which is how a
    file stops being obviously hand-maintained.
    """
    import re

    editorconfig = (ROOT / ".editorconfig").read_text()
    section = re.search(
        r"\[\{?([^\]]*?)\}?\]\s*\nindent_style = tab", editorconfig
    )
    assert section, "no tab-indented section in .editorconfig"
    patterns = [part.strip() for part in section.group(1).split(",")]
    assert patterns, patterns

    offenders = []
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if line.startswith(" "):
                    offenders.append(f"{path.name}:{number}")
    assert not offenders, f"space-indented where .editorconfig says tab: {offenders}"


def test_that_check_is_looking_at_the_files_we_think():
    """An empty glob would pass the check above without reading anything."""
    import re

    editorconfig = (ROOT / ".editorconfig").read_text()
    section = re.search(r"\[\{?([^\]]*?)\}?\]\s*\nindent_style = tab", editorconfig)
    assert section
    found = [
        path.name
        for pattern in (p.strip() for p in section.group(1).split(","))
        for path in ROOT.glob(pattern)
    ]
    assert "chat.html" in found, found
    assert "style.css" in found, found
