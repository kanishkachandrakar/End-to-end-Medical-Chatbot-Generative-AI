"""What git is told about this repository's files, rather than left to guess.

`* text=auto` asks git to decide per file whether to normalise line endings,
by sniffing for NUL bytes. It decides correctly for the two binaries here --
but the cost of a wrong guess is CRLF normalisation applied inside a 15MB
source PDF or the one image the page loads, committed through the index and
invisible until something tries to read it.
"""

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

BINARY = (
    "Data/Medical_book.pdf",
    "static/medicalbotpic.jpeg",
)

TEXT = (
    "static/style.css",
    "templates/chat.html",
    "app.py",
)


def _attributes(path):
    """What git says about `path`, as {attribute: value}."""
    result = subprocess.run(
        ["git", "check-attr", "text", "diff", "--", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    found = {}
    for line in result.stdout.splitlines():
        _, attribute, value = line.rsplit(": ", 2)[0], *line.rsplit(": ", 2)[1:]
        found[attribute] = value
    return found


def test_the_files_are_all_tracked():
    """check-attr answers for paths that do not exist, so this reads first."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True,
        timeout=60,
    ).stdout.split()
    for path in BINARY + TEXT:
        assert path in tracked, path


def test_the_binaries_are_declared_binary():
    """Not left to content sniffing, which is a guess with a corrupt file
    as its failure mode."""
    for path in BINARY:
        attributes = _attributes(path)
        assert attributes["text"] == "unset", (path, attributes)


def test_the_text_files_are_still_normalised():
    """The declaration must not have widened: this repo is LF throughout."""
    for path in TEXT:
        attributes = _attributes(path)
        assert attributes["text"] == "auto", (path, attributes)


def test_the_notebook_is_never_merged_automatically():
    """One JSON document; git's line-based merge produces plausible rubbish."""
    assert "*.ipynb -merge" in (ROOT / ".gitattributes").read_text()
