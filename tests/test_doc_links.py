"""Every internal link in the documentation points at something.

Five markdown files and the notebook's header cross-reference each other and
the code -- SECURITY.md to DEPLOY.md, the README to CONTRIBUTING.md, the
notebook's table to three modules in src/. Nineteen such links, all valid
today, and nothing has ever checked them.

They rot quietly: a module is split, a file renamed, a section moved, and the
link still renders as a link. The notebook's table pointed at src/helper.py
for load_pdf for weeks after it moved to src/loaders.py -- correct-looking,
clickable, wrong.

External links are deliberately not checked. A test that fails because a
third party reorganised their site fails for a reason nobody here can fix,
and the suite is designed to pass with no network at all.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
EXTERNAL = ("http://", "https://", "#", "mailto:")


def _documents():
    """The markdown files, plus the notebook, which carries links in a cell."""
    return sorted(ROOT.glob("*.md")) + sorted(ROOT.glob("research/*.ipynb"))


def _internal_links():
    """(document, label, target) for every link that should resolve on disk."""
    found = []
    for document in _documents():
        for label, target in LINK.findall(document.read_text()):
            if target.startswith(EXTERNAL):
                continue
            found.append((document, label, target))
    return found


def test_there_are_documents_and_links_to_check():
    """Both lists below iterate; an empty one passes without reading."""
    assert len(_documents()) >= 5, [d.name for d in _documents()]
    assert len(_internal_links()) >= 15, len(_internal_links())


def test_every_internal_link_resolves():
    broken = []
    for document, label, target in _internal_links():
        path = (document.parent / target.split("#")[0]).resolve()
        if not path.exists():
            broken.append(f"{document.name}: [{label}]({target})")
    assert not broken, f"links to nothing: {broken}"


def test_no_internal_link_escapes_the_repository():
    """A ../.. that climbs out would resolve on one machine and not another."""
    outside = []
    for document, label, target in _internal_links():
        path = (document.parent / target.split("#")[0]).resolve()
        if ROOT not in path.parents and path != ROOT:
            outside.append(f"{document.name}: [{label}]({target})")
    assert not outside, f"links outside the repository: {outside}"
