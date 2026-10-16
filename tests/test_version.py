"""The declared version and the changelog have to agree.

A version bumped in one place and not the other is worse than no version: the
package says one thing, the changelog another, and whichever a reader happens
to look at is the one they believe.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETUP = (ROOT / "setup.py").read_text()
CHANGELOG = (ROOT / "CHANGELOG.md").read_text()

SEMVER = r"\d+\.\d+\.\d+"


def _declared_version():
    match = re.search(rf'version="({SEMVER})"', SETUP)
    assert match, "setup.py declares no version"
    return match.group(1)


def _changelog_versions():
    """Released versions, newest first, from the `## x.y.z` headings."""
    return re.findall(rf"^## ({SEMVER})", CHANGELOG, re.MULTILINE)


def test_both_sides_were_read():
    assert _declared_version()
    assert _changelog_versions(), "the changelog has no version heading"


def test_the_declared_version_is_the_newest_released_one():
    assert _declared_version() == _changelog_versions()[0]


def test_each_release_heading_carries_a_date():
    """A version with no date cannot be placed against anything else."""
    for line in CHANGELOG.splitlines():
        if re.match(rf"^## {SEMVER}", line):
            assert re.search(r"\d{4}-\d{2}-\d{2}", line), line


def test_the_versions_are_in_descending_order():
    versions = [tuple(int(p) for p in v.split(".")) for v in _changelog_versions()]
    assert versions == sorted(versions, reverse=True), versions


def test_unreleased_work_sits_above_the_releases():
    """Otherwise it reads as part of the version below it."""
    if "## Unreleased" not in CHANGELOG:
        return
    assert CHANGELOG.index("## Unreleased") < CHANGELOG.index(
        f"## {_changelog_versions()[0]}"
    )


def test_the_version_is_not_zero_zero_zero():
    """0.0.0 is the placeholder setuptools accepts and nobody means."""
    assert _declared_version() != "0.0.0"
