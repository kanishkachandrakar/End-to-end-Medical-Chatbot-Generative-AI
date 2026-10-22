"""Every module this project ships is measured, or excluded for a reason.

store_index.py -- 99 statements, the largest file outside src/ -- sat outside
the coverage source for weeks. It was fully covered by its tests the whole
time, so nothing looked wrong: the figure said 100% and the floor was met,
about two thirds of the code.

That is the failure this file prevents. A new module added to src/ is picked up
by the directory entry; a new top-level script is not, and has to be added to
the source or named here with its reason.
"""

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Shipped, deliberately not measured. The reasons are in pyproject.toml beside
# the source list; repeated in one line each here because this is where the
# failure will be read.
UNMEASURED = {
    # Its filename is not an importable module name, so coverage cannot take
    # it as a source entry. test_deploy_config.py executes it instead.
    "gunicorn.conf.py",
    # Run by pip, never by this project.
    "setup.py",
    # A one-shot scaffold that finished before the first commit.
    "template.py",
}


def _shipped_modules():
    """Tracked .py files that are not tests."""
    listing = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return {name for name in listing if not name.startswith("tests/")}


def _coverage_source():
    config = (ROOT / "pyproject.toml").read_text()
    block = re.search(r"^source = \[(.*?)\]", config, re.MULTILINE | re.DOTALL)
    assert block, "no coverage source list found in pyproject.toml"
    return set(re.findall(r'"([^"]+)"', block.group(1)))


def _is_measured(name, source):
    """Whether a source entry covers this file: a directory or a module name."""
    if name.rsplit(".py", 1)[0] in source:
        return True
    return any(name.startswith(f"{entry}/") for entry in source)


def test_both_sides_were_read():
    """Everything below compares two sets; an empty one would pass it all."""
    assert len(_shipped_modules()) >= 15
    assert _coverage_source()


def test_every_shipped_module_is_measured_or_named():
    source = _coverage_source()
    unaccounted = sorted(
        name
        for name in _shipped_modules()
        if not _is_measured(name, source) and name not in UNMEASURED
    )
    assert not unaccounted, (
        f"{unaccounted} are shipped but neither measured nor listed as "
        "unmeasured -- add them to the coverage source, or to UNMEASURED with "
        "the reason, as store_index.py should have been"
    )


def test_nothing_is_listed_unmeasured_that_is_measured():
    """Then the exclusion is wrong and the reason beside it misleads."""
    source = _coverage_source()
    wrong = sorted(name for name in UNMEASURED if _is_measured(name, source))
    assert not wrong, f"listed as unmeasured but in the source: {wrong}"


def test_the_unmeasured_files_still_exist():
    missing = sorted(name for name in UNMEASURED if not (ROOT / name).exists())
    assert not missing, f"listed as unmeasured but absent: {missing}"


def test_the_two_lists_agree_with_pyproject():
    """The reasons live in both places; the names must at least match."""
    config = (ROOT / "pyproject.toml").read_text()
    for name in UNMEASURED:
        assert name in config, f"{name} is excluded here but unexplained in pyproject"
