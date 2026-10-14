"""The lint everyone runs must be the lint CI runs.

ruff was chosen independently in three places and the pre-commit hook's copy
was seven minor releases behind the rest, which means the commit hook could
pass code CI rejects. These assert the arrangement that fixed it stays intact,
because the natural way to add a hook is to paste a pinned mirror back in.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRE_COMMIT = (ROOT / ".pre-commit-config.yaml").read_text()
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
MAKEFILE = (ROOT / "Makefile").read_text()
DEV_REQUIREMENTS = (ROOT / "requirements-dev.txt").read_text()


def test_ruff_is_declared_as_a_dev_requirement():
    assert re.search(r"^ruff[><=]", DEV_REQUIREMENTS, re.MULTILINE)


def test_the_hook_does_not_pin_its_own_ruff():
    """A mirror rev is a second version that drifts from the first."""
    assert "ruff-pre-commit" not in PRE_COMMIT
    assert re.search(r"entry: ruff check", PRE_COMMIT)
    assert "language: system" in PRE_COMMIT


def test_ci_installs_ruff_from_the_requirements():
    assert "requirements-dev.txt ruff" not in WORKFLOW, "installed ad hoc"
    assert "pip install -r requirements-dev.txt" in WORKFLOW


def test_every_runner_of_ruff_invokes_the_same_command():
    for source, name in ((WORKFLOW, "CI"), (MAKEFILE, "Makefile")):
        assert "ruff check ." in source, name


def test_mypy_is_declared_and_not_installed_ad_hoc():
    assert re.search(r"^mypy[><=]", DEV_REQUIREMENTS, re.MULTILINE)
    assert "install mypy" not in WORKFLOW


def test_no_tool_version_is_named_twice():
    """Two statements of one version is one of them going stale."""
    for tool in ("ruff", "mypy", "pytest"):
        declarations = re.findall(rf"^{tool}[><=]\S+", DEV_REQUIREMENTS, re.MULTILINE)
        assert len(declarations) == 1, f"{tool} declared {len(declarations)} times"
