"""What `pip install -e .` puts into the environment.

requirements.txt ends with `-e .`, so this runs on every install, including
the one inside the image. A bare find_packages() takes every directory with
an __init__.py, and tests/ has one -- deliberately, so the suite can do
`from tests.stubs import StubChain`. The result was a distribution declaring
two top-level packages:

    $ cat *.egg-info/top_level.txt
    src
    tests

which is a package named `tests` installed into the environment, ready to
shadow or be shadowed by any other project's, and ready to be included in a
wheel if this were ever built as one.
"""

import ast
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETUP = (ROOT / "setup.py").read_text()


def _packages_argument():
    """The `packages=` expression from setup.py, as source text."""
    tree = ast.parse(SETUP)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup":
            for keyword in node.keywords:
                if keyword.arg == "packages":
                    return ast.unparse(keyword.value)
    raise AssertionError("setup() has no packages= argument")


def test_find_packages_is_constrained():
    """Static, so it runs wherever the suite does."""
    argument = _packages_argument()
    assert argument != "find_packages()", "a bare find_packages() takes tests/ too"
    assert "include=" in argument or "exclude=" in argument, argument


def test_the_tests_directory_is_a_package_on_purpose():
    """Otherwise the fix would be to delete its __init__.py instead."""
    assert (ROOT / "tests" / "__init__.py").exists()
    assert "from tests." in (ROOT / "tests" / "test_webapp.py").read_text()


def test_nothing_but_src_is_declared_as_a_package():
    """Behavioural: run the expression setup.py actually passes.

    Skipped where setuptools is absent -- a 3.12 venv has none unless
    something asked for it -- so the static check above is the one that
    always runs.
    """
    if importlib.util.find_spec("setuptools") is None:  # pragma: no cover
        pytest.skip("setuptools is not importable here")

    import os

    from setuptools import find_packages

    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        found = eval(_packages_argument(), {"find_packages": find_packages})
    finally:
        os.chdir(previous)

    assert found == ["src"], found
