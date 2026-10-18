"""Every test file in tests/ must be one pytest collects.

A file named anything other than test_*.py is silently ignored: no error, no
warning, just a set of assertions that never runs. The suite would report green
while covering less than it did. With `--strict-markers` and warnings as errors
this is the one remaining way to add checks that quietly do nothing.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# Files in tests/ that are deliberately not test modules.
SUPPORTING = {"__init__.py", "conftest.py", "stubs.py"}


def _python_files():
    return sorted(path.name for path in TESTS.glob("*.py"))


def test_there_are_test_files():
    assert len(_python_files()) > 10


def test_every_file_is_collected_or_supporting():
    uncollected = [
        name
        for name in _python_files()
        if name not in SUPPORTING and not name.startswith("test_")
    ]
    assert not uncollected, (
        f"pytest will not collect {uncollected} -- a file in tests/ that is "
        "not named test_*.py runs nothing and reports nothing"
    )


def test_every_collected_file_defines_at_least_one_test():
    """An import-only module collects cleanly and asserts nothing."""
    import re

    empty = []
    for name in _python_files():
        if name in SUPPORTING:
            continue
        source = (TESTS / name).read_text()
        if not re.search(r"^\s*def test_", source, re.MULTILINE):
            empty.append(name)
    assert not empty, f"no test functions found in {empty}"


def test_no_test_function_is_defined_twice_in_a_file():
    """The second definition replaces the first, which then never runs."""
    import ast

    duplicates = {}
    for name in _python_files():
        if name in SUPPORTING:
            continue
        tree = ast.parse((TESTS / name).read_text())
        seen, repeated = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if node.name in seen:
                    repeated.add(node.name)
                seen.add(node.name)
        if repeated:
            duplicates[name] = sorted(repeated)
    assert not duplicates, f"shadowed test functions: {duplicates}"


def _test_functions_pytest_would_skip(source):
    """test_* functions nested in a class pytest does not collect.

    pytest takes test methods only from classes named Test*. A test indented
    one level too far lands inside whatever class precedes it -- a helper, a
    stub -- and is then unreachable. Nothing reports it: no error, no skip, no
    change in the count anyone looks at.
    """
    import ast

    skipped = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name.startswith("Test"):
            continue
        skipped.extend(
            f"{node.name}.{child.name}"
            for child in node.body
            if isinstance(child, ast.FunctionDef) and child.name.startswith("test_")
        )
    return skipped


def test_no_test_hides_inside_a_helper_class():
    """Two tests sat inside a stub class in test_app.py and never ran."""
    hidden = {}
    for name in _python_files():
        if name in SUPPORTING:
            continue
        found = _test_functions_pytest_would_skip((TESTS / name).read_text())
        if found:
            hidden[name] = found
    assert not hidden, (
        f"pytest collects test methods only from Test* classes, so {hidden} "
        "never run"
    )


def test_the_detector_finds_a_planted_one():
    """An absence check that cannot fire is worse than none."""
    planted = (
        "class StubThing:\n"
        "    def invoke(self):\n"
        "        return 1\n"
        "\n"
        "    def test_hidden_away(self):\n"
        "        assert False\n"
    )
    assert _test_functions_pytest_would_skip(planted) == ["StubThing.test_hidden_away"]


def test_a_real_test_class_is_not_flagged():
    allowed = (
        "class TestSomething:\n"
        "    def test_visible(self):\n"
        "        assert True\n"
    )
    assert _test_functions_pytest_would_skip(allowed) == []
