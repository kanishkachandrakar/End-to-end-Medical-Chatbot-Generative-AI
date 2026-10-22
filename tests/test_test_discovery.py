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


# Tests whose assertion is that nothing raised. Each is listed because the
# alternative -- a test that calls code and checks nothing by accident -- looks
# exactly the same from here, and two of those have already shipped.
ASSERTS_BY_NOT_RAISING = {
    # _validate() raises on a bad setting; the committed defaults must not.
    "test_config_validation.py::test_the_shipped_defaults_are_valid",
    # importlib.import_module raises if the module is gone.
    "test_imports.py::test_every_plain_import_resolves",
    # confirm_upsert must not raise on a count that is merely still settling.
    "test_store_index.py::test_it_does_not_raise_on_a_short_count",
    # confirm_deletion returns silently when the right name is typed.
    "test_store_index.py::test_the_index_name_confirms",
}


def _tests_without_a_visible_assertion(filename, source):
    """test_* functions containing no assert, raises or warns."""
    import ast

    found = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        body = ast.unparse(node)
        if "assert" in body or "raises" in body or "warns" in body:
            continue
        found.append(f"{filename}::{node.name}")
    return found


def test_every_test_asserts_something_or_is_listed():
    """A test that checks nothing is a claim that something was checked.

    Some legitimately assert by not raising, and those are named above with
    the reason. Anything else here is a test that runs code and examines
    none of it -- which is how two tests came to sit in this suite asserting
    nothing while appearing to cover a fix.
    """
    silent = []
    for name in _python_files():
        if name in SUPPORTING:
            continue
        silent.extend(
            _tests_without_a_visible_assertion(name, (TESTS / name).read_text())
        )
    unexplained = sorted(set(silent) - ASSERTS_BY_NOT_RAISING)
    assert not unexplained, (
        f"these tests assert nothing: {unexplained} -- add an assertion, or "
        "list it in ASSERTS_BY_NOT_RAISING with the reason"
    )


def test_the_listed_ones_still_exist():
    """A justification for a test that has been renamed or deleted is noise."""
    present = set()
    for name in _python_files():
        if name in SUPPORTING:
            continue
        import ast

        tree = ast.parse((TESTS / name).read_text())
        present.update(
            f"{name}::{node.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        )
    stale = ASSERTS_BY_NOT_RAISING - present
    assert not stale, f"listed but no longer present: {stale}"


def test_the_silence_detector_finds_a_planted_one():
    planted = "def test_checks_nothing():\n    compute_something()\n"
    assert _tests_without_a_visible_assertion("x.py", planted) == [
        "x.py::test_checks_nothing"
    ]


def test_the_silence_detector_accepts_a_raises_block():
    """pytest.raises is an assertion, just not spelled assert."""
    allowed = (
        "def test_it_refuses():\n"
        "    with pytest.raises(ValueError):\n"
        "        thing()\n"
    )
    assert _tests_without_a_visible_assertion("x.py", allowed) == []


def test_naming_one_file_collects_only_that_file():
    """`pytest tests/test_one_thing.py` must mean that file, and no other.

    It did not. pytest-cov's --cov takes an optional value, so the bare
    --cov that used to sit in addopts consumed the path as its argument:
    with no path left pytest fell back to testpaths and collected the whole
    suite, measuring a file coverage then reported as never imported. The
    symptom was 22 seconds and no coverage report where 0.4 seconds was
    expected -- slow and wrong, but never red, so it went unnoticed for as
    long as the flag had been there.

    Driven through a subprocess because the thing under test is argument
    parsing, which has already happened by the time this runs.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_smoke.py",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    named = {
        line.split(":")[0]
        for line in result.stdout.splitlines()
        if line.startswith("tests/")
    }
    assert named == {"tests/test_smoke.py"}, sorted(named)
