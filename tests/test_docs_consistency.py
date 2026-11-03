"""DEPLOY.md's settings table is maintained by hand against src/config.py."""

import pathlib
import re

import pytest

# A child that never exits must not become a suite that never finishes.
SUBPROCESS_TIMEOUT = 120

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEPLOY = (ROOT / "DEPLOY.md").read_text()
README = (ROOT / "README.md").read_text()
CONFIG = (ROOT / "src" / "config.py").read_text()

SUPPORTED = set(re.findall(r'os\.environ\.get\(\s*"([A-Z][A-Z0-9_]*)"', CONFIG))
SECRETS = {"PINECONE_API_KEY", "GROQ_API_KEY"}


def _table_variables():
    """Variable names from the backticked first column of DEPLOY.md's table."""
    return set(re.findall(r"^\| `([A-Z][A-Z0-9_]*)`", DEPLOY, re.MULTILINE))


def test_the_table_is_not_empty():
    """If the regex stops matching, the rest of this file silently passes."""
    assert len(_table_variables()) >= 4


def test_every_documented_variable_exists():
    invented = _table_variables() - SUPPORTED - SECRETS
    assert not invented, f"DEPLOY.md documents settings nothing reads: {invented}"


def test_the_documented_defaults_match_the_code():
    """A wrong default in the table is worse than no table."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src import config

    rows = re.findall(r"^\| `([A-Z][A-Z0-9_]*)` \| `([^`]+)` \|", DEPLOY, re.MULTILINE)
    assert rows, "no default values parsed out of the table"
    for name, documented in rows:
        attribute = "INDEX_NAME" if name == "PINECONE_INDEX" else name
        actual = getattr(config, attribute)
        # Compare numerically where both sides are numbers, so documenting a
        # 60-second timeout as `60` is not a failure against a float 60.0.
        try:
            same = float(documented) == float(actual)
        except (TypeError, ValueError):
            same = str(actual) == documented
        assert same, f"{name}: table says {documented}, code gives {actual}"


def test_both_documents_name_the_same_secrets():
    for secret in SECRETS:
        assert secret in DEPLOY, f"{secret} missing from DEPLOY.md"
        assert secret in README, f"{secret} missing from README.md"


def test_the_readme_points_at_the_deployment_guide():
    assert "DEPLOY.md" in README


def test_the_rate_limit_is_explained_where_it_is_documented():
    """Its being global rather than per-visitor is the surprising part."""
    assert "RATE_LIMIT_PER_MINUTE" in DEPLOY
    assert "global" in DEPLOY


def _indexer_flags():
    """Flags argparse actually defines, read out of store_index.py's source."""
    source = (ROOT / "store_index.py").read_text()
    return set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', source))


def test_the_flags_are_found():
    assert len(_indexer_flags()) >= 3


def test_every_indexer_flag_is_documented():
    undocumented = {f for f in _indexer_flags() if f"`{f}" not in README}
    assert not undocumented, (
        f"store_index.py accepts flags the README omits: {undocumented}"
    )


def test_the_readme_documents_no_flag_that_does_not_exist():
    documented = set(re.findall(r"^\| `(--[a-z-]+)", README, re.MULTILINE))
    assert documented <= _indexer_flags(), f"invented: {documented - _indexer_flags()}"


def test_the_deployment_guide_uses_recreate_for_the_rebuild():
    """A plain re-run cannot clear ids this script no longer generates."""
    assert "--recreate" in DEPLOY


def test_the_makefile_runs_what_ci_runs():
    """A local `make check` that diverges from CI is worse than no shortcut."""
    makefile = (ROOT / "Makefile").read_text()
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    commands = (
        "ruff check .",
        "mypy src app.py store_index.py",
        "node --check static/chat.js",
        "bash -n scripts/deploy_space.sh",
        # `bash -n` catches a syntax error; every other finding in a shell
        # script is shellcheck's, and only CI was running it.
        "shellcheck --severity=error scripts/deploy_space.sh",
        "compileall",
        # The Makefile runs `python -m pre_commit` so that PYTHON= works; CI
        # uses the console script. Compare the operation, not the spelling.
        "run --all-files",
        "--cov-fail-under=95",
    )
    for command in commands:
        assert command in makefile, f"Makefile omits {command!r}"
        assert command in workflow, f"CI omits {command!r}"


def test_the_coverage_floor_is_actually_enforced():
    """--cov-fail-under does nothing unless coverage is collecting.

    pytest-cov accepts the floor either way and exits 0, so a CI step that
    had lost --cov would still report the suite as passing and would simply
    stop checking coverage. --cov used to come from addopts, where it broke
    single-file selection; moving it out put it on these two commands, and
    this is what stops it being dropped from one of them.
    """
    makefile = (ROOT / "Makefile").read_text()
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for name, text in (("Makefile", makefile), ("CI", workflow)):
        # Lines that run pytest, not lines that mention the flag: the
        # Makefile carries a comment explaining this rule, and an earlier
        # version of this test was satisfied by that comment rather than by
        # the command -- which is the failure it exists to catch.
        lines = [
            line
            for line in text.splitlines()
            if "pytest" in line and "--cov-fail-under" in line
        ]
        assert lines, f"{name} has no coverage gate at all"
        for line in lines:
            assert "--cov " in line or "--cov=" in line, f"{name}: {line.strip()}"


def test_the_coverage_flag_never_precedes_a_path():
    """A bare --cov swallows the next argument if it is not an option.

    That is how `pytest tests/test_one_thing.py` came to run the whole
    suite. Any invocation in the repo that writes --cov must follow it with
    another flag.
    """
    import re

    for relative in ("Makefile", ".github/workflows/ci.yml", "README.md"):
        for line in (ROOT / relative).read_text().splitlines():
            # Only the lines that run it; a prose mention of the flag is not
            # an invocation, and the Makefile has one explaining this rule.
            if "pytest" not in line:
                continue
            for match in re.finditer(r"--cov(?![-=\w])\s+(\S+)", line):
                following = match.group(1)
                assert following.startswith("-"), f"{relative}: {line.strip()}"


def test_the_image_target_runs_what_the_image_job_runs():
    """The same argument as the check job, for the half that was exempt.

    `make image` checked one of the six things CI checks against the built
    image, so a local run could pass while CI failed on the size budget, a
    deprecation under -W error, or the gunicorn config inside the container.
    """
    makefile = (ROOT / "Makefile").read_text()
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for fragment in (
        "docker image inspect",          # the size budget
        "-W error",                      # the two import paths
        "embed_query",                   # the embedding dimension
        "gunicorn.conf.py",              # the config inside the container
    ):
        assert fragment in makefile, f"make image omits {fragment!r}"
        assert fragment in workflow, f"CI omits {fragment!r}"


def test_the_image_budget_is_the_same_number_in_both():
    """A local budget looser than CI's teaches the wrong lesson."""
    import re

    makefile = (ROOT / "Makefile").read_text()
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    pattern = r"(\d+) \* 1024 \* 1024 \* 1024"
    assert re.search(pattern, makefile), "no budget in the Makefile"
    local = re.search(pattern, makefile).group(1)
    ci = re.search(pattern, workflow).group(1)
    assert local == ci, f"Makefile budget {local}GB, CI {ci}GB"


def test_both_lint_the_dockerfile():
    """Equivalent operations, two spellings.

    CI calls it through docker/build-push-action with `call: check`; the
    Makefile uses the CLI, which is what a developer has. Comparing the
    literal command would fail on the difference rather than on a divergence.
    """
    makefile = (ROOT / "Makefile").read_text()
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "buildx build --check" in makefile
    assert "call: check" in workflow


def test_check_depends_on_every_local_gate():
    """`make check` is the one target anyone is told to run before pushing."""
    import re

    makefile = (ROOT / "Makefile").read_text()
    target = re.search(r"^check: (.+?)(?:  ##|$)", makefile, re.MULTILINE)
    assert target, "no check target"
    assert set(target.group(1).split()) == {"lint", "hooks", "test"}


def test_every_makefile_target_is_documented():
    """`make help` reads the ## comments, so an undocumented target is hidden."""
    makefile = (ROOT / "Makefile").read_text()
    declared = set(re.findall(r"^\.PHONY: (.+)$", makefile, re.MULTILINE)[0].split())
    # [a-z0-9-] rather than [a-z-]: test312 was invisible to this and to the
    # help target's own grep, both of which had no digits in their patterns.
    documented = set(re.findall(r"^([a-z0-9-]+):.*?## ", makefile, re.MULTILINE))
    assert declared == documented, f"undocumented: {declared - documented}"


def test_contributing_points_at_the_commands_that_exist():
    """A setup guide naming a target that was renamed is worse than none."""
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    makefile = (ROOT / "Makefile").read_text()
    for target in re.findall(r"`make ([a-z-]+)`", contributing):
        assert re.search(rf"^{target}:", makefile, re.MULTILINE), f"no make {target}"


def test_the_readme_links_the_contributing_guide():
    assert "CONTRIBUTING.md" in README


def test_the_documented_coverage_core_matches_the_makefile():
    """Both documents promise this behaviour; the Makefile has to provide it."""
    makefile = (ROOT / "Makefile").read_text()
    assert "COVERAGE_CORE" in makefile
    assert "sysmon" in makefile
    for document in (README, (ROOT / "CONTRIBUTING.md").read_text()):
        assert "COVERAGE_CORE" in document


def test_ci_sets_the_coverage_core_too():
    """Otherwise CI pays the overhead the docs say has been removed.

    No ctrace here any more. It was handed to the 3.10 leg by name, because
    3.10 has no sys.monitoring; every version in the matrix has it now. The
    Makefile still chooses between the two, because a local interpreter can
    be older than anything CI tests.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "COVERAGE_CORE" in workflow
    assert "sysmon" in workflow


def test_the_makefile_still_chooses_a_coverage_core_by_version():
    """CI can assume sys.monitoring; a developer's interpreter cannot."""
    makefile = (ROOT / "Makefile").read_text()
    assert "sysmon" in makefile and "ctrace" in makefile
    assert "version_info" in makefile, "the choice should be computed, not fixed"


def test_the_slow_marker_is_used_and_registered():
    """`make fast` deselects by this marker, so it has to exist and be declared.

    Detected by the module-level declaration rather than any mention of the
    name, so that this file -- which mentions it -- is not counted.
    """
    import re

    declared = [
        path.name
        for path in (ROOT / "tests").glob("test_*.py")
        if re.search(r"^pytestmark = pytest\.mark\.slow", path.read_text(), re.M)
    ]
    assert len(declared) >= 2, f"only {declared} declare the slow marker"

    # --strict-markers is on, so an unregistered marker is an error, not a skip.
    config = (ROOT / "pyproject.toml").read_text()
    assert "--strict-markers" in config
    assert re.search(r'"slow:', config), "the marker is not registered"


def test_the_documents_agree_on_how_to_skip_the_slow_tests():
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    for document in (README, contributing):
        assert "make fast" in document


def test_the_makefile_honours_a_python_override():
    """`make test PYTHON=...` is how anyone with the wrong python on PATH runs it.

    The targets use $(PYTHON) rather than a bare `python` precisely so that
    works, and the documentation tells people to use it. A target that hardcodes
    `python` would ignore the override and fail with a confusing missing-module
    error instead.
    """
    import re

    makefile = (ROOT / "Makefile").read_text()
    # Continuations joined first, so a logical recipe line is matched whole.
    # The exemptions below are per line, and a line ending in `\` put the
    # `docker run` and the interpreter it chooses on different ones -- so a
    # command running inside a container read as a host command ignoring
    # PYTHON=, which is the opposite of what it is.
    joined = re.sub(r"\\\n\s*", " ", makefile)
    recipes = [line for line in joined.splitlines() if line.startswith("\t")]
    assert recipes, "no recipe lines found"

    offenders = [
        line.strip()
        for line in recipes
        if re.search(r"(?<![\w$(./-])python\b", line)
        and "$(PYTHON)" not in line
        and "docker" not in line
        and "node" not in line
    ]
    assert not offenders, (
        f"these recipe lines call python directly and ignore PYTHON=: {offenders}"
    )


def test_the_python_variable_is_overridable():
    """`?=` rather than `=`, or the override on the command line is ignored."""
    import re

    makefile = (ROOT / "Makefile").read_text()
    assert re.search(r"^PYTHON \?=", makefile, re.MULTILINE), (
        "PYTHON must be declared with ?= so a command-line value wins"
    )


def test_the_structure_tree_lists_every_module():
    """The README's layout section is a hand-kept list like the others.

    The dockerignore and sunset-import checks each require their module list
    to be complete, and both have caught omissions. This one did not exist, so
    five of fourteen modules were missing from it before anyone noticed.
    """
    modules = {
        path.name
        for path in (ROOT / "src").glob("*.py")
        if path.name != "__init__.py"
    }
    missing = sorted(name for name in modules if name not in README)
    assert not missing, f"absent from the README's structure tree: {missing}"


# The other directories the tree describes file by file. src/ is handled
# above; these two were listed and never checked, which is how static/ came
# to omit chat.js -- the whole front end -- while naming the stylesheet and
# the avatar beside it.
LISTED_DIRECTORIES = {
    "static": ("*.js", "*.css", "*.jpeg"),
    "scripts": ("*.sh", "*.py"),
}


def test_the_tree_lists_every_file_in_the_directories_it_itemises():
    """A tree that names two of three files reads as naming all three."""
    missing = []
    for directory, patterns in LISTED_DIRECTORIES.items():
        for pattern in patterns:
            for path in (ROOT / directory).glob(pattern):
                if path.name not in README:
                    missing.append(f"{directory}/{path.name}")
    assert not missing, f"absent from the README's structure tree: {sorted(missing)}"


def test_those_directories_are_not_empty():
    """So the check above is comparing against something."""
    for directory, patterns in LISTED_DIRECTORIES.items():
        found = [p for pattern in patterns for p in (ROOT / directory).glob(pattern)]
        assert len(found) >= 2, (directory, found)


def test_every_tree_entry_describes_something_that_exists():
    """And the other direction: a tree naming a deleted module misleads too."""
    import re

    tree = re.search(r"```\n\.\n(.*?)```", README, re.DOTALL)
    assert tree, "no structure tree found in the README"
    # The stem may contain a dot: gunicorn.conf.py was read as "conf.py" by a
    # pattern that stopped at the first one, so adding it to the tree made
    # this test report a file that does not exist.
    named = re.findall(r"([a-z_][a-z_.]*\.py)", tree.group(1))
    assert named, "no python files named in the tree"
    # Anywhere git tracks it, not just the root and src/: the tree itemises
    # scripts/ too, and check_no_keys.py living there was reported as a file
    # that does not exist.
    import subprocess

    tracked = {
        pathlib.PurePosixPath(line).name
        for line in subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=SUBPROCESS_TIMEOUT,
        ).stdout.split()
    }
    for name in named:
        found = name in tracked
        assert found, f"the tree names {name}, which does not exist"


def test_the_local_310_run_uses_the_pinned_interpreter():
    """One 3.10 in this repository, not two that can drift.

    `make test312` reads the image out of the Dockerfile rather than naming
    a digest of its own, so the interpreter the suite is checked against
    locally is the one the container is built on.
    """
    import re

    makefile = (ROOT / "Makefile").read_text()
    assert re.search(r"^PY3\d\d_IMAGE := \$\(shell grep", makefile, re.MULTILINE), (
        "the local run should derive its image from the Dockerfile, not hardcode one"
    )
    # An actual digest, not the word: the shell pattern that extracts it
    # contains "sha256:[0-9a-f]+", which is a regex and not a pin.
    assert not re.search(r"sha256:[0-9a-f]{12,}", makefile), (
        "a digest pinned in two places is two pins"
    )


def test_the_local_310_run_matches_the_matrix_floor():
    """It exists to stand in for the 3.10 leg, so it has to be that version."""
    import re

    makefile = (ROOT / "Makefile").read_text()
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()

    matrix = re.search(r"python-version: \[(.+?)\]", workflow)
    assert matrix, "no version matrix in the workflow"
    versions = re.findall(r"\d+\.\d+", matrix.group(1))
    floor = min(versions, key=lambda v: tuple(int(p) for p in v.split(".")))

    # Backslashes stripped: the Makefile spells the image inside a grep
    # pattern, where the dot is escaped.
    unescaped = makefile.replace("\\", "")
    assert f"python:{floor}-slim" in dockerfile, (floor, "Dockerfile")
    assert f"python:{floor}-slim" in unescaped, (floor, "Makefile")
    assert f"test{floor.replace('.', '')}" in makefile, floor


def test_contributing_says_to_run_the_310_suite():
    """The two bugs it exists for both reached a push; the habit is the fix."""
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    assert "make test312" in contributing


def test_no_recipe_comment_is_printed_to_whoever_runs_it():
    """make echoes every recipe line it is not told to suppress.

    These comments are written for someone reading the Makefile, and make
    read them out to anyone running the command: `make clean` opened with
    two lines about .mypy_cache before deleting anything, and `make lint`
    printed six lines of prose about shellcheck in the middle of a run,
    where output is what the reader is scanning for.

    `@#` keeps the comment next to the command it explains, which is the
    point of having it there, and stops it being echoed.
    """
    import re

    offenders = [
        line.strip()
        for line in (ROOT / "Makefile").read_text().splitlines()
        if re.match(r"^\t\s*#", line)
    ]
    assert not offenders, f"recipe comments make will echo: {offenders}"


def test_the_silenced_comments_are_still_there():
    """So the fix is suppression rather than deletion."""
    import re

    makefile = (ROOT / "Makefile").read_text()
    silenced = re.findall(r"^\t\s*@#", makefile, re.MULTILINE)
    assert len(silenced) > 8, len(silenced)


@pytest.mark.slow
def test_the_documented_split_between_fast_and_slow_still_holds():
    """Both files say most of the suite's time is the slow-marked files.

    An unmarked test that takes seconds would make that false and nothing
    would notice -- the number in the prose was "five seconds" for long
    enough to become four times wrong, because a timing claim ages without
    anything editing it. This does not check the durations, which vary by
    machine; it checks the shape the prose describes: that there is a
    substantial body of slow-marked tests to deselect.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "not slow",
            "--collect-only",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    # Without -q: the quiet form prints per-file counts and no total, so the
    # deselected number is not in it to be read.
    deselected = re.search(r"(\d+) deselected", result.stdout)
    assert deselected, result.stdout[-500:]
    assert int(deselected.group(1)) >= 20, (
        "the prose says deselecting the slow files is most of the saving; "
        f"only {deselected.group(1)} are marked"
    )


def test_no_document_still_claims_five_seconds():
    """The stale number, named so it cannot come back by copy-paste."""
    for name in ("README.md", "CONTRIBUTING.md", ".pre-commit-config.yaml"):
        text = (ROOT / name).read_text()
        for line in text.splitlines():
            if "five seconds" in line:
                assert "once" in line or "said so" in line, f"{name}: {line.strip()}"


def test_ci_checks_the_pin_against_its_own_tag():
    """`FROM python:3.12-slim@sha256:...` -- docker reads the digest only.

    The tag is decoration, so the two halves of that line can disagree and
    the image will happily run an interpreter the Dockerfile does not name,
    the matrix does not test, and python_requires may not allow. Both halves
    were typed by hand the day the floor moved off end-of-life 3.10, which is
    exactly when a mismatch would have been introduced.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "the version the tag claims" in workflow
    assert "sys.version_info" in workflow


def test_the_shell_in_that_step_extracts_the_version_the_same_way():
    """The check and the Makefile read the same line; they should agree on how.

    If one matched `python:3\\.12-slim` and the other `python:3\\.[0-9]+-slim`,
    bumping the floor would silently stop one of them finding anything.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    assert "python:3\\.[0-9]+-slim@sha256:[0-9a-f]+" in workflow, (
        "CI should match any 3.x slim pin, not one hardcoded version"
    )
    assert "slim@sha256:[0-9a-f]+" in makefile


def test_the_size_budget_leaves_headroom_rather_than_a_factor_of_four():
    """A budget set against an older, larger image stops detecting anything.

    The guard exists to catch the CPU torch wheel being replaced by the CUDA
    one, which the Dockerfile puts at about 2.5GB on its own. The budget was
    3GB, set when the image was 2.3GB. Moving the base to 3.12 took the image
    to 526MB -- at which point a CUDA torch could return, land somewhere near
    3GB, and pass.

    So the budget has to stay within a small multiple of the real size. This
    does not build the image; it checks the two places that state the budget
    agree and that the number is the recalibrated one.
    """
    import re

    makefile = (ROOT / "Makefile").read_text()
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    budgets = set()
    for text in (makefile, workflow):
        found = re.findall(r"(\d+) \* 1024 \* 1024 \* 1024", text)
        assert found, "no size budget expressed in GB"
        budgets.update(int(value) for value in found)
    assert len(budgets) == 1, f"the two budgets disagree: {sorted(budgets)}"
    budget = budgets.pop()
    assert budget <= 2, (
        f"the budget is {budget}GB against a 526MB image; a CUDA torch would "
        "fit under it"
    )
