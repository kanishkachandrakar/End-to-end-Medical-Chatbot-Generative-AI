"""DEPLOY.md's settings table is maintained by hand against src/config.py."""

import pathlib
import re

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
        "compileall",
        # The Makefile runs `python -m pre_commit` so that PYTHON= works; CI
        # uses the console script. Compare the operation, not the spelling.
        "run --all-files",
        "--cov-fail-under=95",
    )
    for command in commands:
        assert command in makefile, f"Makefile omits {command!r}"
        assert command in workflow, f"CI omits {command!r}"


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
    documented = set(re.findall(r"^([a-z-]+):.*?## ", makefile, re.MULTILINE))
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
    """Otherwise CI pays the overhead the docs say has been removed."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "COVERAGE_CORE" in workflow
    assert "sysmon" in workflow and "ctrace" in workflow


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
    recipes = [line for line in makefile.splitlines() if line.startswith("\t")]
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
