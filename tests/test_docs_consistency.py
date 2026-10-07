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
    commands = ("ruff check .", "node --check static/chat.js", "--cov-fail-under=95")
    for command in commands:
        assert command in makefile, f"Makefile omits {command!r}"
        assert command in workflow, f"CI omits {command!r}"


def test_every_makefile_target_is_documented():
    """`make help` reads the ## comments, so an undocumented target is hidden."""
    makefile = (ROOT / "Makefile").read_text()
    declared = set(re.findall(r"^\.PHONY: (.+)$", makefile, re.MULTILINE)[0].split())
    documented = set(re.findall(r"^([a-z-]+):.*?## ", makefile, re.MULTILINE))
    assert declared == documented, f"undocumented: {declared - documented}"
