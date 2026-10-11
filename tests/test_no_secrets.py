"""No tracked file may contain something shaped like an API key.

This repository has already leaked both of its keys into a public history, and
SECURITY.md tells the owner to rotate them. The point of this test is that it
not happen a second time: it fails on the commit that would introduce a key,
rather than on the day someone notices.

It scans what git tracks, not the working tree -- .env is deliberately present
and deliberately ignored.
"""

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The prefixes the two providers use, plus OpenAI's, which earlier versions of
# this project also held. Long enough to need a body, so a prose mention of
# "gsk_" in a comment does not trip it.
KEY_PATTERNS = (
    re.compile(r"pcsk_[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}"),
)

# Files that discuss key formats rather than containing keys.
ALLOWED = {"tests/test_no_secrets.py", "tests/test_env.py", "src/env.py"}


def _tracked_text_files():
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for name in listing.split("\0"):
        if not name or name in ALLOWED:
            continue
        path = ROOT / name
        binary = {".pdf", ".jpeg", ".jpg", ".png"}
        if not path.is_file() or path.suffix.lower() in binary:
            continue
        yield name, path


def test_there_are_files_to_scan():
    """Everything below is an absence check; an empty listing would pass."""
    assert len(list(_tracked_text_files())) > 15


def test_no_tracked_file_contains_an_api_key():
    offenders = []
    for name, path in _tracked_text_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for pattern in KEY_PATTERNS:
            if pattern.search(content):
                offenders.append(f"{name} matches {pattern.pattern}")
    assert not offenders, "possible API keys in tracked files: " + "; ".join(offenders)


def test_the_env_file_is_not_tracked():
    tracked = {name for name, _ in _tracked_text_files()}
    assert ".env" not in tracked


def test_the_patterns_would_actually_catch_a_key():
    """A test that cannot fail is worse than no test."""
    samples = (
        "pcsk_" + "a" * 40,
        "gsk_" + "b" * 40,
        "sk-" + "c" * 40,
    )
    for sample in samples:
        assert any(p.search(sample) for p in KEY_PATTERNS), sample


def test_the_patterns_do_not_fire_on_prose():
    """SECURITY.md and comments mention these prefixes; that must stay fine."""
    for harmless in ("keys start with pcsk_", "a gsk_ prefix", "sk-... redacted"):
        assert not any(p.search(harmless) for p in KEY_PATTERNS), harmless
