"""No tracked file may contain something shaped like an API key.

This repository has already leaked both of its keys into a public history, and
SECURITY.md tells the owner to rotate them. The point is that it not happen a
second time.

This file used to say it "fails on the commit that would introduce a key,
rather than on the day someone notices", and SECURITY.md said such a key is
"caught at commit time". Neither was true: the suite is deliberately not a
pre-commit hook, so both described CI, which runs after the commit exists --
and a key in a local commit is in that history permanently whatever the next
commit does. scripts/check_no_keys.py is the hook that makes the claim true,
and the patterns below are imported from it so there is one copy of them.

It scans what git tracks, not the working tree -- .env is deliberately present
and deliberately ignored.
"""

import importlib.util
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _hook_module():
    """The pre-commit hook, loaded by path: scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "check_no_keys", ROOT / "scripts" / "check_no_keys.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HOOK = _hook_module()
KEY_PATTERNS = _HOOK.KEY_PATTERNS
ALLOWED = _HOOK.ALLOWED


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


def test_the_hook_is_installed_in_pre_commit():
    """The patterns are only commit-time cover if something runs them then."""
    config = (ROOT / ".pre-commit-config.yaml").read_text()
    assert "check_no_keys.py" in config
    assert "no-api-keys" in config


def test_the_hook_rejects_a_planted_key(tmp_path):
    """End to end, through the entry point pre-commit calls."""
    import subprocess as sp
    import sys

    planted = tmp_path / "leaked.py"
    planted.write_text('KEY = "pcsk_' + "a" * 40 + '"\n')
    result = sp.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_keys.py"), str(planted)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "looks like it contains an API key" in result.stderr
    assert "rotate it" in result.stderr


def test_the_hook_accepts_a_file_with_no_key(tmp_path):
    """So the check above is not passing on every input."""
    import subprocess as sp
    import sys

    ordinary = tmp_path / "fine.py"
    ordinary.write_text('# keys start with pcsk_ and gsk_\nVALUE = 1\n')
    result = sp.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_keys.py"), str(ordinary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_the_hook_and_the_suite_share_one_set_of_patterns():
    """Two copies would drift, and the drifting one would be the hook."""
    assert KEY_PATTERNS is _HOOK.KEY_PATTERNS
    assert len(KEY_PATTERNS) >= 3
