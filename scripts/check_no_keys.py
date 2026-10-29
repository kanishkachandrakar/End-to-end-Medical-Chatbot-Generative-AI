#!/usr/bin/env python3
"""Refuse a file containing something shaped like an API key.

Run as a pre-commit hook, so a key is rejected before the commit exists
rather than by CI after it does. That distinction is the whole point here: a
key in a local commit is in that history permanently, and removing it in the
next commit does not remove it from the one before. This repository has
already leaked both of its keys that way.

tests/test_no_secrets.py imports the patterns below rather than keeping its
own copy, and checks that they catch a key and do not fire on prose.
"""

from __future__ import annotations

import pathlib
import re
import sys

# The prefixes the two providers use, plus OpenAI's, which earlier versions of
# this project also held. Each requires a body long enough that a prose mention
# of the prefix -- which SECURITY.md and src/env.py both make -- does not match.
KEY_PATTERNS = (
    re.compile(r"pcsk_[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}"),
)

# Files that discuss key formats rather than holding keys.
ALLOWED = {
    "scripts/check_no_keys.py",
    "tests/test_no_secrets.py",
    "tests/test_env.py",
    "src/env.py",
}

BINARY_SUFFIXES = {".pdf", ".jpeg", ".jpg", ".png", ".ico", ".safetensors", ".bin"}


def offenders_in(paths: list[str]) -> list[str]:
    """Describe every path whose contents match a key pattern."""
    found = []
    for name in paths:
        if name in ALLOWED:
            continue
        path = pathlib.Path(name)
        if not path.is_file() or path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for pattern in KEY_PATTERNS:
            if pattern.search(content):
                found.append(f"{name} matches {pattern.pattern}")
    return found


def main(argv: list[str]) -> int:
    found = offenders_in(argv)
    if not found:
        return 0
    print("a file in this commit looks like it contains an API key:", file=sys.stderr)
    for line in found:
        print(f"  {line}", file=sys.stderr)
    print(
        "\nIf it is a real key: do not commit it, and rotate it -- it has been "
        "on disk. If it is a sample, add the file to ALLOWED in "
        "scripts/check_no_keys.py with a reason.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
