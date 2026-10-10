"""Reading required environment variables.

Separate from src/config.py, which deliberately reads no secrets so that it
stays importable in tests and CI without a .env.
"""

import os


def require_env(*names: str) -> dict[str, str]:
    """Return the named variables, or raise naming every one that is missing.

    Reporting them together matters for a deploy: finding out about a second
    missing secret only after fixing the first means another build and another
    cold start.
    """
    found = {name: os.environ.get(name) or "" for name in names}
    missing = [name for name, value in found.items() if not value.strip()]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in your keys, or set them as "
            "secrets if you are deploying."
        )
    return found


# Prefixes the providers use. Checked only to catch a swap or a truncation --
# a key with the right prefix can still be revoked, and that is the API's job
# to say, not ours.
KEY_PREFIXES = {
    "PINECONE_API_KEY": "pcsk_",
    "GROQ_API_KEY": "gsk_",
}


def warn_on_suspicious_keys(values: dict[str, str]) -> list[str]:
    """Describe any key whose shape does not match its variable.

    Pasting the Groq key into PINECONE_API_KEY is an easy mistake and an
    expensive one: both are present, both are non-empty, so the app starts and
    then fails on the first request with an authentication error from whichever
    provider got the wrong one. The prefixes make that visible at startup.

    Returns descriptions rather than raising -- a provider is free to change
    its key format, and refusing to start over a prefix would be worse than
    the mistake this catches.
    """
    complaints = []
    for name, expected in KEY_PREFIXES.items():
        value = values.get(name)
        if value and not value.startswith(expected):
            looks_like = [
                other
                for other, prefix in KEY_PREFIXES.items()
                if other != name and value.startswith(prefix)
            ]
            if looks_like:
                complaints.append(
                    f"{name} looks like a {looks_like[0]} -- are the two swapped?"
                )
            else:
                complaints.append(
                    f"{name} does not start with {expected!r}; it may be truncated"
                )
    return complaints
