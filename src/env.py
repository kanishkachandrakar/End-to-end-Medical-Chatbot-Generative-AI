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
