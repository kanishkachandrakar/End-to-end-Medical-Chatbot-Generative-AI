"""`.env.example` is the only place the optional settings are discoverable."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Read by the modules that need them, never by src/config.py.
SECRETS = {"PINECONE_API_KEY", "GROQ_API_KEY"}
# Read directly from os.environ in app.py's __main__ block.
DIRECT = {"FLASK_DEBUG"}


def _documented():
    text = (ROOT / ".env.example").read_text()
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, re.MULTILINE))


def _supported():
    text = (ROOT / "src" / "config.py").read_text()
    return set(re.findall(r'os\.environ\.get\(\s*"([A-Z][A-Z0-9_]*)"', text))


def test_every_setting_is_documented():
    missing = _supported() - _documented()
    assert not missing, f"settings exist but .env.example never mentions them: {missing}"


def test_nothing_documented_that_is_not_read():
    invented = _documented() - _supported() - SECRETS - DIRECT
    assert not invented, f".env.example promises settings nothing reads: {invented}"


def test_the_two_secrets_are_listed_uncommented():
    """They are required, so they must be ready to fill in, not commented out."""
    lines = (ROOT / ".env.example").read_text().splitlines()
    uncommented = {ln.split("=", 1)[0].strip() for ln in lines if "=" in ln and not ln.startswith("#")}
    assert SECRETS <= uncommented


def test_optional_settings_are_commented_out():
    """Copying the file must not change any default."""
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            name, value = (part.strip() for part in line.split("=", 1))
            assert value == "", f"{name} ships with a value, which overrides the default"
