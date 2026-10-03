"""scripts/deploy_space.sh copies a hardcoded list of files to the Space.

Anything the app needs at runtime and the list omits produces a Space that
builds and then crashes on import -- and the list is maintained by hand.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = (ROOT / "scripts" / "deploy_space.sh").read_text()


def _payload():
    block = re.search(r"PAYLOAD=\(\s*(.*?)\s*\)", SCRIPT, re.DOTALL)
    assert block, "PAYLOAD array not found in the deploy script"
    return [line.strip() for line in block.group(1).split("\n") if line.strip()]


def test_the_payload_entries_all_exist():
    for entry in _payload():
        assert (ROOT / entry).exists(), f"deploy script copies a missing path: {entry}"


def test_the_runtime_essentials_are_shipped():
    payload = set(_payload())
    for required in ("Dockerfile", "requirements.txt", "setup.py", "app.py", "src"):
        assert required in payload, required


def test_the_space_config_is_shipped():
    """Spaces reads sdk and app_port from README.md's frontmatter."""
    assert "README.md" in _payload()


def test_everything_the_app_imports_is_covered():
    """A new top-level module would otherwise be left behind silently."""
    app_py = (ROOT / "app.py").read_text()
    local_imports = set(re.findall(r"^from (\w+)[. ]", app_py, re.MULTILINE))
    local_imports -= {"dotenv", "flask", "langchain", "langchain_core", "typing"}
    local_imports = {
        m
        for m in local_imports
        if (ROOT / m).exists() or (ROOT / f"{m}.py").exists()
    }
    payload = set(_payload())
    for module in local_imports:
        assert module in payload or f"{module}.py" in payload, module


def test_the_templates_and_static_assets_are_shipped():
    payload = set(_payload())
    assert "templates" in payload and "static" in payload


def test_the_source_pdf_is_not_shipped():
    """15MB, unused at runtime, and over the Hub's non-LFS file limit."""
    assert "Data" not in _payload()


def test_the_env_file_is_not_shipped():
    assert ".env" not in _payload()
