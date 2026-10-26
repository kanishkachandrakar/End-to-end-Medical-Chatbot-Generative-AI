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
    for required in (
        "Dockerfile",
        "gunicorn.conf.py",
        "requirements.txt",
        "setup.py",
        "app.py",
        "src",
    ):
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


def test_the_payload_covers_everything_the_image_is_built_from():
    """The Space's build context must not be smaller than CI's.

    CI's image job is the only check that the image a Space builds actually
    builds, and it runs against this repository. Anything .dockerignore lets
    into the image that the deploy script leaves behind is a difference
    between the two, which is a difference CI cannot see: it stays green
    while the Space breaks. pyproject.toml was that difference.
    """
    from tests.test_dockerignore import REQUIRED

    payload = set(_payload())
    missing = [
        path
        for path in REQUIRED
        if path not in payload and path.split("/")[0] not in payload
    ]
    assert not missing, f"the image is built from these; the Space is not: {missing}"


def test_the_exclusion_rules_are_shipped():
    """Without .dockerignore the Space's `COPY . .` excludes nothing.

    Measured with a throwaway context: a build with no .dockerignore puts the
    whole .git directory into the image, so every Space deploy was shipping
    its own history inside the image it built. CI never saw it, because this
    repository has the file and the Space's copy did not.
    """
    assert ".dockerignore" in _payload()


def test_nothing_reaches_the_ci_image_that_the_space_s_image_lacks():
    """The two build contexts must produce the same image contents.

    .dockerignore decides what `COPY . .` takes, and it is read in both
    builds now -- but it is applied to different directories: this whole
    repository in CI, and only what the deploy script pushed in the Space.
    A tracked file that .dockerignore permits and the payload omits is in one
    image and not the other, which is a difference CI cannot see.

    Checked in one direction only. README.md is deliberately in the payload
    and deliberately excluded from the image: Spaces reads `sdk` and
    `app_port` from its frontmatter in the repository, and the container has
    no use for it.
    """
    import subprocess

    from tests.test_dockerignore import _ignored

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    payload = set(_payload())

    missing = [
        path
        for path in tracked
        if not _ignored(path)
        and path not in payload
        and path.split("/")[0] not in payload
    ]
    assert not missing, f"in CI's image, not in the Space's: {missing}"
