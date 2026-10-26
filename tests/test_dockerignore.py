"""What reaches the image, and what does not.

The Dockerfile ends with `COPY . .`, so .dockerignore is the whole of the
answer. Two ways to get it wrong: exclude something the app needs, which fails
at runtime rather than at build, or include something it does not, which
invalidates the layer cache on commits that change nothing the container runs.
"""

import fnmatch
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Everything the image is built from: what app.py, gunicorn and the templates
# need at runtime, plus the two files the pip layer installs from. The deploy
# script ships this set too, so that the context a Space builds in matches the
# one CI builds in -- test_deploy_payload.py checks the two against each other.
REQUIRED = (
    "app.py",
    "gunicorn.conf.py",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "src/__init__.py",
    "src/assets.py",
    "src/chunking.py",
    "src/config.py",
    "src/env.py",
    "src/errors.py",
    "src/headers.py",
    "src/helper.py",
    "src/messages.py",
    "src/middleware.py",
    "src/loaders.py",
    "src/prompt.py",
    "src/ratelimit.py",
    "src/text.py",
    "src/webapp.py",
    "templates/chat.html",
    "static/chat.js",
    "static/style.css",
    "static/medicalbotpic.jpeg",
)

# Tracked things the container has no use for.
UNWANTED = (
    "Data/Medical_book.pdf",
    "research/trials.ipynb",
    "tests/test_webapp.py",
    "requirements-dev.txt",
    "tests/conftest.py",
    "scripts/deploy_space.sh",
    ".github/workflows/ci.yml",
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "DEPLOY.md",
    ".editorconfig",
    ".pre-commit-config.yaml",
    ".env",
    # Shipped Python that the container never runs: store_index.py reads the
    # 15MB PDF excluded above and is run from a checkout, and template.py is a
    # scaffold that finished its job before the first commit.
    "store_index.py",
    "template.py",
    "Makefile",
    ".env.example",
    ".gitattributes",
)


def _patterns():
    return [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _ignored(path):
    for pattern in _patterns():
        pattern = pattern.rstrip("/")
        if (
            fnmatch.fnmatch(path, pattern)
            or path.startswith(pattern + "/")
            or fnmatch.fnmatch(pathlib.Path(path).name, pattern)
        ):
            return True
    return False


def test_the_patterns_were_read():
    assert len(_patterns()) > 10


def test_every_runtime_file_reaches_the_image():
    missing = [name for name in REQUIRED if _ignored(name)]
    assert not missing, f"excluded from the image but needed at runtime: {missing}"


def test_the_runtime_files_all_exist():
    """Otherwise the check above passes by naming files that are not there."""
    absent = [name for name in REQUIRED if not (ROOT / name).exists()]
    assert not absent, f"REQUIRED names files that do not exist: {absent}"


def test_nothing_the_container_cannot_use_reaches_it():
    included = [name for name in UNWANTED if not _ignored(name)]
    assert not included, f"copied into the image for no reason: {included}"


def test_every_module_src_contains_is_listed_as_required():
    """A new module would otherwise be untested against .dockerignore."""
    modules = {f"src/{p.name}" for p in (ROOT / "src").glob("*.py")}
    assert modules <= set(REQUIRED), f"not covered: {modules - set(REQUIRED)}"


# Entries that name nothing in a fresh checkout on purpose: they exclude what a
# working copy grows, not what the repo tracks.
TRANSIENT = frozenset(
    {
        ".env",
        ".DS_Store",
        "__pycache__/",
        "*.py[cod]",
        "*.egg-info/",
        ".venv/",
        "venv/",
        "env/",
        ".tox/",
        ".ipynb_checkpoints/",
    }
)


def test_no_pattern_names_something_that_is_no_longer_there():
    """A stale exclusion costs nothing and claims something false.

    This list said `pytest.ini` and `ruff.toml` long after both settled into
    pyproject.toml, so it read as keeping the tooling config out of the image
    while pyproject.toml -- which pip needs for the `-e .` in
    requirements.txt, and which holds all four tools' settings -- went in. The
    check above had grown a conditional to tolerate the first of them.
    """
    stale = [
        pattern
        for pattern in _patterns()
        if pattern not in TRANSIENT and not (ROOT / pattern.rstrip("/")).exists()
    ]
    assert not stale, f"excluded but absent: {stale}"


def test_the_transient_list_holds_no_tracked_path():
    """Otherwise an entry could be excused by being named here."""
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
    )
    for pattern in TRANSIENT:
        assert pattern.rstrip("/") not in tracked, pattern


def test_a_local_virtualenv_is_excluded():
    """200MB of host-platform wheels, and COPY . . would take them."""
    for candidate in (".venv/bin/python", "venv/lib/x.py", ".tox/py310/x"):
        assert _ignored(candidate), candidate


def test_the_tracked_file_list_is_reachable():
    """Sanity: the repo is a git checkout, so UNWANTED reflects reality."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    for name in UNWANTED:
        if name != ".env":
            assert name in tracked, f"{name} is not tracked; update UNWANTED"
