"""What reaches the image, and what does not.

The Dockerfile ends with `COPY . .`, so .dockerignore is the whole of the
answer. Two ways to get it wrong: exclude something the app needs, which fails
at runtime rather than at build, or include something it does not, which
invalidates the layer cache on commits that change nothing the container runs.
"""

import pathlib
import re
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


def _as_regex(pattern):
    """A .dockerignore pattern as a regex, with docker's glob rules.

    `*` does not cross a separator, `**` does, and `**/` may match nothing.
    fnmatch has none of those distinctions, which is where the previous
    version of this went wrong.
    """
    out = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            out.append(r"(?:[^/]+/)*")
            index += 3
        elif pattern.startswith("**", index):
            out.append(r".*")
            index += 2
        elif pattern[index] == "*":
            out.append(r"[^/]*")
            index += 1
        elif pattern[index] == "?":
            out.append(r"[^/]")
            index += 1
        elif pattern[index] == "[":
            close = pattern.index("]", index)
            out.append(pattern[index : close + 1])
            index = close + 1
        else:
            out.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(out) + r"\Z")


def _ignored(path):
    """Would `docker build` leave this path out of the image?

    Patterns are anchored at the context root, and a pattern that matches a
    directory excludes everything under it -- so each prefix of the path is
    matched whole. This used to compare basenames as well, which made
    `__pycache__/` look like it excluded src/__pycache__. It does not, and
    the image had the .pyc files to prove it.
    """
    parts = path.strip("/").split("/")
    prefixes = ["/".join(parts[:count]) for count in range(1, len(parts) + 1)]
    for pattern in _patterns():
        regex = _as_regex(pattern.rstrip("/"))
        if any(regex.match(prefix) for prefix in prefixes):
            return True
    return False


# Measured against docker itself: a throwaway context holding each of these
# was built twice, once with the patterns anchored at the root and once with
# **/, and the second build was the one that excluded all of them.
NESTED_ARTIFACTS = (
    "src/__pycache__/webapp.cpython-312.pyc",
    "src/__pycache__",
    "deep/nest/__pycache__/b.pyc",
    "templates/.DS_Store",
    "static/sub/x.pyo",
)


def test_a_nested_artifact_is_excluded_too():
    """`__pycache__/` is root-anchored; src/__pycache__ is not the root.

    The image was carrying src/__pycache__/*.cpython-312.pyc -- bytecode from
    a 3.12 interpreter, in an image built on 3.10, where the Dockerfile sets
    PYTHONDONTWRITEBYTECODE=1 to keep bytecode out on purpose.
    """
    reaching = [path for path in NESTED_ARTIFACTS if not _ignored(path)]
    assert not reaching, f"copied into the image: {reaching}"


def test_the_root_level_cases_still_match():
    """The ** must not have been bought by breaking the simple case."""
    for path in ("__pycache__/app.pyc", "app.pyc", ".DS_Store", ".env"):
        assert _ignored(path), path


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


# Literal entries that name nothing in a fresh checkout on purpose: they
# exclude what a working copy grows, not what the repo tracks. Patterns
# containing a glob are exempt from that check by shape -- `**/*.py[cod]`
# describes a kind of file, not a path -- which is two fewer entries here
# than when this was a list of strings to excuse.
TRANSIENT = frozenset(
    {
        ".env",
        ".venv/",
        "venv/",
        "env/",
        ".tox/",
        ".ipynb_checkpoints/",
        # The tool caches belong here for the same reason, and listing them
        # is not optional: which of them exists depends on what has been run.
        # A fresh checkout has none, and CI's has some by the time the suite
        # runs and others not yet, so judging them by existence would make
        # this check report the state of the machine.
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".coverage",
        "htmlcov/",
    }
)

GLOB_CHARACTERS = set("*?[")


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
        if pattern not in TRANSIENT
        and not (GLOB_CHARACTERS & set(pattern))
        and not (ROOT / pattern.rstrip("/")).exists()
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


# Everything this project's own tooling writes into a checkout, spelled as
# paths rather than patterns so _ignored sees what `COPY . .` would see.
LOCAL_ARTIFACTS = (
    ".mypy_cache/missing_stubs.json",
    ".pytest_cache/CACHEDIR.TAG",
    ".ruff_cache/0.16.0/entry",
    ".coverage",
    ".coverage.host.12345.XbYz",
    "htmlcov/index.html",
    "src/__pycache__/webapp.cpython-312.pyc",
    ".venv/bin/python",
    "Generative_AI_Project.egg-info/PKG-INFO",
)


def test_no_tool_cache_reaches_the_image():
    """`COPY . .` takes whatever the tooling left behind.

    .mypy_cache was 128MB of it -- larger than the embedding weights the
    image bakes in deliberately -- and .coverage is a SQLite file of absolute
    paths from the last person to run the suite. None of it appeared in CI,
    whose checkout is fresh, so the only images affected were the ones built
    locally to stand in for CI's.

    A fixed list rather than a scan of the working copy, because the working
    copy in CI is clean and a scan there would be checking nothing.
    """
    reaching = [path for path in LOCAL_ARTIFACTS if not _ignored(path)]
    assert not reaching, f"copied into the image: {reaching}"


def test_everything_git_refuses_to_track_stays_out_of_the_image_too():
    """The companion check: one names what is known, this catches the rest.

    A file git will not track is a local artifact by definition, and a
    runtime image has no use for one. This sees only what the checkout
    happens to hold, which is why the fixed list above is also here.
    """
    ignored = [
        line[3:].strip()
        for line in subprocess.run(
            ["git", "status", "--porcelain", "--ignored"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        if line.startswith("!!")
    ]
    reaching = [path for path in ignored if not _ignored(path.rstrip("/"))]
    assert not reaching, f"git ignores these, the image takes them: {reaching}"


def test_make_clean_removes_every_cache_the_dockerignore_names():
    """Two lists of the same artifacts; .mypy_cache was in only one.

    If a cache is worth keeping out of the image it is worth `make clean`
    removing, and the 128MB one was the one that had fallen out.
    """
    clean = next(
        line
        for line in (ROOT / "Makefile").read_text().splitlines()
        if line.strip().startswith("rm -rf")
    )
    for cache in (".mypy_cache", ".pytest_cache", ".ruff_cache", ".coverage"):
        assert cache in clean, f"make clean leaves {cache} behind"
