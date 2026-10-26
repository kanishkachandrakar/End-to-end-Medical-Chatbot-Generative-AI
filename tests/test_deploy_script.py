"""Actually run scripts/deploy_space.sh, against a local repository.

It is the code that ships this project, and until SPACES_BASE existed nothing
could execute it -- the Hub host was hardcoded. Every check on it so far has
read it as text.

Marked slow because each case does real git work in a temp directory.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "deploy_space.sh"

pytestmark = pytest.mark.slow


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def space(tmp_path):
    """A bare repository standing in for the Space, plus a clone to inspect."""
    bare = tmp_path / "spaces" / "owner" / "demo.git"
    bare.parent.mkdir(parents=True)
    _git("init", "--quiet", "--bare", "--initial-branch=main", str(bare), cwd=tmp_path)

    # A bare repo has no commits; give it one so `git clone` produces a branch.
    seed = tmp_path / "seed"
    _git("clone", "--quiet", str(bare), str(seed), cwd=tmp_path)
    (seed / "placeholder").write_text("replaced on first deploy\n")
    _git("add", "-A", cwd=seed)
    _git("-c", "user.email=t@example.com", "-c", "user.name=T",
         "commit", "--quiet", "-m", "seed", cwd=seed)
    _git("push", "--quiet", "origin", "main", cwd=seed)
    shutil.rmtree(seed)
    return bare


def _run(space, tmp_path, argument="owner/demo.git"):
    env = {
        **os.environ,
        "SPACES_BASE": str(space.parent.parent),
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(
        ["bash", str(SCRIPT), argument],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def _deployed(space, tmp_path):
    out = tmp_path / "inspect"
    _git("clone", "--quiet", str(space), str(out), cwd=tmp_path)
    return out


def test_it_deploys_and_reports_success(space, tmp_path):
    result = _run(space, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "pushed" in result.stdout


def test_the_runtime_files_arrive(space, tmp_path):
    assert _run(space, tmp_path).returncode == 0
    deployed = _deployed(space, tmp_path)
    for required in ("Dockerfile", "app.py", "requirements.txt", "README.md"):
        assert (deployed / required).is_file(), required
    for directory in ("src", "templates", "static"):
        assert (deployed / directory).is_dir(), directory
    assert (deployed / "src" / "webapp.py").is_file()
    assert (deployed / "static" / "chat.js").is_file()


def test_the_pdf_and_tests_are_left_behind(space, tmp_path):
    """15MB over the Hub's non-LFS limit, and of no use to the container."""
    assert _run(space, tmp_path).returncode == 0
    deployed = _deployed(space, tmp_path)
    assert not (deployed / "Data").exists()
    assert not (deployed / "tests").exists()
    assert not (deployed / ".env").exists()


def test_the_revision_is_stamped_into_the_dockerfile(space, tmp_path):
    """The Space builds the image itself, so there is no --build-arg to pass."""
    assert _run(space, tmp_path).returncode == 0
    dockerfile = (_deployed(space, tmp_path) / "Dockerfile").read_text()
    revision = _git("rev-parse", "--short", "HEAD", cwd=ROOT).strip()
    assert f"ARG APP_REVISION={revision}" in dockerfile


def test_the_commit_message_names_the_revision(space, tmp_path):
    assert _run(space, tmp_path).returncode == 0
    deployed = _deployed(space, tmp_path)
    revision = _git("rev-parse", "--short", "HEAD", cwd=ROOT).strip()
    assert revision in _git("log", "-1", "--pretty=%s", cwd=deployed)


def test_a_second_run_with_nothing_changed_does_nothing(space, tmp_path):
    assert _run(space, tmp_path).returncode == 0
    before = _git("rev-parse", "HEAD", cwd=_deployed(space, tmp_path)).strip()

    again = _run(space, tmp_path)
    assert again.returncode == 0
    assert "already matches" in again.stdout

    shutil.rmtree(tmp_path / "inspect")
    after = _git("rev-parse", "HEAD", cwd=_deployed(space, tmp_path)).strip()
    assert after == before, "an unchanged deploy should not add a commit"


def test_it_refuses_without_an_argument():
    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage" in result.stderr


def test_it_says_how_to_check_the_deploy_landed(space, tmp_path):
    """The revision reporting exists for this question; the script should ask it."""
    result = _run(space, tmp_path)
    assert result.returncode == 0, result.stderr
    revision = _git("rev-parse", "--short", "HEAD", cwd=ROOT).strip()
    assert "hf.space/healthz" in result.stdout
    assert revision in result.stdout


def test_the_app_host_follows_hugging_face_s_scheme(space, tmp_path):
    """A Space at owner/name serves at owner-name.hf.space."""
    result = _run(space, tmp_path, argument="owner/demo.git")
    assert "https://owner-demo.hf.space/healthz" in result.stdout, result.stdout


@pytest.fixture
def space_with_a_stale_module(space, tmp_path):
    """A Space as an earlier deploy would have left it: holding a module
    this repository no longer has."""
    seed = tmp_path / "staleseed"
    _git("clone", "--quiet", str(space), str(seed), cwd=tmp_path)
    (seed / "src").mkdir(exist_ok=True)
    (seed / "src" / "deleted_last_week.py").write_text("stale\n")
    _git("add", "-A", cwd=seed)
    _git("-c", "user.email=t@example.com", "-c", "user.name=T",
         "commit", "--quiet", "-m", "an earlier deploy", cwd=seed)
    _git("push", "--quiet", "origin", "main", cwd=seed)
    shutil.rmtree(seed)
    return space


def test_a_module_this_repo_deleted_does_not_survive_in_the_space(
    space_with_a_stale_module, tmp_path
):
    """`cp -R src dest/` merges into dest/src rather than replacing it.

    So every module ever deployed stayed deployed. Nothing imported them, but
    the Space's build context has no .dockerignore, so they were copied into
    its image -- and this repository splits modules often enough for that to
    be a question of when.
    """
    assert _run(space_with_a_stale_module, tmp_path).returncode == 0
    deployed = _deployed(space_with_a_stale_module, tmp_path)
    assert (deployed / "src" / "webapp.py").is_file(), "the real payload arrived"
    assert not (deployed / "src" / "deleted_last_week.py").exists()


def test_no_bytecode_reaches_the_space(space, tmp_path):
    """cp took the working tree, which has a __pycache__ after any test run.

    Eighteen .pyc files, from two different interpreters, on every deploy --
    and because they change whenever the suite runs, they also made the
    "already matches this checkout" shortcut below stop short-circuiting.
    """
    import subprocess as sp

    # Guarantee the condition rather than relying on a previous test having
    # left bytecode behind.
    sp.run(
        ["python3", "-c", "import compileall; compileall.compile_dir('src', quiet=2)"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert (ROOT / "src" / "__pycache__").is_dir(), "precondition not met"

    assert _run(space, tmp_path).returncode == 0
    deployed = _deployed(space, tmp_path)
    assert list(deployed.rglob("*.pyc")) == []
    assert list(deployed.rglob("__pycache__")) == []


def test_what_is_deployed_is_the_revision_that_is_stamped(space, tmp_path):
    """The Dockerfile says HEAD; the files beside it should also be HEAD's.

    cp shipped the working tree, so a dirty checkout produced a Space that
    reported a revision it was not running -- which is the one question
    /healthz's revision exists to answer.
    """
    scratch = ROOT / "src" / "_deploy_probe.py"
    scratch.write_text("# untracked, must not ship\n")
    try:
        assert _run(space, tmp_path).returncode == 0
        deployed = _deployed(space, tmp_path)
        assert not (deployed / "src" / "_deploy_probe.py").exists()
    finally:
        scratch.unlink()


def test_it_refuses_a_payload_entry_that_is_not_committed(space, tmp_path):
    """The pre-flight check reads HEAD now, because that is what ships.

    It used to test the working tree with `[ -e ]`, which passes for a file
    that exists and has never been committed -- and the archive would then
    fail after the clone, with git's wording rather than the script's.

    Run against a copy of the script with one entry added to PAYLOAD, since
    the point is what the check does with an entry the repository does not
    have at HEAD.
    """
    # Beside the original: the script locates the repository from its own
    # path, so a copy in tmp_path would look for a checkout that is not there.
    patched = SCRIPT.parent / "_patched_for_test.sh"
    patched.write_text(
        SCRIPT.read_text().replace("PAYLOAD=(\n", "PAYLOAD=(\n    only_on_disk\n", 1)
    )
    (ROOT / "only_on_disk").write_text("untracked\n")
    try:
        result = subprocess.run(
            ["bash", str(patched), "owner/demo.git"],
            cwd=ROOT,
            env={**os.environ, "SPACES_BASE": str(space.parent.parent)},
            capture_output=True,
            text=True,
        )
    finally:
        (ROOT / "only_on_disk").unlink()
        patched.unlink()

    assert result.returncode == 1, result.stdout + result.stderr
    assert "not tracked at HEAD: only_on_disk" in result.stderr
    assert "cloning" not in result.stdout, "it should refuse before doing any work"


def test_the_space_can_exclude_things_from_its_build(space, tmp_path):
    """The deployed Space must carry the rules its own docker build reads."""
    assert _run(space, tmp_path).returncode == 0
    deployed = _deployed(space, tmp_path)
    rules = (deployed / ".dockerignore").read_text()
    assert ".git" in rules.split("\n"), "the Space's image would contain .git"
