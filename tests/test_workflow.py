"""The CI workflow, checked the way its Python is checked.

Its `run:` blocks are shell scripts that only ever execute on a runner, so a
quoting mistake in one is found by pushing. Each has been extracted and passed
through `bash -n` by hand while being written -- a check that runs when someone
remembers to run it.
"""

import pathlib
import re
import subprocess

# A child that never exits must not become a suite that never finishes.
SUBPROCESS_TIMEOUT = 120

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text()


def _run_blocks():
    """(step name, script) for every `run: |` block, de-indented."""
    blocks = []
    pattern = re.compile(
        r"- name: (?P<name>.+?)\n"          # the step
        r"(?:.*?\n)??"                       # env, uses, etc.
        r"        run: \|\n"                 # a block scalar
        r"(?P<body>(?:(?:          .*)?\n)+)",
    )
    for match in pattern.finditer(WORKFLOW):
        body = "\n".join(
            line[10:] if line.startswith(" " * 10) else line
            for line in match.group("body").split("\n")
        )
        blocks.append((match.group("name").strip(), body))
    return blocks


def test_the_run_blocks_were_found():
    """Everything below iterates them; finding none would pass silently."""
    assert len(_run_blocks()) >= 3, [name for name, _ in _run_blocks()]


def test_every_run_block_is_valid_shell():
    for name, script in _run_blocks():
        result = subprocess.run(
            ["bash", "-n"],
            input=script,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0, f"{name}: {result.stderr.strip()}"


def test_every_action_is_pinned_by_commit_sha():
    """A major tag is mutable, and these run in CI holding a token.

    This asserted only that the ref was not @main, @master or @latest --
    which let five actions sit on moving major tags that their publishers
    can repoint at any commit. The Dockerfile's base is pinned by digest for
    exactly this reason, so half the build was hardened and half was not.
    """
    references = re.findall(r"uses: (\S+)", WORKFLOW)
    assert references, "no actions found"
    for reference in references:
        assert "@" in reference, f"{reference} has no version"
        ref = reference.split("@", 1)[1]
        assert re.fullmatch(r"[0-9a-f]{40}", ref), (
            f"{reference} is not pinned to a commit SHA"
        )


def test_every_pin_says_which_version_it_is():
    """A bare SHA is unreadable, and dependabot keys on the comment.

    `uses: owner/repo@<sha> # v5` is the format it reads to propose the next
    one; without the comment the pin is both opaque and unwatched.
    """
    for line in WORKFLOW.splitlines():
        if "uses:" not in line:
            continue
        assert re.search(r"uses: \S+@[0-9a-f]{40} # v\d", line), line.strip()


def test_the_workflow_uses_no_tabs():
    """YAML forbids them, and a tab in a run block is a different shell."""
    assert "\t" not in WORKFLOW


def test_both_jobs_are_bounded_by_a_timeout():
    """The default is six hours of a runner for a job that has hung."""
    assert WORKFLOW.count("timeout-minutes:") == WORKFLOW.count("runs-on:")


def test_the_token_is_read_only():
    assert re.search(r"^permissions:\n  contents: read$", WORKFLOW, re.MULTILINE)


def test_no_checkout_leaves_the_token_on_disk():
    """actions/checkout writes the job's token into .git/config by default.

    Every later step in the job can then read it. Nothing in this workflow
    pushes -- the token is wanted for the clone and nothing after it -- and
    it is read-only anyway, since permissions is contents: read at the top.
    So this is the second lock rather than the first, which is the only kind
    worth adding without an incident to point at.
    """
    import yaml

    workflow = yaml.safe_load(WORKFLOW)
    checkouts = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "actions/checkout" in str(step.get("uses", ""))
    ]
    assert len(checkouts) == 2, len(checkouts)
    for step in checkouts:
        assert step.get("with", {}).get("persist-credentials") is False, step


def test_the_workflow_asks_for_no_more_than_read_access():
    """The first lock, and the reason the one above is only the second."""
    import yaml

    workflow = yaml.safe_load(WORKFLOW)
    assert workflow["permissions"] == {"contents": "read"}, workflow["permissions"]
