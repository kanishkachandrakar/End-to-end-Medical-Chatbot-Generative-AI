"""The CI workflow, checked the way its Python is checked.

Its `run:` blocks are shell scripts that only ever execute on a runner, so a
quoting mistake in one is found by pushing. Each has been extracted and passed
through `bash -n` by hand while being written -- a check that runs when someone
remembers to run it.
"""

import pathlib
import re
import subprocess

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
            ["bash", "-n"], input=script, capture_output=True, text=True
        )
        assert result.returncode == 0, f"{name}: {result.stderr.strip()}"


def test_no_action_is_used_unpinned():
    """`@main` or a bare name tracks whatever upstream does next."""
    for reference in re.findall(r"uses: (\S+)", WORKFLOW):
        assert "@" in reference, f"{reference} has no version"
        version = reference.split("@", 1)[1]
        assert version not in ("main", "master", "latest"), reference


def test_the_workflow_uses_no_tabs():
    """YAML forbids them, and a tab in a run block is a different shell."""
    assert "\t" not in WORKFLOW


def test_both_jobs_are_bounded_by_a_timeout():
    """The default is six hours of a runner for a job that has hung."""
    assert WORKFLOW.count("timeout-minutes:") == WORKFLOW.count("runs-on:")


def test_the_token_is_read_only():
    assert re.search(r"^permissions:\n  contents: read$", WORKFLOW, re.MULTILINE)
