"""The updates dependabot is told not to propose, and why.

An ignore is a decision to stop hearing about something, which is the kind of
thing that is right when it is made and wrong a year later. Both of the ones
here exist because an upstream package caps something, so both have to name
the cap -- that is what makes them reviewable rather than permanent.
"""

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / ".github" / "dependabot.yml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
TEXT = CONFIG_PATH.read_text()


def _ecosystem(name):
    for update in CONFIG["updates"]:
        if update["package-ecosystem"] == name:
            return update
    raise AssertionError(f"no {name} ecosystem configured")


def test_the_three_ecosystems_are_watched():
    """pip, github-actions and docker: the three things that change by themselves."""
    watched = {u["package-ecosystem"] for u in CONFIG["updates"]}
    assert watched == {"pip", "github-actions", "docker"}, watched


def test_pinecone_is_held_below_the_ceiling_langchain_pinecone_sets():
    """langchain-pinecone requires pinecone[asyncio]<8.0.0, so 8+ cannot resolve."""
    ignores = _ecosystem("pip").get("ignore") or []
    pinecone = [i for i in ignores if i.get("dependency-name") == "pinecone"]
    assert pinecone, "pinecone 8+ cannot resolve and is not ignored"
    assert pinecone[0]["versions"] == [">=8"], pinecone


def test_the_base_image_is_held_below_the_python_langchain_pinecone_allows():
    """requires_python <3.14 there means the image cannot be built on 3.14."""
    ignores = _ecosystem("docker").get("ignore") or []
    python = [i for i in ignores if i.get("dependency-name") == "python"]
    assert python, "a python 3.14+ base image cannot install the requirements"
    assert python[0]["versions"] == [">=3.14"], python


def test_every_ignore_names_the_package_that_forces_it():
    """An ignore with no reason is indistinguishable from one nobody revisits.

    Both of these are langchain-pinecone's ceilings, and both comments say
    so, so the day that package raises them there is something to search for.
    """
    for update in CONFIG["updates"]:
        if not update.get("ignore"):
            continue
        for entry in update["ignore"]:
            name = entry["dependency-name"]
            # The comment block sits above the entry in the file.
            before = TEXT.split(f'dependency-name: "{name}"')[0]
            assert "langchain-pinecone" in before, (
                f"the ignore for {name} does not say what forces it"
            )


def test_the_ignores_are_not_a_blanket():
    """A floor, not a ban: patch and digest updates still have to arrive.

    The Dockerfile's digest pin is only tolerable because something proposes
    the updates -- that is what the file says about itself -- so an ignore
    that covered everything would turn the pin back into a liability.
    """
    for update in CONFIG["updates"]:
        for entry in update.get("ignore") or []:
            for specifier in entry["versions"]:
                assert specifier.startswith(">="), (
                    f"{entry['dependency-name']} is ignored as {specifier!r}, "
                    "which is wider than the ceiling that forces it"
                )
