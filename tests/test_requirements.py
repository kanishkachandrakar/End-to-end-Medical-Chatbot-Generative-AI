"""Every dependency needs a floor.

Both times this project failed to start on a clean install, the cause was a
requirement with no bound. The second was the instructive one: resolving
`pinecone[grpc]` constrained `pinecone`, and rather than report a conflict pip
walked `langchain-pinecone` backwards to 0.0.1 -- a release predating the class
app.py imports. A lower bound is what makes that a resolution error instead of
a working install of the wrong thing.

So: no bare package names. Not as style, but because a bare name is a licence
for the resolver to pick anything at all.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Lines that are not a requirement.
NON_REQUIREMENTS = {"-e .", "."}


def _requirements(filename):
    """(name, specifier) for each requirement line, extras stripped."""
    parsed = []
    for raw in (ROOT / filename).read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line or line in NON_REQUIREMENTS or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)(\[[^\]]*\])?(.*)$", line)
        assert match, f"could not parse requirement: {raw!r}"
        parsed.append((match.group(1).lower(), match.group(3).strip()))
    return parsed


def test_the_requirements_were_parsed():
    """Everything below is a per-line check; an empty list would pass them."""
    assert len(_requirements("requirements.txt")) >= 10
    assert len(_requirements("requirements-dev.txt")) >= 5


def test_every_runtime_requirement_has_a_lower_bound():
    unbounded = [
        name
        for name, spec in _requirements("requirements.txt")
        if ">=" not in spec and "==" not in spec
    ]
    assert not unbounded, (
        f"no floor on {unbounded} -- the resolver may pick any version, which is "
        "how langchain-pinecone 0.0.1 got installed"
    )


def test_every_dev_requirement_has_a_lower_bound():
    unbounded = [
        name
        for name, spec in _requirements("requirements-dev.txt")
        if ">=" not in spec and "==" not in spec
    ]
    assert not unbounded, f"no floor on {unbounded}"


def test_no_requirement_is_pinned_to_a_single_version():
    """An == pin here would silently stop dependabot's security updates."""
    for filename in ("requirements.txt", "requirements-dev.txt"):
        for name, spec in _requirements(filename):
            assert "==" not in spec, f"{name} in {filename} is pinned exactly"


def test_the_two_files_agree_about_shared_packages():
    """Testing against versions prod will not install hides exactly these bugs."""
    runtime = dict(_requirements("requirements.txt"))
    dev = dict(_requirements("requirements-dev.txt"))
    for name in sorted(set(runtime) & set(dev)):
        assert runtime[name] == dev[name], (
            f"{name}: requirements.txt says {runtime[name]!r} but "
            f"requirements-dev.txt says {dev[name]!r}"
        )


def test_the_packages_that_have_broken_this_project_have_ceilings():
    """A floor stops a backslide; a ceiling stops a surprise major release."""
    runtime = dict(_requirements("requirements.txt"))
    brittle = (
        "langchain",
        "langchain-classic",
        "langchain-core",
        "sentence-transformers",
    )
    for name in brittle:
        assert "<" in runtime[name], f"{name} has no upper bound"


def test_the_pinecone_floor_stays_below_its_upstream_ceiling():
    """langchain-pinecone caps pinecone, so a floor above that cap cannot resolve.

    0.2.13 -- its latest -- requires pinecone[asyncio]<8.0.0,>=6.0.0. A
    dependabot PR raising our floor to 10.x is unsatisfiable by construction,
    and the install quietly backtracks rather than failing loudly, so this is
    the thing that says so.
    """
    import re

    floors = dict(_requirements("requirements.txt"))
    specifier = floors.get("pinecone")
    assert specifier, "pinecone is not declared"
    found = re.search(r">=(\d+)", specifier)
    assert found, specifier
    assert int(found.group(1)) < 8, (
        f"pinecone floor is {specifier}, and langchain-pinecone requires "
        "pinecone<8.0.0 -- this cannot resolve"
    )


def test_the_reason_pinecone_is_held_back_is_written_down():
    """Otherwise the next unmergeable bump gets investigated from scratch."""
    text = (ROOT / "requirements.txt").read_text()
    assert "langchain-pinecone" in text.split("pinecone>=5.0")[0] or (
        "pinecone[asyncio]<8.0.0" in text
    ), "requirements.txt should name the upstream ceiling"
