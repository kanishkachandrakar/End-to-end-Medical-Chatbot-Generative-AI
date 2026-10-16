"""langchain-community is sunset; keep it to the one module that needs it.

It says so on import: no longer actively maintained, contents moving to
standalone integrations. The embeddings half has already gone, and this project
has twice been broken by a langchain package relocating something -- so the
remaining usage is worth confining rather than letting it spread back through
the import graph.

src/loaders.py is the permitted home. Nothing the web app imports may reach it.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUNSET = "langchain_community"

# The loaders have no standalone home yet, so these files may still import
# them. Anything else reaching for langchain_community is a regression.
PERMITTED = {"src/loaders.py"}

# The embeddings do have one -- langchain-huggingface -- and it is adopted. So
# this submodule is forbidden everywhere, including the notebook, which kept
# importing it for a day after src/ had moved.
RETIRED_SUBMODULES = ("langchain_community.embeddings",)

# Modules reachable from the web app. store_index.py is not here: reading PDFs
# is its job, and it is the thing src/loaders.py exists for.
APP_GRAPH = [
    "app.py",
    "src/webapp.py",
    "src/assets.py",
    "src/middleware.py",
    "src/helper.py",
    "src/chunking.py",
    "src/config.py",
    "src/env.py",
    "src/errors.py",
    "src/headers.py",
    "src/prompt.py",
    "src/ratelimit.py",
    "src/text.py",
]


def _imported_modules(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_the_app_graph_names_files_that_exist():
    """Otherwise this passes by checking a list of typos."""
    for name in APP_GRAPH:
        assert (ROOT / name).is_file(), name


def test_the_app_graph_covers_every_module_it_should():
    """The list is maintained by hand, so it has to be checked.

    Everything in src/ is in it except the module that is allowed to import
    the sunset package -- and store_index.py, whose job is reading PDFs. A new
    module missing from here would be free to import langchain_community
    without the check above noticing.
    """
    modules = {
        f"src/{path.name}"
        for path in (ROOT / "src").glob("*.py")
        if path.name != "__init__.py"
    }
    expected = modules - PERMITTED
    missing = expected - set(APP_GRAPH)
    assert not missing, (
        f"{missing} are in src/ but absent from APP_GRAPH, so nothing checks "
        "what they import"
    )


def test_the_graph_and_the_permitted_set_do_not_overlap():
    """A module cannot both be forbidden the package and be its only importer."""
    assert not set(APP_GRAPH) & PERMITTED


def test_nothing_in_the_app_graph_imports_the_sunset_package():
    offenders = [
        name
        for name in APP_GRAPH
        if any(module.startswith(SUNSET) for module in _imported_modules(name))
    ]
    assert not offenders, (
        f"{offenders} import {SUNSET}, which is sunset -- the web app has no "
        "PDFs to read, so it should not depend on the package that reads them"
    )


def test_only_the_permitted_module_imports_it():
    found = {
        str(path.relative_to(ROOT))
        for path in list(ROOT.glob("*.py")) + list((ROOT / "src").glob("*.py"))
        if any(
            module.startswith(SUNSET)
            for module in _imported_modules(path.relative_to(ROOT))
        )
    }
    assert found == PERMITTED, f"expected {PERMITTED}, found {found}"


def test_the_permitted_module_really_does_import_it():
    """If the loaders move on too, this file should be deleted, not left lying."""
    assert any(
        module.startswith(SUNSET)
        for module in _imported_modules("src/loaders.py")
    )


def test_the_sunset_package_is_bounded_below_its_removal_boundary():
    """Its notice says the deprecated contents survive until 1.0."""
    import re

    requirements = (ROOT / "requirements.txt").read_text()
    line = re.search(r"^langchain-community(\S+)", requirements, re.MULTILINE)
    assert line, "langchain-community is not declared"
    assert "<1" in line.group(1), line.group(0)


def _every_python_source():
    """Every file in the repo that could import something, notebook included."""
    import json

    for path in list(ROOT.glob("*.py")) + list((ROOT / "src").glob("*.py")):
        yield str(path.relative_to(ROOT)), path.read_text()

    notebook = json.loads((ROOT / "research" / "trials.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            yield f"research/trials.ipynb[{index}]", "".join(cell["source"])


def test_there_are_sources_to_scan():
    assert len(list(_every_python_source())) > 20


def test_the_retired_submodules_are_imported_nowhere():
    """A replacement that is adopted in src/ but not elsewhere is half-adopted."""
    offenders = []
    for name, source in _every_python_source():
        for retired in RETIRED_SUBMODULES:
            if retired in source:
                offenders.append(f"{name} imports {retired}")
    assert not offenders, (
        "a standalone replacement exists and is in use elsewhere: "
        + "; ".join(offenders)
    )


def test_the_notebook_is_scanned_too():
    """It was the one place the earlier check did not look."""
    scanned = {name for name, _ in _every_python_source()}
    assert any(name.startswith("research/trials.ipynb") for name in scanned)
