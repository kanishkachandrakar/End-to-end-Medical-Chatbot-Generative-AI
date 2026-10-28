"""Every declared dependency is imported, and everything imported is declared.

This file used to open by saying "the other direction is already checked:
nothing may be imported without being declared". There was no such check
anywhere. Both halves live here now, and the half that did not exist found
two: werkzeug, imported by src/middleware.py for HTTPException, and
langchain-text-splitters, imported by src/chunking.py and declared only in
requirements-dev.txt.

The declared-but-unused half catches the opposite drift -- a package that
stays in requirements.txt after the code that needed it is gone.
langchain_openai sat there for weeks that way.

Some dependencies are legitimately never imported by this project: a server it
is run under, a library a langchain integration imports on our behalf. Each has
to be named here with the reason, which is the point -- an unexplained entry in
requirements.txt is the thing being looked for.
"""

import ast
import functools
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Declared but never imported here, deliberately.
JUSTIFIED = {
    # The WSGI server. Invoked by the Dockerfile, not imported.
    "gunicorn",
    # Imported by langchain-huggingface to build the model, not by us. Declared
    # because its version decides the embedding dimension, which decides
    # whether the existing Pinecone index is usable at all.
    "sentence-transformers",
    # langchain-classic depends on it; declared with an upper bound because a
    # major release of it is what moved create_retrieval_chain in the first
    # place.
    "langchain",
    # Read by langchain-community's PyPDFLoader, not by this code.
    "pypdf",
}


def _declared(filename):
    declarations = {}
    for raw in (ROOT / filename).read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[><=\[]", line, maxsplit=1)[0].strip()
        if name:
            declarations[name.lower()] = line
    return declarations


@functools.lru_cache(maxsize=1)
def _distribution_modules():
    """Distribution name (lowercased) -> the top-level modules it provides.

    Built once. packages_distributions() walks every installed distribution's
    metadata, and calling it per dependency made these tests 4.5 seconds of an
    11-second suite -- on a file that is not marked slow, so it was in the fast
    path too.
    """
    import collections
    import importlib.metadata as metadata

    mapping = collections.defaultdict(set)
    for module, distributions in metadata.packages_distributions().items():
        for distribution in distributions:
            mapping[distribution.lower()].add(module)
    return mapping


def _modules_of(distribution):
    """Top-level modules a distribution provides.

    A distribution name is not an import name -- python-dotenv imports as
    dotenv -- so this asks the installed metadata rather than guessing from
    the string.
    """
    provided = _distribution_modules().get(distribution.lower())
    return provided or {distribution.replace("-", "_")}


def _imported_top_level():
    """Top-level module names imported anywhere in the project."""
    names = set()
    sources = [
        path.read_text()
        for path in list(ROOT.glob("*.py")) + list((ROOT / "src").glob("*.py"))
    ]
    notebook = json.loads((ROOT / "research" / "trials.ipynb").read_text())
    sources += [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]

    for source in sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
    return names


def test_both_sides_were_read():
    assert len(_declared("requirements.txt")) >= 10
    assert len(_imported_top_level()) >= 10


def test_every_runtime_dependency_is_imported_or_justified():
    imported = _imported_top_level()
    unexplained = [
        name
        for name in _declared("requirements.txt")
        if name not in JUSTIFIED and not (_modules_of(name) & imported)
    ]
    assert not unexplained, (
        f"declared but never imported and not justified: {unexplained} -- "
        "either the code that needed it is gone, or it belongs in JUSTIFIED "
        "with the reason"
    )


def test_the_justifications_are_all_still_declared():
    """A justification for a dependency that no longer exists is dead weight."""
    declared = set(_declared("requirements.txt"))
    stale = JUSTIFIED - declared
    assert not stale, f"justified but no longer declared: {stale}"


def test_nothing_is_justified_that_is_actually_imported():
    """Then the justification is wrong and misleads whoever reads it next."""
    imported = _imported_top_level()
    wrong = {name for name in JUSTIFIED if _modules_of(name) & imported}
    assert not wrong, f"justified as unimported but imported: {wrong}"


def _imported_top_level_modules():
    """Top-level modules that the shipped code imports, third-party only."""
    import sys

    shipped = [ROOT / "app.py", ROOT / "store_index.py", *(ROOT / "src").glob("*.py")]
    modules = set()
    for path in shipped:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return {
        name
        for name in modules
        if name not in sys.stdlib_module_names and name != "src"
    }


def _modules_to_distribution():
    """Top-level module name -> the declared distribution providing it."""
    mapping = {}
    for distribution, provided in _distribution_modules().items():
        for module in provided:
            mapping.setdefault(module, set()).add(distribution)
    return mapping


def test_the_imports_were_found():
    """The check below iterates them, so an empty set would pass it."""
    found = _imported_top_level_modules()
    assert len(found) > 8, found
    # The two the missing direction caught, named so neither can quietly
    # become transitive again.
    assert {"werkzeug", "langchain_text_splitters"} <= found


def test_nothing_is_imported_without_being_declared():
    """A transitive dependency is not a declared one.

    requirements.txt says so twice in its own comments, about langchain-core
    and pydantic: both arrive with langchain, both are imported by name, so
    both are declared. werkzeug arrived with flask and was imported by name
    and was not -- which means a flask release that stopped depending on it,
    or depended on a different major version of it, would break an import
    this project makes directly while every declared bound still held.
    """
    declared = set(_declared("requirements.txt")) | set(
        _declared("requirements-dev.txt")
    )
    provides = _modules_to_distribution()

    undeclared = []
    for module in sorted(_imported_top_level_modules()):
        candidates = provides.get(module, {module})
        if not {name.lower() for name in candidates} & declared:
            undeclared.append((module, sorted(candidates)))
    assert not undeclared, f"imported but not declared: {undeclared}"
