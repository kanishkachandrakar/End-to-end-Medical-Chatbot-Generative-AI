"""Every declared dependency is imported, or justified for not being.

The other direction is already checked: nothing may be imported without being
declared. This one catches the opposite drift -- a package that stays in
requirements.txt after the code that needed it is gone. langchain_openai sat
there for weeks that way.

Some dependencies are legitimately never imported by this project: a server it
is run under, a library a langchain integration imports on our behalf. Each has
to be named here with the reason, which is the point -- an unexplained entry in
requirements.txt is the thing being looked for.
"""

import ast
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


def _modules_of(distribution):
    """Top-level modules a distribution provides.

    A distribution name is not an import name -- python-dotenv imports as
    dotenv -- so this asks the installed metadata rather than guessing from
    the string.
    """
    import importlib.metadata as metadata

    provided = {
        module
        for module, distributions in metadata.packages_distributions().items()
        if any(d.lower() == distribution.lower() for d in distributions)
    }
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
