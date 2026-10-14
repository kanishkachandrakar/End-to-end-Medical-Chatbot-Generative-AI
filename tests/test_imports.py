"""Every name app.py imports must exist in the package it is imported from.

tests/test_app.py stubs the third-party modules, which is what lets the startup
path be tested without a network -- but it also means the import paths in
app.py are whatever the stubs say they are. A package moving a name is then
invisible until something starts the app for real.

That is not hypothetical: langchain 1.0 moved create_retrieval_chain into
langchain-classic, the app stopped starting, and the unit suite stayed green.
These tests read app.py's own import statements and check them against the
installed packages, so the next such move fails here.

No network and no torch: importing these modules is cheap, and only
constructing the embedding model is not.
"""

import ast
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.slow


def _imports(filename):
    """(module, name) for every `from X import Y` in the file."""
    tree = ast.parse((ROOT / filename).read_text())
    return [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    ]


def _plain_imports(filename):
    """Module names from every `import X` in the file."""
    tree = ast.parse((ROOT / filename).read_text())
    return [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]


MODULES = ["app.py", "store_index.py", "gunicorn.conf.py"]


def _notebook_imports():
    """(module, name) for every `from X import Y` in the notebook's code cells.

    The notebook is a prototype and not maintained alongside src/, which is
    exactly why its imports go stale unnoticed: nothing executes it. The
    langchain 1.0 move broke app.py and the notebook together, and only app.py
    was noticed.
    """
    import json

    notebook = json.loads((ROOT / "research" / "trials.ipynb").read_text())
    found = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # A cell with magics such as %pwd is not valid Python on its own.
            continue
        found.extend(
            (node.module, alias.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            for alias in node.names
        )
    return found


def test_the_import_statements_were_found():
    """These are all absence-style checks; an empty list would pass them."""
    assert len(_imports("app.py")) >= 10


@pytest.mark.parametrize("filename", MODULES)
def test_every_from_import_resolves(filename):
    for module_name, name in _imports(filename):
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"{module_name} has no {name!r}"


@pytest.mark.parametrize("filename", MODULES)
def test_every_plain_import_resolves(filename):
    for module_name in _plain_imports(filename):
        importlib.import_module(module_name)


def test_the_chain_helpers_are_where_app_py_says():
    """Named explicitly, because this is the pair that moved."""
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import (
        create_stuff_documents_chain,
    )

    assert callable(create_retrieval_chain)
    assert callable(create_stuff_documents_chain)


def test_the_old_location_is_really_gone():
    """If langchain restores langchain.chains, this test says so."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("langchain.chains")


def test_everything_imported_is_declared_as_a_dependency():
    """A module that happens to be installed transitively is not a dependency."""
    declared = (ROOT / "requirements.txt").read_text().replace("-", "_").lower()
    # The project's own package, and the standard library -- neither belongs
    # in requirements.txt. Listed rather than detected because sys.stdlib_
    # module_names would also excuse a third-party package shadowing a
    # stdlib name, which is exactly the mistake worth catching.
    skip = {
        "src",
        "flask",
        "dotenv",
        "argparse",
        "collections",
        "concurrent",
        "hashlib",
        "logging",
        "os",
        "pathlib",
        "re",
        "sys",
        "threading",
        "time",
        "typing",
        "uuid",
    }
    top_level = {
        module.split(".")[0]
        for filename in MODULES
        for module, _ in _imports(filename)
    } | {
        module.split(".")[0]
        for filename in MODULES
        for module in _plain_imports(filename)
    }
    for module in sorted(top_level - skip):
        assert module.lower() in declared, f"{module} is imported but not required"


def test_the_notebook_imports_were_found():
    assert len(_notebook_imports()) >= 5


def test_every_notebook_import_resolves():
    """It is a prototype, not a reason for it to reference a dead module."""
    for module_name, name in _notebook_imports():
        if module_name.startswith("src"):
            continue
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"{module_name} has no {name!r}"


def test_the_notebook_does_not_use_the_moved_chains_module():
    """Named explicitly: this is the import that broke and went unnoticed."""
    modules = {module for module, _ in _notebook_imports()}
    assert not any(m.startswith("langchain.chains") for m in modules), modules
