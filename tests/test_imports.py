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
    skip = {"src", "flask", "dotenv", "os", "sys", "logging", "re", "hashlib",
            "argparse", "pathlib", "time", "uuid", "threading", "typing",
            "collections"}
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
