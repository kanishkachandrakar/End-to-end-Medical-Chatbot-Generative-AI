"""The three things handed to create_app, against the one it declares.

src/webapp.py says of RagChain: "Spelled out as a Protocol so the stub the
tests pass in is a declared part of the interface rather than an accident of
duck typing." Nothing checked it. A Protocol is a type annotation, mypy runs
over `src app.py store_index.py` and not over tests/, so the annotation and
the stub were never in the same room.

Three implementations exist: tests.stubs.StubChain, app._Refusing (which
stands in when the chain cannot be built) and whatever
create_retrieval_chain returns, which is a langchain Runnable.
"""

import inspect

import pytest

from src.webapp import RagChain
from tests.stubs import StubChain


def _runnable():
    """langchain's Runnable, which is what the real chain is."""
    from langchain_core.runnables import Runnable

    return Runnable


def _refusing():
    """app._Refusing, imported without app.py's module-level side effects.

    app.py imports cleanly -- nothing happens until create() is called -- so
    this is an ordinary import, and that is worth asserting here too.
    """
    import app

    return app._Refusing


def _implementations():
    """Every type that is passed to create_app, resolved on demand."""
    return {
        "StubChain": StubChain,
        "app._Refusing": _refusing(),
        "langchain Runnable": _runnable(),
    }


def test_the_protocol_is_checkable_at_all():
    """Without runtime_checkable every assertion below raises TypeError."""
    assert isinstance(StubChain(), RagChain)


def test_all_three_implementations_were_found():
    """Each check below iterates them; an empty mapping would pass silently."""
    assert len(_implementations()) == 3


@pytest.mark.parametrize("name", sorted(_implementations()))
def test_each_implementation_satisfies_the_protocol(name):
    implementation = _implementations()[name]
    assert hasattr(implementation, "invoke"), name
    assert issubclass(implementation, RagChain), name


@pytest.mark.parametrize("name", sorted(_implementations()))
def test_invoke_takes_the_one_argument_the_route_passes(name):
    """The call site is `rag_chain.invoke({"input": msg})` -- positional.

    So the parameter's name is not part of the contract, and comparing names
    would reject the real chain, whose is `input` rather than `payload`. What
    has to hold is that one positional argument binds.
    """
    signature = inspect.signature(_implementations()[name].invoke)
    # bound as an instance method would be, so `self` is not counted
    parameters = [p for p in signature.parameters.values() if p.name != "self"]
    unbound = signature.replace(parameters=parameters)
    try:
        bound = unbound.bind({"input": "a question"})
    except TypeError as exc:  # pragma: no cover - the failure this guards
        raise AssertionError(f"{name}.invoke{unbound}: {exc}") from exc
    assert len(bound.args) == 1, (name, bound.args)


def test_the_protocol_declares_nothing_but_invoke():
    """create_app needs one method; a Protocol that grew would hide that."""
    declared = {name for name in vars(RagChain) if not name.startswith("_")}
    assert declared == {"invoke"}, declared
