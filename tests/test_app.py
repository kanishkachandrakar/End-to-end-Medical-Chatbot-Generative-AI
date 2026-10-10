"""app.py's startup path: the code that decides whether the app comes up.

Reachable only because create() is a factory -- importing this module used to
load the embedding model and open a Pinecone connection.
"""

import importlib
import logging
import sys
import types

import pytest


@pytest.fixture
def app_module(monkeypatch):
    """Import app.py with its third-party dependencies stubbed out."""
    stubs = {
        "langchain_classic.chains": {"create_retrieval_chain": lambda *a: "chain"},
        "langchain_classic.chains.combine_documents": {
            "create_stuff_documents_chain": lambda *a: "qa"
        },
        "langchain_core.prompts": {
            "ChatPromptTemplate": types.SimpleNamespace(
                from_messages=lambda m: "prompt"
            )
        },
        "langchain_groq": {"ChatGroq": lambda **kw: "llm"},
        "langchain_pinecone": {
            "PineconeVectorStore": types.SimpleNamespace(
                from_existing_index=lambda **kw: types.SimpleNamespace(
                    as_retriever=lambda **k: "retriever"
                )
            )
        },
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    helper = types.ModuleType("src.helper")
    helper.download_hugging_face_embeddings = lambda: "embeddings"
    monkeypatch.setitem(sys.modules, "src.helper", helper)

    monkeypatch.delitem(sys.modules, "app", raising=False)
    module = importlib.import_module("app")
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    yield module
    sys.modules.pop("app", None)


def _pinecone(stats=None, raises=None):
    """A stub pinecone module shaped like the bit read_index_size uses."""
    module = types.ModuleType("pinecone")

    class Client:
        def __init__(self, api_key=None):
            if raises is not None:
                raise raises

        def Index(self, name):  # noqa: N802 - matches the real client
            return types.SimpleNamespace(describe_index_stats=lambda: stats or {})

    module.Pinecone = Client
    return module


class TestReadIndexSize:
    def test_returns_the_vector_count(self, app_module, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "pinecone", _pinecone({"total_vector_count": 5860})
        )
        assert app_module.read_index_size("key") == 5860

    def test_an_empty_index_is_zero_not_none(self, app_module, monkeypatch):
        """Zero and 'unknown' mean different things at /healthz."""
        monkeypatch.setitem(
            sys.modules, "pinecone", _pinecone({"total_vector_count": 0})
        )
        assert app_module.read_index_size("key") == 0

    def test_a_missing_key_in_the_response_is_zero(self, app_module, monkeypatch):
        monkeypatch.setitem(sys.modules, "pinecone", _pinecone({}))
        assert app_module.read_index_size("key") == 0

    def test_an_unreachable_pinecone_is_none(self, app_module, monkeypatch):
        """Must not raise: it would take the whole app down with it."""
        monkeypatch.setitem(
            sys.modules, "pinecone", _pinecone(raises=RuntimeError("no network"))
        )
        assert app_module.read_index_size("key") is None


class TestLogIndexSize:
    def test_a_populated_index_is_logged_at_info(self, app_module, caplog):
        with caplog.at_level(logging.INFO):
            app_module.log_index_size(logging.getLogger("t"), 5860)
        assert "holds 5860 vectors" in caplog.text

    def test_an_empty_index_warns_and_says_what_to_run(self, app_module, caplog):
        with caplog.at_level(logging.WARNING):
            app_module.log_index_size(logging.getLogger("t"), 0)
        assert "is empty" in caplog.text
        assert "store_index.py" in caplog.text

    def test_an_unknown_size_warns(self, app_module, caplog):
        with caplog.at_level(logging.WARNING):
            app_module.log_index_size(logging.getLogger("t"), None)
        assert "could not read stats" in caplog.text


class TestCreate:
    def test_builds_a_working_app(self, app_module, monkeypatch):
        monkeypatch.setenv("PINECONE_API_KEY", "p")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setitem(
            sys.modules, "pinecone", _pinecone({"total_vector_count": 10})
        )
        monkeypatch.setattr(app_module, "build_chain", lambda p, g: StubChain())

        client = app_module.create().test_client()
        assert client.get("/healthz").data.decode().startswith("ok")
        assert client.post("/get", data={"msg": "q"}).status_code == 200

    def test_missing_keys_are_reported_rather_than_guessed_at(
        self, app_module, monkeypatch
    ):
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
            app_module.create()

    def test_a_failing_chain_yields_a_degraded_app_not_an_exception(
        self, app_module, monkeypatch
    ):
        """The crash-loop case: it must come up and say what is wrong."""
        monkeypatch.setenv("PINECONE_API_KEY", "p")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setitem(sys.modules, "pinecone", _pinecone({}))

        def explode(pinecone_api_key, groq_api_key):
            raise ConnectionError("pinecone is down")

        monkeypatch.setattr(app_module, "build_chain", explode)

        client = app_module.create().test_client()
        health = client.get("/healthz")
        assert health.status_code == 503
        assert b"chain-unavailable" in health.data
        assert client.post("/get", data={"msg": "q"}).status_code == 503

    def test_the_refusing_stand_in_raises_if_it_is_ever_reached(self, app_module):
        """Belt and braces: the routes refuse first, but this must not answer."""
        stand_in = app_module._Refusing(ValueError("original"))
        with pytest.raises(RuntimeError, match="unavailable"):
            stand_in.invoke({"input": "q"})


class TestBuildChain:
    def test_wires_the_retriever_and_the_llm_together(self, app_module):
        assert app_module.build_chain("p", "g") == "chain"


class StubChain:
    calls: list = []

    def invoke(self, payload):
        return {"answer": "an answer"}
