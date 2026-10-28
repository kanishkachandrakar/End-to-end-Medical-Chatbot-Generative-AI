"""app.py's startup path: the code that decides whether the app comes up.

Reachable only because create() is a factory -- importing this module used to
load the embedding model and open a Pinecone connection.
"""

import importlib
import logging
import sys
import types

import pytest

from tests.stubs import StubChain


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

    def test_the_reason_it_could_not_be_read_is_reported(
        self, app_module, monkeypatch, caplog
    ):
        """None was returned with the reason discarded entirely.

        A revoked key, a renamed index and a region that no longer resolves
        all land here, and all produced the one line log_index_size emits
        for an unknown size -- "could not read stats" with nothing after it.
        The chain is built separately and lazily, so it can succeed while
        this fails, leaving that line as the only clue there was one.
        """
        import logging

        monkeypatch.setitem(
            sys.modules,
            "pinecone",
            _pinecone(raises=PermissionError("401 invalid api key")),
        )
        with caplog.at_level(logging.WARNING):
            assert app_module.read_index_size("key") is None

        assert "PermissionError" in caplog.text
        assert "401 invalid api key" in caplog.text
        assert "medicalbot" in caplog.text

    def test_a_timeout_says_it_timed_out(self, app_module, monkeypatch, caplog):
        """Distinguished from a refusal: one is slow, the other is wrong."""
        import logging
        import time
        import types

        module = types.ModuleType("pinecone")

        class Client:
            def __init__(self, api_key=None):
                pass

            def Index(self, name):  # noqa: N802 - matches the real client
                return types.SimpleNamespace(
                    describe_index_stats=lambda: time.sleep(30) or {}
                )

        module.Pinecone = Client
        monkeypatch.setitem(sys.modules, "pinecone", module)

        with caplog.at_level(logging.WARNING):
            assert app_module.read_index_size("key", timeout=0.2) is None

        assert "timed out" in caplog.text
        assert "0s" in caplog.text

    def test_a_hanging_pinecone_is_abandoned(self, app_module, monkeypatch):
        """It runs before the port is bound; waiting there is a crash loop."""
        import sys
        import time
        import types

        module = types.ModuleType("pinecone")

        class Client:
            def __init__(self, api_key=None):
                pass

            def Index(self, name):  # noqa: N802 - matches the real client
                return types.SimpleNamespace(
                    describe_index_stats=lambda: time.sleep(30) or {}
                )

        module.Pinecone = Client
        monkeypatch.setitem(sys.modules, "pinecone", module)

        started = time.monotonic()
        assert app_module.read_index_size("key", timeout=0.2) is None
        assert time.monotonic() - started < 5, "it waited for the hang"


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

    def test_the_lines_survive_a_worker_that_configured_no_logging(
        self, app_module, monkeypatch
    ):
        """The test below passes for a reason gunicorn does not supply.

        caplog.at_level(logging.INFO) raises the root logger's level and
        installs a handler. A gunicorn worker does neither: root stays at
        WARNING with no handlers, so an INFO record from this module was
        rejected by the level and then had nowhere to go anyway. The
        revision and the config summary -- the two lines that exist so a
        misconfigured deploy is not invisible -- were dropped in exactly
        the deployment they were written for.

        So this one captures on the module's own logger and leaves root
        alone, which is the condition that matters.
        """
        import logging
        import sys

        records = []

        class Probe(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        # A worker as gunicorn leaves it, and a module logger as an import
        # leaves it: no level of its own, so its effective level is the
        # root's. The probe gives records somewhere to go if the level ever
        # lets them through -- which is the thing being tested, and is why
        # this does not call configure_logging() itself.
        root = logging.getLogger()
        monkeypatch.setattr(root, "level", logging.WARNING)
        monkeypatch.setattr(root, "handlers", [])
        module_logger = logging.getLogger(app_module.__name__)
        monkeypatch.setattr(module_logger, "level", logging.NOTSET)
        monkeypatch.setattr(module_logger, "handlers", [Probe()])

        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_x")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
        monkeypatch.setitem(sys.modules, "pinecone", _pinecone({}))
        monkeypatch.setattr(app_module, "build_chain", lambda p, g: StubChain())

        app_module.create()

        assert any(line.startswith("revision:") for line in records), records
        assert any("index=" in line for line in records), records


    def test_the_revision_and_config_are_logged(self, app_module, monkeypatch, caplog):
        """Which build, and what it was configured with, before anything else."""
        import logging
        import sys

        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_x")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
        monkeypatch.setitem(sys.modules, "pinecone", _pinecone({}))
        monkeypatch.setattr(app_module, "build_chain", lambda p, g: StubChain())

        with caplog.at_level(logging.INFO):
            app_module.create()

        text = caplog.text
        assert "revision:" in text
        assert "config:" in text and "index=" in text


class TestBuildChain:
    def test_wires_the_retriever_and_the_llm_together(self, app_module):
        assert app_module.build_chain("p", "g") == "chain"
