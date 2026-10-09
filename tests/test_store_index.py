"""The indexer's dry run: it must report the counts and touch nothing."""

import importlib
import sys
import types

import pytest


class _Chunk:
    def __init__(self, text):
        self.page_content = text


@pytest.fixture
def store_index(monkeypatch):
    """Import store_index with its heavyweight dependencies stubbed out.

    The module reaches for pinecone and the embedding model at import time, so
    none of this is reachable otherwise -- which is the reason the dry run was
    never covered.
    """
    pinecone = types.ModuleType("pinecone")
    pinecone.ServerlessSpec = object
    grpc = types.ModuleType("pinecone.grpc")
    grpc.PineconeGRPC = _unusable("pinecone client")
    pinecone.grpc = grpc

    vectorstores = types.ModuleType("langchain_pinecone")
    vectorstores.PineconeVectorStore = _unusable("vector store")

    helper = types.ModuleType("src.helper")
    helper.load_pdf = lambda path: [object()] * 2
    helper.download_hugging_face_embeddings = _unusable("embedding model")

    chunking = types.ModuleType("src.chunking")
    chunking.text_split = lambda pages: [
        _Chunk("alpha"),
        _Chunk("beta"),
        _Chunk("alpha"),
    ]

    for name, module in (
        ("pinecone", pinecone),
        ("pinecone.grpc", grpc),
        ("langchain_pinecone", vectorstores),
        ("src.helper", helper),
        ("src.chunking", chunking),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.delitem(sys.modules, "store_index", raising=False)
    module = importlib.import_module("store_index")
    # Otherwise main() reloads the developer's real .env and puts the key
    # straight back, so the tests below would not be testing what they say.
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    yield module
    sys.modules.pop("store_index", None)


def _unusable(what):
    def _raise(*args, **kwargs):
        raise AssertionError(f"the dry run must not reach the {what}")

    return _raise


def test_dry_run_reports_the_counts(store_index, capsys, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    store_index.main(["--dry-run"])
    output = capsys.readouterr().out
    assert "2 pages -> 3 chunks" in output
    assert "would upsert 2 unique chunks" in output


def test_dry_run_needs_no_api_key(store_index, monkeypatch):
    """The whole point: check the chunking before setting anything up."""
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    store_index.main(["--dry-run"])


def test_dry_run_reaches_neither_pinecone_nor_the_model(store_index, monkeypatch):
    """The stubs raise if touched, so reaching either fails this test."""
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    store_index.main(["--dry-run"])


def test_dry_run_reports_collapsing_duplicates(store_index, capsys, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    store_index.main(["--dry-run"])
    assert "1 chunks are byte-identical" in capsys.readouterr().out


def test_a_real_run_still_demands_a_key(store_index, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="PINECONE_API_KEY"):
        store_index.main([])


def test_the_flag_defaults_to_off(store_index):
    assert store_index.parse_args([]).dry_run is False
    assert store_index.parse_args(["--dry-run"]).dry_run is True


@pytest.fixture
def indexer(monkeypatch):
    """store_index with recording stubs, so the upsert path can be inspected.

    The dry-run fixture above makes the heavy dependencies raise if touched.
    This one lets them be called and writes down what with.
    """
    calls = {"existing": [], "created": [], "upserts": [], "opened": []}

    class _Store:
        @staticmethod
        def from_existing_index(index_name, embedding):
            calls["opened"].append(index_name)
            return _Store()

        def add_documents(self, documents, ids):
            calls["upserts"].append(([c.page_content for c in documents], list(ids)))

    class _Client:
        def __init__(self, api_key=None):
            calls["api_key"] = api_key

        def list_indexes(self):
            return types.SimpleNamespace(names=lambda: calls["existing"])

        def create_index(self, **kwargs):
            calls["created"].append(kwargs)

        def describe_index(self, name):
            return types.SimpleNamespace(
                dimension=calls.get("dimension", 384), name=name
            )

        def delete_index(self, name):
            calls.setdefault("deleted", []).append(name)
            calls["existing"] = [n for n in calls["existing"] if n != name]

    pinecone = types.ModuleType("pinecone")
    pinecone.ServerlessSpec = lambda **kw: ("spec", kw)
    grpc = types.ModuleType("pinecone.grpc")
    grpc.PineconeGRPC = _Client
    pinecone.grpc = grpc

    vectorstores = types.ModuleType("langchain_pinecone")
    vectorstores.PineconeVectorStore = _Store

    helper = types.ModuleType("src.helper")
    helper.load_pdf = lambda path: [object()]
    helper.download_hugging_face_embeddings = lambda: "embeddings"

    chunking = types.ModuleType("src.chunking")
    # seven chunks, one of them a duplicate of another
    chunking.text_split = lambda pages: [_Chunk(t) for t in "abcdefa"]

    for name, module in (
        ("pinecone", pinecone),
        ("pinecone.grpc", grpc),
        ("langchain_pinecone", vectorstores),
        ("src.helper", helper),
        ("src.chunking", chunking),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.delitem(sys.modules, "store_index", raising=False)
    module = importlib.import_module("store_index")
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(module, "BATCH_SIZE", 3)
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    yield module, calls
    sys.modules.pop("store_index", None)


def test_the_index_is_created_when_absent(indexer):
    module, calls = indexer
    calls["existing"] = []
    module.main([])
    assert len(calls["created"]) == 1
    assert calls["created"][0]["name"] == module.INDEX_NAME
    assert calls["created"][0]["dimension"] == module.EMBED_DIM


def test_creation_is_skipped_when_the_index_exists(indexer):
    """Creating it again is a 409, which used to abort the whole script."""
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main([])
    assert calls["created"] == []
    assert calls["upserts"], "it should still upsert into the existing index"


def test_the_chunks_are_upserted_in_batches(indexer):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main([])
    sizes = [len(docs) for docs, _ in calls["upserts"]]
    assert sizes == [3, 3, 1], "seven chunks at BATCH_SIZE=3"


def test_every_chunk_is_upserted_exactly_once(indexer):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main([])
    sent = [text for docs, _ in calls["upserts"] for text in docs]
    assert sent == list("abcdefa")


def test_ids_stay_aligned_with_their_chunks(indexer):
    """A mismatch here labels a vector with another chunk's id, silently."""
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main([])
    from src.text import chunk_ids

    for documents, ids in calls["upserts"]:
        assert ids == chunk_ids([_Chunk(t) for t in documents])


def test_duplicate_chunks_share_an_id(indexer):
    """The two 'a' chunks must collapse onto one vector, not two."""
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main([])
    pairs = {
        text: i
        for docs, ids in calls["upserts"]
        for text, i in zip(docs, ids, strict=True)
    }
    all_ids = [i for _, ids in calls["upserts"] for i in ids]
    assert len(set(all_ids)) == 6, "six distinct texts among seven chunks"
    assert pairs["a"] in all_ids


def test_the_api_key_reaches_the_client(indexer):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main([])
    assert calls["api_key"] == "test-key"


def test_limit_restricts_what_is_upserted(indexer):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main(["--limit", "2"])
    sent = [text for docs, _ in calls["upserts"] for text in docs]
    assert sent == ["a", "b"]


def test_limit_above_the_chunk_count_is_harmless(indexer):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main(["--limit", "500"])
    sent = [text for docs, _ in calls["upserts"] for text in docs]
    assert sent == list("abcdefa")


def test_limit_is_reported(indexer, capsys):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main(["--limit", "2"])
    assert "limiting to the first 2 chunks" in capsys.readouterr().out


def test_a_nonsensical_limit_is_rejected(indexer):
    module, _ = indexer
    with pytest.raises(SystemExit, match="--limit"):
        module.main(["--limit", "0"])


def test_limit_combines_with_dry_run(indexer, capsys, monkeypatch):
    """Check the chunking of the first few without touching Pinecone."""
    module, calls = indexer
    module.main(["--limit", "3", "--dry-run"])
    assert calls["upserts"] == []
    assert "would upsert 3 unique chunks" in capsys.readouterr().out


def test_limit_defaults_to_everything(indexer):
    module, _ = indexer
    assert module.parse_args([]).limit is None


def test_recreate_deletes_then_rebuilds(indexer):
    """The only way to clear vectors written under ids we no longer generate."""
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main(["--recreate", "--yes"])
    assert calls.get("deleted") == [module.INDEX_NAME]
    assert len(calls["created"]) == 1
    assert calls["upserts"], "it must repopulate what it deleted"


def test_recreate_on_a_missing_index_just_creates_it(indexer):
    module, calls = indexer
    calls["existing"] = []
    module.main(["--recreate", "--yes"])
    assert calls.get("deleted") is None
    assert len(calls["created"]) == 1


def test_without_recreate_the_index_is_left_alone(indexer):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main([])
    assert calls.get("deleted") is None
    assert calls["created"] == []


def test_recreate_reports_the_deletion(indexer, capsys):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main(["--recreate", "--yes"])
    assert "deleting index" in capsys.readouterr().out


def test_a_dry_run_never_deletes(indexer):
    """--dry-run returns before Pinecone is contacted at all."""
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    module.main(["--recreate", "--dry-run"])
    assert calls.get("deleted") is None
    assert calls["upserts"] == []


def test_a_dimension_mismatch_stops_before_upserting(indexer):
    """Pinecone would reject each batch instead, after all the slow work."""
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    calls["dimension"] = 768
    with pytest.raises(SystemExit) as caught:
        module.main([])
    message = str(caught.value)
    assert "768" in message and str(module.EMBED_DIM) in message
    assert calls["upserts"] == [], "nothing may be written on a mismatch"


def test_the_mismatch_message_says_what_to_do(indexer):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    calls["dimension"] = 1024
    with pytest.raises(SystemExit, match="--recreate"):
        module.main([])


def test_a_matching_dimension_proceeds(indexer):
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    calls["dimension"] = module.EMBED_DIM
    module.main([])
    assert calls["upserts"]


def test_the_dimension_is_not_checked_on_a_fresh_index(indexer):
    """There is nothing to disagree with; it is created at EMBED_DIM."""
    module, calls = indexer
    calls["existing"] = []
    calls["dimension"] = 768
    module.main([])
    assert calls["created"][0]["dimension"] == module.EMBED_DIM


def test_recreate_sidesteps_a_mismatch(indexer):
    """The documented escape: delete the old index and build it correctly."""
    module, calls = indexer
    calls["existing"] = [module.INDEX_NAME]
    calls["dimension"] = 768
    module.main(["--recreate", "--yes"])
    assert calls["created"][0]["dimension"] == module.EMBED_DIM


class TestRecreateConfirmation:
    """--recreate deletes a live index; it should be hard to do by accident."""

    def test_a_matching_index_name_confirms(self, indexer, monkeypatch):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *a: module.INDEX_NAME)
        module.main(["--recreate"])
        assert calls.get("deleted") == [module.INDEX_NAME]

    def test_anything_else_aborts_without_deleting(self, indexer, monkeypatch):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *a: "yes")
        with pytest.raises(SystemExit, match="not confirmed"):
            module.main(["--recreate"])
        assert calls.get("deleted") is None

    def test_an_empty_answer_aborts(self, indexer, monkeypatch):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *a: "")
        with pytest.raises(SystemExit):
            module.main(["--recreate"])
        assert calls.get("deleted") is None

    def test_yes_skips_the_prompt(self, indexer, monkeypatch):
        def refuse(*args):
            raise AssertionError("--yes must not prompt")

        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        monkeypatch.setattr("builtins.input", refuse)
        module.main(["--recreate", "--yes"])
        assert calls.get("deleted") == [module.INDEX_NAME]

    def test_without_a_terminal_it_refuses_rather_than_hanging(
        self, indexer, monkeypatch
    ):
        """A script piping stdin would otherwise block on input() forever."""
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(SystemExit, match="--yes"):
            module.main(["--recreate"])
        assert calls.get("deleted") is None

    def test_nothing_is_asked_when_there_is_no_index_to_delete(
        self, indexer, monkeypatch
    ):
        def refuse(*args):
            raise AssertionError("there is nothing to confirm")

        module, calls = indexer
        calls["existing"] = []
        monkeypatch.setattr("builtins.input", refuse)
        module.main(["--recreate"])
        assert calls["created"]

    def test_a_plain_run_never_prompts(self, indexer, monkeypatch):
        def refuse(*args):
            raise AssertionError("only --recreate is destructive")

        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        monkeypatch.setattr("builtins.input", refuse)
        module.main([])
        assert calls["upserts"]
