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
