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
