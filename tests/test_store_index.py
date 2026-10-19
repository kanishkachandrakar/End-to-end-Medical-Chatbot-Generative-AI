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
    loaders = types.ModuleType("src.loaders")
    loaders.load_pdf = lambda path: [object()] * 2
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
        ("src.loaders", loaders),
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


def test_a_dry_run_needs_no_key_and_touches_nothing(store_index, monkeypatch, capsys):
    """The whole point: check the chunking before setting anything up.

    Two tests used to make this call, with different names and identical
    bodies, and neither asserted anything -- they relied on the fixture's
    stubs raising if Pinecone or the model were reached, which is a real
    guarantee but an invisible one. The output assertion is what shows the
    call ran the whole way through rather than returning early for some
    other reason.
    """
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    store_index.main(["--dry-run"])
    assert "dry run: would upsert" in capsys.readouterr().out


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

        def Index(self, name):  # noqa: N802 - matches the real client
            calls.setdefault("stats_reads", []).append(name)
            return types.SimpleNamespace(
                describe_index_stats=lambda: {
                    "total_vector_count": calls.get("reported", 6)
                }
            )

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
    loaders = types.ModuleType("src.loaders")
    loaders.load_pdf = lambda path: [object()]
    helper.download_hugging_face_embeddings = lambda: "embeddings"

    chunking = types.ModuleType("src.chunking")
    # seven chunks, one of them a duplicate of another
    chunking.text_split = lambda pages: [_Chunk(t) for t in "abcdefa"]

    for name, module in (
        ("pinecone", pinecone),
        ("pinecone.grpc", grpc),
        ("langchain_pinecone", vectorstores),
        ("src.helper", helper),
        ("src.loaders", loaders),
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


class TestConfirmUpsert:
    """The script's last word should be what arrived, not what it sent."""

    def test_it_reports_the_count_once_it_matches(self, indexer, capsys):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        calls["reported"] = 6
        module.main([])
        assert "now reports 6 vectors" in capsys.readouterr().out

    def test_a_lagging_count_is_explained_not_treated_as_failure(
        self, indexer, capsys, monkeypatch
    ):
        """Pinecone is eventually consistent; a short count is usually normal."""
        module, calls = indexer
        monkeypatch.setattr(module.time, "sleep", lambda s: None)
        client = module.Pinecone(api_key="k")
        calls["reported"] = 0
        module.confirm_upsert(client, expected=6, timeout=0.01)
        output = capsys.readouterr().out
        assert "0 of 6 vectors" in output
        assert "eventually consistent" in output

    def test_it_does_not_raise_on_a_short_count(self, indexer, monkeypatch):
        module, calls = indexer
        monkeypatch.setattr(module.time, "sleep", lambda s: None)
        calls["reported"] = 1
        module.confirm_upsert(module.Pinecone(api_key="k"), expected=99, timeout=0.01)

    def test_a_higher_count_than_expected_is_accepted(self, indexer, capsys):
        """An index with other data in it is not this script's business."""
        module, calls = indexer
        calls["reported"] = 100
        module.confirm_upsert(module.Pinecone(api_key="k"), expected=6, timeout=1)
        assert "now reports 100 vectors" in capsys.readouterr().out

    def test_a_dry_run_never_asks(self, indexer, calls_check=None):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        module.main(["--dry-run"])
        assert calls.get("stats_reads") is None


class TestProgressReporting:
    """A rebuild takes minutes; silence for minutes is indistinguishable
    from being stuck."""

    def test_the_load_reports_how_long_it_took(self, indexer, capsys):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        module.main([])
        assert "chunks in " in capsys.readouterr().out

    def test_each_batch_reports_a_rate_and_an_estimate(self, indexer, capsys):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        module.main([])
        output = capsys.readouterr().out
        assert "/s" in output and "s left" in output

    def test_the_counts_still_add_up_to_the_total(self, indexer, capsys):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        module.main([])
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "chunks  " in ln]
        assert lines, "no progress lines"
        assert lines[-1].strip().startswith("7/7")

    def test_the_model_load_is_timed_separately(self, indexer, capsys):
        """It is the slow part of a first run and worth distinguishing."""
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        module.main([])
        assert "embedding model ready in" in capsys.readouterr().out

    def test_a_dry_run_times_the_load_but_not_the_model(self, indexer, capsys):
        module, calls = indexer
        module.main(["--dry-run"])
        output = capsys.readouterr().out
        assert "chunks in " in output
        assert "embedding model ready" not in output


class TestPrepareIndex:
    """The part that refuses, called on its own rather than through main()."""

    def test_it_returns_a_client_for_an_existing_index(self, indexer):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        client = module.prepare_index("key", recreate=False, yes=False)
        assert client is not None
        assert calls["created"] == []

    def test_it_creates_a_missing_index(self, indexer):
        module, calls = indexer
        calls["existing"] = []
        module.prepare_index("key", recreate=False, yes=False)
        assert calls["created"][0]["dimension"] == module.EMBED_DIM

    def test_recreate_deletes_first_when_confirmation_is_skipped(self, indexer):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        module.prepare_index("key", recreate=True, yes=True)
        assert calls["deleted"] == [module.INDEX_NAME]
        assert calls["created"]

    def test_a_dimension_mismatch_refuses_here(self, indexer):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        calls["dimension"] = 768
        with pytest.raises(SystemExit, match="768"):
            module.prepare_index("key", recreate=False, yes=False)

    def test_the_api_key_is_passed_through(self, indexer):
        module, calls = indexer
        calls["existing"] = [module.INDEX_NAME]
        module.prepare_index("given-key", recreate=False, yes=False)
        assert calls["api_key"] == "given-key"


class TestConfirmDeletion:
    """Separated from prepare_index so the prompt can be tested as a prompt."""

    def test_the_index_name_confirms(self, indexer, monkeypatch):
        module, _ = indexer
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *a: module.INDEX_NAME)
        module.confirm_deletion(module.INDEX_NAME)

    def test_anything_else_aborts(self, indexer, monkeypatch):
        module, _ = indexer
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *a: "yes")
        with pytest.raises(SystemExit, match="not confirmed"):
            module.confirm_deletion(module.INDEX_NAME)

    def test_without_a_terminal_it_refuses_rather_than_waiting(
        self, indexer, monkeypatch
    ):
        module, _ = indexer
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        def refuse(*args):
            raise AssertionError("it must not reach input() with no terminal")

        monkeypatch.setattr("builtins.input", refuse)
        with pytest.raises(SystemExit, match="--yes"):
            module.confirm_deletion(module.INDEX_NAME)


class TestUpsertChunks:
    """The embed-and-upload, called directly rather than through main()."""

    def test_it_sends_every_chunk_once(self, indexer):
        module, calls = indexer
        chunks = [_Chunk(t) for t in "abcdef"]
        ids = [f"id{i}" for i in range(6)]
        module.upsert_chunks("embeddings", chunks, ids)
        sent = [text for batch, _ in calls["upserts"] for text in batch]
        assert sent == list("abcdef")

    def test_it_keeps_the_ids_with_their_chunks(self, indexer):
        """Two sequences zipped together; drifting apart mislabels every vector."""
        module, calls = indexer
        chunks = [_Chunk(t) for t in "abcdefg"]
        ids = [f"id{i}" for i in range(7)]
        module.upsert_chunks("embeddings", chunks, ids)
        for batch, batch_ids in calls["upserts"]:
            assert len(batch) == len(batch_ids)
        flattened = [i for _, batch_ids in calls["upserts"] for i in batch_ids]
        assert flattened == ids

    def test_a_mismatched_length_raises_rather_than_truncating(self, indexer):
        """strict=True on the zip: silently dropping the tail is the bad case."""
        module, _ = indexer
        with pytest.raises(ValueError):
            module.upsert_chunks("embeddings", [_Chunk("a")] * 5, ["only-one"])

    def test_it_reports_a_rate_and_an_estimate(self, indexer, capsys):
        module, _ = indexer
        ids = [f"id{i}" for i in range(4)]
        module.upsert_chunks("embeddings", [_Chunk("a")] * 4, ids)
        output = capsys.readouterr().out
        assert "/s" in output and "s left" in output

    def test_an_empty_list_sends_nothing(self, indexer):
        module, calls = indexer
        module.upsert_chunks("embeddings", [], [])
        assert calls["upserts"] == []

    def test_it_opens_the_configured_index(self, indexer):
        module, calls = indexer
        module.upsert_chunks("embeddings", [_Chunk("a")], ["id0"])
        assert calls["opened"] == [module.INDEX_NAME]
