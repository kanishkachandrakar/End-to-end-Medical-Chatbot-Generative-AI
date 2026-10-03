"""Vector ids decide whether re-indexing overwrites or duplicates."""

import hashlib

import pytest

from src.text import batched, chunk_ids


class Chunk:
    """Stands in for a LangChain Document, which is all chunk_ids touches."""

    def __init__(self, page_content):
        self.page_content = page_content


def test_same_text_gives_the_same_id():
    """The property the whole design rests on: re-running is an overwrite."""
    assert chunk_ids([Chunk("acne")]) == chunk_ids([Chunk("acne")])


def test_different_text_gives_different_ids():
    first, second = chunk_ids([Chunk("acne"), Chunk("asthma")])
    assert first != second


def test_duplicate_chunks_collapse_onto_one_id():
    """Six copies of a chunk should occupy one vector, not six."""
    ids = chunk_ids([Chunk("same")] * 6)
    assert len(set(ids)) == 1


def test_id_does_not_depend_on_position():
    a = chunk_ids([Chunk("x"), Chunk("y")])
    b = chunk_ids([Chunk("y"), Chunk("x")])
    assert a == list(reversed(b))


def test_matches_a_plain_sha1_of_the_content():
    """Pins the scheme: changing it silently orphans every existing vector."""
    assert chunk_ids([Chunk("acne")]) == [hashlib.sha1(b"acne").hexdigest()]


def test_handles_non_ascii_content():
    """The book contains accented terms; encoding must not raise."""
    ids = chunk_ids([Chunk("Ménière's disease")])
    assert len(ids) == 1 and len(ids[0]) == 40


def test_empty_input():
    assert chunk_ids([]) == []


class TestBatched:
    """Slicing the upsert into batches; an off-by-one silently skips chunks."""

    def test_splits_into_full_batches(self):

        assert [list(b) for b in batched([1, 2, 3, 4], 2)] == [[1, 2], [3, 4]]

    def test_the_last_batch_is_short(self):

        assert [list(b) for b in batched([1, 2, 3], 2)] == [[1, 2], [3]]

    def test_every_item_appears_exactly_once(self):

        for total in (0, 1, 249, 250, 251, 5860):
            items = list(range(total))
            flattened = [x for batch in batched(items, 250) for x in batch]
            assert flattened == items, total

    def test_a_batch_larger_than_the_input_yields_one_batch(self):

        assert [list(b) for b in batched([1, 2], 100)] == [[1, 2]]

    def test_empty_input_yields_nothing(self):

        assert list(batched([], 10)) == []

    def test_a_zero_size_is_rejected(self):


        with pytest.raises(ValueError):
            list(batched([1], 0))

    def test_documents_and_ids_batch_in_lockstep(self):
        """store_index zips these two; a mismatch would mislabel every vector."""

        docs = list(range(7))
        ids = [f"id{i}" for i in range(7)]
        pairs = list(zip(batched(docs, 3), batched(ids, 3), strict=True))
        assert [len(d) for d, _ in pairs] == [3, 3, 1]
        assert all(len(d) == len(i) for d, i in pairs)
