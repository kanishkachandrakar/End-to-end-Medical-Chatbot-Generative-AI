"""Vector ids decide whether re-indexing overwrites or duplicates."""

import hashlib

from src.text import chunk_ids


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
