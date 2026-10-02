"""Chunk size and overlap decide what a retrieved passage can contain."""

from langchain_core.documents import Document

from src.chunking import text_split
from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def _page(text, page=1):
    return Document(page_content=text, metadata={"source": "book.pdf", "page": page})


def test_short_page_stays_one_chunk():
    chunks = text_split([_page("Acne is a skin condition.")])
    assert len(chunks) == 1
    assert chunks[0].page_content == "Acne is a skin condition."


def test_long_page_is_split():
    chunks = text_split([_page("word " * 400)])
    assert len(chunks) > 1


def test_no_chunk_exceeds_the_configured_size():
    """Oversized chunks waste the context window the prompt is budgeted for."""
    chunks = text_split([_page("sentence. " * 500)])
    assert chunks
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)


def test_metadata_is_carried_onto_every_chunk():
    """Answers are only traceable to a page because this survives splitting."""
    chunks = text_split([_page("word " * 400, page=42)])
    assert all(c.metadata["page"] == 42 for c in chunks)
    assert all(c.metadata["source"] == "book.pdf" for c in chunks)


def test_empty_input_gives_no_chunks():
    assert text_split([]) == []


def test_pages_are_split_independently():
    chunks = text_split([_page("alpha", 1), _page("beta", 2)])
    assert [c.metadata["page"] for c in chunks] == [1, 2]


def test_overlap_is_smaller_than_the_chunk():
    """Equal values would make the splitter loop forever on long input."""
    assert 0 <= CHUNK_OVERLAP < CHUNK_SIZE
