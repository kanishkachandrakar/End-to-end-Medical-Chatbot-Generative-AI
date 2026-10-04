"""load_pdf is the only module boundary that touches the filesystem."""

import pathlib

import pytest

# Not the assertions -- the langchain_community import is what is slow.
pytestmark = pytest.mark.slow

pytest.importorskip("langchain_community")
pytest.importorskip("pypdf")

from pypdf import PdfWriter  # noqa: E402

from src.helper import load_pdf  # noqa: E402


def _write_pdf(path: pathlib.Path, pages: int = 1) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)


def test_an_empty_directory_yields_no_documents(tmp_path):
    assert load_pdf(str(tmp_path)) == []


def test_one_document_per_page(tmp_path):
    _write_pdf(tmp_path / "book.pdf", pages=3)
    assert len(load_pdf(str(tmp_path))) == 3


def test_the_source_path_is_recorded(tmp_path):
    """This metadata is the only link from an answer back to its page."""
    _write_pdf(tmp_path / "book.pdf")
    document = load_pdf(str(tmp_path))[0]
    assert document.metadata["source"].endswith("book.pdf")
    assert "page" in document.metadata


def test_several_pdfs_are_all_loaded(tmp_path):
    _write_pdf(tmp_path / "one.pdf")
    _write_pdf(tmp_path / "two.pdf", pages=2)
    assert len(load_pdf(str(tmp_path))) == 3


def test_non_pdf_files_are_ignored(tmp_path):
    _write_pdf(tmp_path / "book.pdf")
    (tmp_path / "notes.txt").write_text("not a pdf")
    (tmp_path / "data.csv").write_text("a,b")
    assert len(load_pdf(str(tmp_path))) == 1


def test_subdirectories_are_not_searched(tmp_path):
    """The glob is *.pdf, not **/*.pdf -- as the docstring claims."""
    nested = tmp_path / "chapters"
    nested.mkdir()
    _write_pdf(nested / "buried.pdf")
    assert load_pdf(str(tmp_path)) == []
