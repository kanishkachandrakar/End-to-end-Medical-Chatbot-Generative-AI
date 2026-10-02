"""Splitting loaded pages into chunks.

Separate from src/helper.py on purpose. Chunking needs only
langchain-text-splitters, which is pure Python, whereas loading PDFs pulls in
langchain-community and building the embedding model pulls in torch. Keeping
them apart means this -- the part with behaviour worth asserting -- can be
imported and tested without either.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def text_split(extracted_data: list[Document]) -> list[Document]:
    """Split page Documents into overlapping chunks small enough to embed."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(extracted_data)
