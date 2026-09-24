"""PDF loading, chunking and the embedding model used for the index."""

from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL


def load_pdf(data_dir: str) -> List[Document]:
    """Load every top-level *.pdf in ``data_dir``, one Document per page."""
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def text_split(extracted_data: List[Document]) -> List[Document]:
    """Split page Documents into overlapping chunks small enough to embed."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(extracted_data)


def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """Build the embedding model, downloading the weights on first use."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
