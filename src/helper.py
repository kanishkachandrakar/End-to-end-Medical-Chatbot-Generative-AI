"""Loading PDFs and building the embedding model.

Both ends of this module are heavy: DirectoryLoader needs langchain-community
and HuggingFaceEmbeddings pulls in torch. Chunking lives in src/chunking.py so
it can be imported without either.
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import EMBED_MODEL


def load_pdf(data_dir: str) -> list[Document]:
    """Load every top-level *.pdf in ``data_dir``, one Document per page."""
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """Build the embedding model, downloading the weights on first use."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
