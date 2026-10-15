"""Reading the source PDFs.

The only module that imports langchain-community, which is sunset and says so
on import: the package is no longer actively maintained and its contents are
moving to standalone integrations. The embeddings half of this already went to
langchain-huggingface; DirectoryLoader and PyPDFLoader have no standalone home
yet, so they stay here, isolated, with an upper bound in requirements.txt at
the 1.0 boundary the deprecation notice names.

Separate from src/helper.py so that importing the embedding model -- which the
web app does -- no longer pulls a sunset package in. Only store_index.py reads
PDFs, and that runs locally rather than in the container.
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document


def load_pdf(data_dir: str) -> list[Document]:
    """Load every top-level *.pdf in ``data_dir``, one Document per page."""
    # The ignore below is upstream's fault: langchain_community annotates
    # loader_cls with a union of four loaders that excludes PyPDFLoader, even
    # though passing it is the documented way to load PDFs.
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,  # type: ignore[arg-type]
    )
    return loader.load()
