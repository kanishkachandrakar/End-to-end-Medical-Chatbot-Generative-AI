"""The embedding model.

Kept as the module the Dockerfile prefetches from, so the image's warm-up step
and the application agree on one entry point.

langchain-huggingface rather than langchain-community: the latter is sunset and
deprecated this class in favour of the standalone package. Same class name,
same ``model_name``, same weights -- and the vectors already in Pinecone stay
valid, which CI confirms by embedding a word inside the image and comparing the
length against EMBED_DIM.

PDF loading lives in src/loaders.py so that importing this does not drag a
sunset package into the web app.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBED_MODEL


def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """Build the embedding model, downloading the weights on first use."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
