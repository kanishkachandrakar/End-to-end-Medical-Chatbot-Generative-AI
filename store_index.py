from src.config import (
    EMBED_DIM,
    INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
)
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import hashlib
import os


def main() -> None:
    """Chunk the PDFs in Data/, create the index if needed and upsert them."""
    load_dotenv()

    api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        raise SystemExit(
            "PINECONE_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    extracted_data = load_pdf("Data/")
    text_chunks = text_split(extracted_data)
    print(f"loaded {len(extracted_data)} pages -> {len(text_chunks)} chunks")

    embeddings = download_hugging_face_embeddings()

    pc = Pinecone(api_key=api_key)

    if INDEX_NAME in pc.list_indexes().names():
        print(f"index {INDEX_NAME!r} already exists, skipping creation")
    else:
        print(f"creating index {INDEX_NAME!r}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

    # Derive each vector's id from the chunk text so that re-running this
    # script overwrites the previous upsert instead of adding a second copy
    # under a fresh uuid. Without this, every run multiplied the index: a
    # top-k retrieval then returned k copies of one chunk rather than k
    # different ones.
    ids = [
        hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
        for chunk in text_chunks
    ]
    unique = len(set(ids))
    if unique != len(ids):
        print(f"note: {len(ids) - unique} chunks are byte-identical and will collapse")

    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=INDEX_NAME,
        embedding=embeddings,
        ids=ids,
    )

    print(f"upserted {unique} unique chunks into {INDEX_NAME!r}")


if __name__ == "__main__":
    main()
