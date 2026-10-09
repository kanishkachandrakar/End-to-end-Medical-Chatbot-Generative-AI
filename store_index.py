import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone

from src.chunking import text_split
from src.config import (
    EMBED_DIM,
    EMBED_MODEL,
    INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
)
from src.helper import download_hugging_face_embeddings, load_pdf
from src.text import batched, chunk_ids

# Chunks per upsert. Small enough to see progress and to keep the embedding
# batch off the heap, large enough not to pay request overhead 5,000 times.
BATCH_SIZE = 250


def parse_args(argv=None) -> argparse.Namespace:
    """Command line for the indexer."""
    parser = argparse.ArgumentParser(
        description="Chunk the PDFs in Data/ and upsert them into Pinecone."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete the index first, then rebuild it. The only way to clear "
        "vectors written under ids this script no longer generates -- notably "
        "the duplicates left by the notebook's upsert cell.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation that --recreate asks for. Required when "
        "running without a terminal, such as in a script.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="index only the first N chunks. A cheap end-to-end check of the "
        "whole pipeline -- embedding and upsert included -- before committing "
        "to a full rebuild.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load and chunk the PDFs, report the counts, and touch nothing "
        "in Pinecone. Use this to check the chunking before a rebuild.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Chunk the PDFs in Data/, create the index if needed and upsert them."""
    args = parse_args(argv)
    load_dotenv()

    api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key and not args.dry_run:
        raise SystemExit(
            "PINECONE_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    extracted_data = load_pdf("Data/")
    text_chunks = text_split(extracted_data)
    print(f"loaded {len(extracted_data)} pages -> {len(text_chunks)} chunks")

    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        text_chunks = text_chunks[: args.limit]
        print(f"limiting to the first {len(text_chunks)} chunks")

    ids = chunk_ids(text_chunks)
    unique = len(set(ids))
    if unique != len(ids):
        print(f"note: {len(ids) - unique} chunks are byte-identical and will collapse")

    if args.dry_run:
        print(f"dry run: would upsert {unique} unique chunks into {INDEX_NAME!r}")
        return

    embeddings = download_hugging_face_embeddings()

    pc = Pinecone(api_key=api_key)

    exists = INDEX_NAME in pc.list_indexes().names()

    if exists and args.recreate and not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                f"--recreate would delete the index {INDEX_NAME!r} and there is "
                "no terminal to confirm at. Pass --yes if that is intended."
            )
        print(f"This deletes the index {INDEX_NAME!r} and everything in it.")
        if input("Type the index name to confirm: ").strip() != INDEX_NAME:
            raise SystemExit("not confirmed, nothing was changed")

    if exists and args.recreate:
        # Deterministic ids make a re-run an overwrite, but only for ids this
        # script would generate. Anything upserted under a random uuid -- every
        # duplicate the notebook left behind -- can only be removed with the
        # index itself.
        print(f"deleting index {INDEX_NAME!r}")
        pc.delete_index(INDEX_NAME)
        exists = False

    if exists:
        print(f"index {INDEX_NAME!r} already exists, skipping creation")
        # Upserting a differently-sized vector is rejected by Pinecone per
        # batch, after the PDF has been parsed and the model loaded, with an
        # error that names neither side of the mismatch. Check it up front.
        existing_dim = pc.describe_index(INDEX_NAME).dimension
        if existing_dim != EMBED_DIM:
            raise SystemExit(
                f"index {INDEX_NAME!r} has {existing_dim} dimensions but "
                f"{EMBED_MODEL} produces {EMBED_DIM}. Either point EMBED_MODEL "
                f"at a {existing_dim}-dimensional model, or rebuild the index "
                f"with --recreate."
            )
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


    store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    total = len(text_chunks)
    done = 0
    for chunks, batch_ids in zip(
        batched(text_chunks, BATCH_SIZE), batched(ids, BATCH_SIZE), strict=True
    ):
        store.add_documents(chunks, ids=batch_ids)
        done += len(chunks)
        print(f"  {done}/{total} chunks", flush=True)

    print(f"upserted {unique} unique chunks into {INDEX_NAME!r}")


if __name__ == "__main__":
    main()
