import argparse
import os
import pathlib
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.chunking import text_split
from src.config import (
    EMBED_DIM,
    EMBED_MODEL,
    INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
)
from src.helper import download_hugging_face_embeddings
from src.loaders import load_pdf
from src.text import batched, chunk_ids

# The PDFs, resolved against this file rather than the working directory.
# `load_pdf("Data/")` meant the script only worked from the repository root,
# and from anywhere else it died inside langchain's DirectoryLoader with
# "FileNotFoundError: Directory not found: 'Data/'" under a stack of frames.
# It is the one command this project's README still asks an operator to run.
DATA_DIR = pathlib.Path(__file__).resolve().parent / "Data"

# Chunks per upsert. Small enough to see progress and to keep the embedding
# batch off the heap, large enough not to pay request overhead 5,000 times.
BATCH_SIZE = 250

# Seconds to wait for an index to become ready, or to finish being deleted.
# Both operations block by default and, passed no timeout, block for as long
# as it takes -- which for a control plane that has stopped answering means
# forever, with nothing printed. That is the state confirm_deletion already
# refuses to create: indistinguishable from the slow indexing it precedes.
# Serverless creation is usually a few seconds; a minute is generous.
INDEX_READY_TIMEOUT = 60


def _at_least_one(value: str) -> int:
    """A --limit argparse can reject itself, before any work happens.

    The check used to live in main(), after load_pdf and text_split -- so
    `--limit 0` spent the full PDF parse before saying the flag was wrong.
    prepare_index is placed ahead of the PDF for exactly this reason; an
    argument that cannot be right belongs further forward still.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {number}")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        type=_at_least_one,
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


def _bounded(operation: Callable[[], Any], what: str) -> None:
    """Run a control-plane call that waits, and say so if the wait runs out.

    The client's own timeout turns an unbounded block into an exception; this
    turns that exception into a sentence naming the index and what to do,
    rather than a traceback from inside the SDK's polling loop.
    """
    try:
        operation()
    except Exception as exc:
        raise SystemExit(
            f"{what} after {INDEX_READY_TIMEOUT}s: {type(exc).__name__}: {exc}. "
            "Nothing has been upserted. Check the Pinecone console and re-run; "
            "the operation may have completed on their side."
        ) from exc


def confirm_deletion(index_name: str) -> None:
    """Make --recreate hard to do by accident.

    Refuses rather than blocking when there is no terminal: a script with its
    stdin redirected would otherwise wait on input() indefinitely, which looks
    exactly like the slow indexing it is supposed to be doing.
    """
    if not sys.stdin.isatty():
        raise SystemExit(
            f"--recreate would delete the index {index_name!r} and there is "
            "no terminal to confirm at. Pass --yes if that is intended."
        )
    print(f"This deletes the index {index_name!r} and everything in it.")
    if input("Type the index name to confirm: ").strip() != index_name:
        raise SystemExit("not confirmed, nothing was changed")


def require_data_dir() -> None:
    """Refuse early if there are no PDFs to read.

    In the same position as prepare_index's refusals and for the same reason:
    everything that can say no should say it before the minutes of parsing and
    model loading, and it should say it in a sentence. Without this a deleted
    or renamed Data/ surfaces as a FileNotFoundError from inside
    DirectoryLoader, which names the relative path and not what to do.
    """
    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"no PDFs to index: {DATA_DIR} is not a directory. The source "
            "book lives there in this repository; restore it, or point "
            "store_index.py at another copy."
        )
    if not any(DATA_DIR.glob("*.pdf")):
        raise SystemExit(
            f"no PDFs to index: {DATA_DIR} holds no *.pdf files. Chunking "
            "would produce nothing and the upsert would leave the index empty."
        )


def prepare_index(api_key: str, *, recreate: bool, yes: bool) -> Pinecone:
    """Leave the index in a state the upsert can use, or refuse.

    The two flags are keyword-only. `recreate` decides whether the index is
    deleted and `yes` decides whether that deletion is confirmed first, so
    prepare_index(key, True, True) and prepare_index(key, True, False) differ
    by whether anyone is asked before the data goes -- which is not a
    distinction to leave resting on argument order.

    Everything that can say no lives here -- an unreachable Pinecone, a
    declined confirmation, a dimension that disagrees with the embedding model
    -- and main() calls it before opening the PDF, because each of these used
    to cost minutes of parsing and model loading before reporting.
    """
    pc = Pinecone(api_key=api_key)
    exists = INDEX_NAME in pc.list_indexes().names()

    if exists and recreate:
        if not yes:
            confirm_deletion(INDEX_NAME)
        # Deterministic ids make a re-run an overwrite, but only for ids this
        # script would generate. Anything upserted under a random uuid -- every
        # duplicate the notebook left behind -- can only go with the index.
        print(f"deleting index {INDEX_NAME!r}", flush=True)
        _bounded(
            lambda: pc.delete_index(INDEX_NAME, timeout=INDEX_READY_TIMEOUT),
            f"index {INDEX_NAME!r} was not gone",
        )
        exists = False

    if exists:
        print(f"index {INDEX_NAME!r} already exists, skipping creation")
        # Pinecone rejects a differently-sized vector per batch, with an error
        # naming neither side of the mismatch. Compare it here instead.
        existing_dim = pc.describe_index(INDEX_NAME).dimension
        if existing_dim != EMBED_DIM:
            raise SystemExit(
                f"index {INDEX_NAME!r} has {existing_dim} dimensions but "
                f"{EMBED_MODEL} produces {EMBED_DIM}. Either point EMBED_MODEL "
                f"at a {existing_dim}-dimensional model, or rebuild the index "
                f"with --recreate."
            )
    else:
        print(f"creating index {INDEX_NAME!r}", flush=True)
        _bounded(
            lambda: pc.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                timeout=INDEX_READY_TIMEOUT,
            ),
            f"index {INDEX_NAME!r} was not ready",
        )

    return pc


def upsert_chunks(
    embeddings: Any, chunks: list[Any], ids: list[str]
) -> None:
    """Embed and upload the chunks, reporting progress as it goes.

    A rebuild is several minutes of work. The rate and the estimate are what
    distinguish it from being stuck, which is otherwise indistinguishable from
    silence.
    """
    store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    total = len(chunks)
    done = 0
    started = time.monotonic()
    for batch, batch_ids in zip(
        batched(chunks, BATCH_SIZE), batched(ids, BATCH_SIZE), strict=True
    ):
        store.add_documents(batch, ids=batch_ids)
        done += len(batch)
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed else 0
        remaining = (total - done) / rate if rate else 0
        print(
            f"  {done}/{total} chunks  {rate:.0f}/s  ~{remaining:.0f}s left",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Chunk the PDFs in Data/, create the index if needed and upsert them."""
    args = parse_args(argv)
    load_dotenv()
    require_data_dir()

    # A dry run answers a question about the PDF, so it needs none of this:
    # no key, no connection, no index created, no confirmation prompt.
    pc: Pinecone | None = None
    if not args.dry_run:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise SystemExit(
                "PINECONE_API_KEY is not set. Copy .env.example to .env and "
                "fill it in."
            )
        pc = prepare_index(api_key, recreate=args.recreate, yes=args.yes)

    started = time.monotonic()
    extracted_data = load_pdf(str(DATA_DIR))
    text_chunks = text_split(extracted_data)
    print(
        f"loaded {len(extracted_data)} pages -> {len(text_chunks)} chunks "
        f"in {time.monotonic() - started:.0f}s"
    )

    if args.limit is not None:
        text_chunks = text_chunks[: args.limit]
        print(
            f"limiting to the first {len(text_chunks)} chunks -- this leaves "
            f"the index holding only these, so a rebuild is still needed "
            f"before the app can answer from the whole book"
        )

    ids = chunk_ids(text_chunks)
    unique = len(set(ids))
    if unique != len(ids):
        print(f"note: {len(ids) - unique} chunks are byte-identical and will collapse")

    if args.dry_run:
        print(f"dry run: would upsert {unique} unique chunks into {INDEX_NAME!r}")
        return

    # Past the dry-run return above, so prepare_index has run and the client
    # exists. Stated rather than assumed: mypy cannot see the connection
    # between the two branches, and neither could a reader.
    #
    # A raise rather than an assert, because `python -O` strips asserts and
    # this is the last thing standing between a None client and an
    # AttributeError thirty lines into the upsert.
    if pc is None:  # pragma: no cover - unreachable past the dry-run return
        raise RuntimeError(
            "prepare_index did not run; this is reachable only if the "
            "dry-run branch above stopped returning early"
        )

    started = time.monotonic()
    embeddings = download_hugging_face_embeddings()
    print(f"embedding model ready in {time.monotonic() - started:.0f}s")

    upsert_chunks(embeddings, text_chunks, ids)

    print(f"upserted {unique} unique chunks into {INDEX_NAME!r}")
    confirm_upsert(pc, unique)


def confirm_upsert(pc: Pinecone, expected: int, timeout: float = 60.0) -> None:
    """Report the index's vector count, waiting briefly for it to settle.

    Upserts return before the vectors are queryable, so the count lags. Without
    this the script's last word is what it sent, not what arrived -- and an
    empty index is the failure this project has already shipped once.

    Reports rather than raises: a count still catching up is normal, and
    exiting non-zero on it would make a successful rebuild look broken.
    """
    index = pc.Index(INDEX_NAME)
    deadline = time.monotonic() + timeout
    count = 0

    # Read once before testing the deadline. With the loop the other way round
    # a timeout of zero never read at all, and the message below reported
    # "None of 5861 vectors" -- the one number it exists to deliver, missing.
    while True:
        count = index.describe_index_stats().get("total_vector_count") or 0
        if count >= expected:
            print(f"index {INDEX_NAME!r} now reports {count} vectors")
            # More than was sent means vectors are there under ids this run
            # did not write: upserts left by an earlier run whose chunking
            # differed. Bumping pypdf or langchain-text-splitters is enough to
            # do that, since the ids are hashes of the chunk text -- so a
            # re-index appends instead of overwriting and the old vectors stay.
            # This is the state the live index has been in for weeks, 6x over,
            # and `count >= expected` called it success.
            if count > expected:
                extra = count - expected
                print(
                    f"  note: {extra} more than the {expected} just upserted. "
                    "Those are under ids this run did not write -- an earlier "
                    "index built from different chunking. Retrieval will "
                    "return them alongside the new ones. `--recreate` is the "
                    "only way to clear them."
                )
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(2)

    print(
        f"index {INDEX_NAME!r} reports {count} of {expected} vectors after "
        f"{timeout:.0f}s. Pinecone is eventually consistent, so this is "
        f"probably still settling -- check /healthz or re-run with --dry-run "
        f"to compare."
    )


if __name__ == "__main__":
    main()
