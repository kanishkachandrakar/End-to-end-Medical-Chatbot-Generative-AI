"""Entry point: builds the retrieval chain, then the Flask app around it.

``create()`` does all of it -- reads the keys, loads the embedding model, opens
the Pinecone index, constructs the Groq client -- and gunicorn calls it as an
application factory (``app:create()``). Nothing happens on import, so this
module can be imported, inspected and tested without keys or a network.
"""

import logging
import os

from dotenv import load_dotenv
from flask import Flask
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

from src.config import (
    GROQ_MODEL,
    GROQ_TIMEOUT,
    INDEX_NAME,
    PORT,
    RATE_LIMIT_PER_MINUTE,
    TOP_K,
)
from src.env import require_env
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.ratelimit import per_minute
from src.webapp import create_app


def read_index_size(api_key: str) -> int | None:
    """Vector count for the index, or None if it could not be read."""
    try:
        from pinecone import Pinecone

        index = Pinecone(api_key=api_key).Index(INDEX_NAME)
        return index.describe_index_stats().get("total_vector_count") or 0
    except Exception:
        return None


def build_chain(pinecone_api_key: str, groq_api_key: str):
    """Assemble the retrieval chain: Pinecone retriever + Groq answerer."""
    embeddings = download_hugging_face_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )

    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL,
        # Without this a stalled call occupies a gunicorn thread until the
        # worker timeout kills it at 120s, and there are only four threads.
        timeout=GROQ_TIMEOUT,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))


def log_index_size(logger: logging.Logger, index_size: int | None) -> None:
    """Say what the index holds, because an empty one still answers happily."""
    if index_size is None:
        logger.warning("could not read stats for index %r", INDEX_NAME)
    elif index_size:
        logger.info("index %r holds %s vectors", INDEX_NAME, index_size)
    else:
        logger.warning(
            "index %r is empty -- run 'python store_index.py' first, or every "
            "answer will be produced with no retrieved context",
            INDEX_NAME,
        )


def create() -> Flask:
    """Build the application. Gunicorn calls this; see gunicorn.conf.py."""
    load_dotenv()
    keys = require_env("PINECONE_API_KEY", "GROQ_API_KEY")

    index_size = read_index_size(keys["PINECONE_API_KEY"])
    app = create_app(
        build_chain(keys["PINECONE_API_KEY"], keys["GROQ_API_KEY"]),
        limiter=per_minute(RATE_LIMIT_PER_MINUTE),
        index_size=index_size,
    )
    log_index_size(app.logger, index_size)
    return app


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    create().run(host="0.0.0.0", port=PORT, debug=debug)
