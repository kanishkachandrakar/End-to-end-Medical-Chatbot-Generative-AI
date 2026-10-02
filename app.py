"""Entry point: builds the retrieval chain, then the Flask app around it.

Everything happens at import time -- the embedding model is loaded, the Pinecone
index is opened and the Groq client is created -- so the first request does not
pay for any of it, and so gunicorn can serve ``app:app``. It also means
importing this module needs a configured .env and network access; the routes
themselves live in src/webapp.py, which needs neither.
"""

import os

from dotenv import load_dotenv
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
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.ratelimit import per_minute
from src.webapp import create_app

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

_missing = [
    name
    for name, value in (
        ("PINECONE_API_KEY", PINECONE_API_KEY),
        ("GROQ_API_KEY", GROQ_API_KEY),
    )
    if not value
]
if _missing:
    raise RuntimeError(
        "Missing required environment variable(s): "
        + ", ".join(_missing)
        + ". Copy .env.example to .env and fill in your keys."
    )


def report_index_size(logger) -> None:
    """Say how many vectors are in the index, so an empty one is obvious."""
    try:
        from pinecone import Pinecone

        index = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
        count = index.describe_index_stats().get("total_vector_count") or 0
    except Exception:
        logger.warning("could not read stats for index %r", INDEX_NAME, exc_info=True)
        return

    if count:
        logger.info("index %r holds %s vectors", INDEX_NAME, count)
    else:
        logger.warning(
            "index %r is empty -- run 'python store_index.py' first, or every "
            "answer will be produced with no retrieved context",
            INDEX_NAME,
        )


def build_chain():
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
        groq_api_key=GROQ_API_KEY,
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


app = create_app(build_chain(), limiter=per_minute(RATE_LIMIT_PER_MINUTE))
report_index_size(app.logger)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=PORT, debug=debug)
