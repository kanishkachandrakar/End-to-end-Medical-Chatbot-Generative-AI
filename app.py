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
from src.env import require_env
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.ratelimit import per_minute
from src.webapp import create_app

load_dotenv()

_keys = require_env("PINECONE_API_KEY", "GROQ_API_KEY")
PINECONE_API_KEY = _keys["PINECONE_API_KEY"]
GROQ_API_KEY = _keys["GROQ_API_KEY"]


def read_index_size() -> int | None:
    """Vector count for the index, or None if it could not be read."""
    try:
        from pinecone import Pinecone

        index = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
        return index.describe_index_stats().get("total_vector_count") or 0
    except Exception:
        return None


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


_index_size = read_index_size()
app = create_app(
    build_chain(),
    limiter=per_minute(RATE_LIMIT_PER_MINUTE),
    index_size=_index_size,
)

if _index_size is None:
    app.logger.warning("could not read stats for index %r", INDEX_NAME)
elif _index_size:
    app.logger.info("index %r holds %s vectors", INDEX_NAME, _index_size)
else:
    app.logger.warning(
        "index %r is empty -- run 'python store_index.py' first, or every "
        "answer will be produced with no retrieved context",
        INDEX_NAME,
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=PORT, debug=debug)
