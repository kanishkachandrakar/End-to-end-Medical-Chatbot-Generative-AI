"""Flask front end for the medical chatbot.

Building the RAG chain happens at import time: the embedding model is
loaded, the existing Pinecone index is opened and the Groq client is
created, so the first request does not pay for any of it. That also means
importing this module needs a configured .env and network access.
"""

from flask import Flask, Response, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.config import (
    GROQ_MODEL,
    GROQ_TIMEOUT,
    INDEX_NAME,
    LOG_LEVEL,
    MAX_QUESTION_CHARS,
    PORT,
    TOP_K,
)
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
import os
import re

app = Flask(__name__)

# Outside debug mode Flask leaves app.logger at the root logger's level, which
# is WARNING -- so every logger.info() call below was silently dropped under
# gunicorn. Setting it explicitly is what makes them show up in the logs.
app.logger.setLevel(LOG_LEVEL)

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


embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})


def _report_index_size():
    """Say how many vectors are in the index, so an empty one is obvious."""
    try:
        from pinecone import Pinecone

        stats = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME).describe_index_stats()
        count = stats.get("total_vector_count") or 0
    except Exception:
        app.logger.warning("could not read stats for index %r", INDEX_NAME, exc_info=True)
        return

    if count:
        app.logger.info("index %r holds %s vectors", INDEX_NAME, count)
    else:
        app.logger.warning(
            "index %r is empty -- run 'python store_index.py' first, or every "
            "answer will be produced with no retrieved context",
            INDEX_NAME,
        )


_report_index_size()

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    # Without this a stalled call occupies a gunicorn thread until the worker
    # timeout kills it at 120s, and there are only four threads.
    timeout=GROQ_TIMEOUT,
)

# deepseek-r1 wraps its chain of thought in <think>...</think>; it is useful
# in the logs but should not reach the user.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def _text(body, status=200):
    """Reply with plain text; the front end renders it as text, not markup."""
    return Response(body, status=status, mimetype="text/plain")


def _failure_reply(exc):
    """Map a chain failure to (message, status).

    Rate limiting is the one a visitor can do something about -- waiting --
    so it is worth telling them apart from everything else. The status code is
    read off the exception rather than imported from groq, so this keeps
    working if the provider client changes shape.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)

    if status == 429:
        return "I am being rate limited right now -- please try again in a moment.", 429
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "That took too long to answer -- please try again.", 504
    return "Sorry, I could not answer that right now. Please try again.", 502


@app.route("/")
def index():
    """Serve the chat page."""
    return render_template("chat.html")



@app.route("/healthz")
def healthz():
    """Liveness probe: the chain is built at import, so a reply means it's up."""
    return _text("ok")


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Answer one question and return the reply as plain text."""
    msg = (request.values.get("msg") or "").strip()
    if not msg:
        return _text("Please type a question.", 400)
    if len(msg) > MAX_QUESTION_CHARS:
        return _text(
            f"That question is too long -- please keep it under "
            f"{MAX_QUESTION_CHARS} characters.",
            413,
        )

    # Deliberately not logging the question itself: on a public URL these are
    # strangers' health questions, and Space logs are retained and readable by
    # anyone with access to the Space.
    app.logger.info("question received (%d chars)", len(msg))
    try:
        response = rag_chain.invoke({"input": msg})
    except Exception as exc:
        app.logger.exception("the retrieval chain failed")
        return _text(*_failure_reply(exc))

    answer = response.get("answer", "")
    cleaned_answer = THINK_BLOCK.sub("", answer).strip()

    return _text(cleaned_answer or "I don't have an answer for that.")



if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=PORT, debug=debug)
