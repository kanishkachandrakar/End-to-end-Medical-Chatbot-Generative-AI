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
from src.config import GROQ_MODEL, INDEX_NAME, MAX_QUESTION_CHARS, PORT, TOP_K
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
import os
import re

app = Flask(__name__)

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

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL
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

    app.logger.info("question: %s", msg)
    try:
        response = rag_chain.invoke({"input": msg})
    except Exception:
        app.logger.exception("the retrieval chain failed")
        return _text("Sorry, I could not answer that right now. Please try again.", 502)

    answer = response.get("answer", "")
    cleaned_answer = THINK_BLOCK.sub("", answer).strip()

    return _text(cleaned_answer or "I don't have an answer for that.")



if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=PORT, debug=debug)
