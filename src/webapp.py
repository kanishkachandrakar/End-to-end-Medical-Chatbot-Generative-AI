"""The Flask application, independent of how the retrieval chain is built.

Separated from app.py so the routes can be exercised without a Pinecone key, a
Groq key or the embedding model: create_app() takes the chain as an argument and
only needs something with an .invoke() method. app.py builds the real one.

Flask is pointed at the project's templates/ and static/ explicitly, because
this module lives in src/ and would otherwise look for them there.
"""

import pathlib

from flask import Flask, Response, render_template, request

from src.config import LOG_LEVEL, MAX_QUESTION_CHARS
from src.errors import failure_reply
from src.text import strip_reasoning

ROOT = pathlib.Path(__file__).resolve().parent.parent

NO_QUESTION = "Please type a question."
NO_ANSWER = "I don't have an answer for that."


def text_reply(body: str, status: int = 200) -> Response:
    """Reply with plain text; the front end renders it as text, not markup."""
    return Response(body, status=status, mimetype="text/plain")


def create_app(rag_chain) -> Flask:
    """Build the Flask app around an object exposing .invoke({"input": ...})."""
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )

    # Outside debug mode Flask leaves app.logger at the root logger's level,
    # which is WARNING -- so every logger.info() call below would be dropped
    # under gunicorn. Setting it explicitly is what makes them appear.
    app.logger.setLevel(LOG_LEVEL)

    @app.route("/")
    def index():
        """Serve the chat page."""
        return render_template("chat.html")

    @app.route("/healthz")
    def healthz():
        """Liveness probe: a reply means the chain was built and the app is up."""
        return text_reply("ok")

    @app.route("/get", methods=["GET", "POST"])
    def chat():
        """Answer one question and return the reply as plain text."""
        msg = (request.values.get("msg") or "").strip()
        if not msg:
            return text_reply(NO_QUESTION, 400)
        if len(msg) > MAX_QUESTION_CHARS:
            return text_reply(
                "That question is too long -- please keep it under "
                f"{MAX_QUESTION_CHARS} characters.",
                413,
            )

        # Deliberately not logging the question itself: on a public URL these
        # are strangers' health questions, and Space logs are retained and
        # readable by anyone with access to the Space.
        app.logger.info("question received (%d chars)", len(msg))
        try:
            response = rag_chain.invoke({"input": msg})
        except Exception as exc:  # noqa: BLE001 - mapped to a reply below
            app.logger.exception("the retrieval chain failed")
            return text_reply(*failure_reply(exc))

        answer = strip_reasoning(response.get("answer", ""))
        return text_reply(answer or NO_ANSWER)

    return app
