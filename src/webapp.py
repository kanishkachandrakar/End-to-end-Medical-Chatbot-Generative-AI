"""The Flask application, independent of how the retrieval chain is built.

Separated from app.py so the routes can be exercised without a Pinecone key, a
Groq key or the embedding model: create_app() takes the chain as an argument and
only needs something with an .invoke() method. app.py builds the real one.

Flask is pointed at the project's templates/ and static/ explicitly, because
this module lives in src/ and would otherwise look for them there.
"""

import pathlib
from typing import Any, Protocol

from flask import Flask, Response, render_template, request

from src.config import LOG_LEVEL, MAX_QUESTION_CHARS
from src.errors import failure_reply
from src.ratelimit import TokenBucket
from src.text import strip_reasoning

ROOT = pathlib.Path(__file__).resolve().parent.parent


class RagChain(Protocol):
    """What create_app() needs from a chain: nothing but invoke().

    Spelled out as a Protocol so the stub the tests pass in is a declared part
    of the interface rather than an accident of duck typing.
    """

    def invoke(self, payload: dict[str, str]) -> dict[str, Any]: ...

# Content-Security-Policy. Every origin here is one the page actually loads
# from; a test cross-checks this against the template and chat.js, because a
# policy that silently blocks the stylesheet is worse than none at all.
# Fonts come from use.fontawesome.com's own domain, and the user avatar from
# i.ibb.co. 'self' covers chat.js and style.css.
CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' https://code.jquery.com https://stackpath.bootstrapcdn.com",
        "style-src 'self' https://stackpath.bootstrapcdn.com https://use.fontawesome.com",
        "font-src 'self' https://use.fontawesome.com",
        "img-src 'self' https://i.ibb.co data:",
        "connect-src 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
    ]
)

SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    # The replies are text/plain and contain passages from a PDF; without this
    # a browser is free to sniff one as HTML and run it.
    "X-Content-Type-Options": "nosniff",
    # frame-ancestors above covers modern browsers; this covers the rest.
    "X-Frame-Options": "DENY",
    # Questions are in the URL on a GET, so do not leak them to the CDNs.
    "Referrer-Policy": "no-referrer",
}

NO_QUESTION = "Please type a question."
NO_ANSWER = "I don't have an answer for that."
TOO_BUSY = (
    "This demo answers a limited number of questions a minute -- "
    "please try again shortly."
)


def text_reply(body: str, status: int = 200) -> Response:
    """Reply with plain text; the front end renders it as text, not markup."""
    return Response(body, status=status, mimetype="text/plain")


def create_app(rag_chain: RagChain, limiter: TokenBucket | None = None) -> Flask:
    """Build the Flask app around an object exposing .invoke({"input": ...}).

    ``limiter`` is an optional TokenBucket spending one token per answered
    question. None means no limit, which is what the tests use.
    """
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )

    # Outside debug mode Flask leaves app.logger at the root logger's level,
    # which is WARNING -- so every logger.info() call below would be dropped
    # under gunicorn. Setting it explicitly is what makes them appear.
    app.logger.setLevel(LOG_LEVEL)

    @app.after_request
    def _add_security_headers(response: Response) -> Response:
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    @app.route("/")
    def index() -> str:
        """Serve the chat page."""
        # The template mirrors the server limit in a maxlength attribute, so
        # the two cannot drift apart.
        return render_template("chat.html", max_question_chars=MAX_QUESTION_CHARS)

    @app.route("/healthz")
    def healthz() -> Response:
        """Liveness probe: a reply means the chain was built and the app is up."""
        return text_reply("ok")

    @app.route("/get", methods=["POST"])
    def chat() -> Response:
        """Answer one question and return the reply as plain text.

        POST only. Answering costs a Groq call and spends the shared rate
        limit, which is a side effect -- so it does not belong behind a verb
        that crawlers follow and browsers prefetch.
        """
        msg = (request.form.get("msg") or "").strip()
        if not msg:
            return text_reply(NO_QUESTION, 400)
        if len(msg) > MAX_QUESTION_CHARS:
            return text_reply(
                "That question is too long -- please keep it under "
                f"{MAX_QUESTION_CHARS} characters.",
                413,
            )

        # Checked after validation so a rejected question costs no budget.
        if limiter is not None and not limiter.take():
            app.logger.warning("rate limit reached, question refused")
            return text_reply(TOO_BUSY, 429)

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
