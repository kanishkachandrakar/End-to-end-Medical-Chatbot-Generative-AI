"""The Flask application, independent of how the retrieval chain is built.

Separated from app.py so the routes can be exercised without a Pinecone key, a
Groq key or the embedding model: create_app() takes the chain as an argument and
only needs something with an .invoke() method. app.py builds the real one.

Flask is pointed at the project's templates/ and static/ explicitly, because
this module lives in src/ and would otherwise look for them there.
"""

import os
import pathlib
import time
import uuid
from typing import Any, Protocol

from flask import Flask, Response, g, render_template, request

from src.config import LOG_LEVEL, MAX_CONTENT_BYTES, MAX_QUESTION_CHARS
from src.errors import failure_reply
from src.ratelimit import TokenBucket
from src.text import strip_reasoning

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Set at image build time. Without it there is no way to tell which revision a
# running Space is serving, which matters most when a deploy appears not to
# have taken effect.
APP_REVISION = os.environ.get("APP_REVISION", "unknown")


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

# Crawlers cannot reach /get now that it is POST-only, but indexing the page
# invites traffic that spends the shared rate limit on nobody's question.
ROBOTS_TXT = "User-agent: *\nDisallow: /\n"

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

    # Checked by Flask before the body is read, so a huge upload is refused
    # rather than buffered. MAX_QUESTION_CHARS only applies after parsing, by
    # which point the bytes are already in memory -- and the worker has four
    # threads sharing a free tier's RAM.
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_BYTES

    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(500)
    def _plain_error(error):
        """Answer errors in text too.

        Flask's defaults are HTML pages. The front end appends whatever comes
        back to the chat log, so an HTML error document would arrive as a wall
        of markup in a bubble -- and a client told to expect text/plain from
        every other response has no reason to handle HTML from this one.
        """
        status = getattr(error, "code", 500)
        name = getattr(error, "name", "Error")
        return text_reply(f"{name}.", status)

    @app.before_request
    def _assign_request_id() -> None:
        """Tag each request so its log lines can be found from a reply."""
        g.request_id = uuid.uuid4().hex[:8]

    @app.after_request
    def _add_security_headers(response: Response) -> Response:
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        request_id = g.get("request_id")
        if request_id:
            response.headers.setdefault("X-Request-Id", request_id)
        return response

    @app.route("/")
    def index() -> str:
        """Serve the chat page."""
        # The template mirrors the server limit in a maxlength attribute, so
        # the two cannot drift apart.
        return render_template("chat.html", max_question_chars=MAX_QUESTION_CHARS)

    @app.route("/robots.txt")
    def robots() -> Response:
        """Ask crawlers to stay away; this is a demo, not a resource to index."""
        return text_reply(ROBOTS_TXT)

    @app.route("/healthz")
    def healthz() -> Response:
        """Liveness probe: a reply means the chain was built and the app is up.

        Also names the revision, so a deploy can be confirmed without a log.
        """
        return text_reply(f"ok {APP_REVISION}")

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
        app.logger.info("[%s] question received (%d chars)", g.request_id, len(msg))
        started = time.perf_counter()
        try:
            response = rag_chain.invoke({"input": msg})
        except Exception as exc:  # noqa: BLE001 - mapped to a reply below
            app.logger.exception(
                "[%s] the retrieval chain failed after %.1fs",
                g.request_id,
                time.perf_counter() - started,
            )
            message, status = failure_reply(exc)
            # The id lets a reported failure be matched to its traceback in the
            # log, which is otherwise guesswork on a shared deployment.
            return text_reply(f"{message} (ref {g.request_id})", status)

        answer = strip_reasoning(response.get("answer", ""))
        # Retrieval plus generation, which is the only number that explains a
        # slow demo -- and the reasoning tokens that get stripped are part of it.
        app.logger.info(
            "[%s] answered in %.1fs (%d chars)",
            g.request_id,
            time.perf_counter() - started,
            len(answer),
        )
        return text_reply(answer or NO_ANSWER)

    return app
