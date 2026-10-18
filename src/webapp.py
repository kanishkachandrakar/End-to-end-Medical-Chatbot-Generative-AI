"""The Flask application, independent of how the retrieval chain is built.

Separated from app.py so the routes can be exercised without a Pinecone key, a
Groq key or the embedding model: create_app() takes the chain as an argument and
only needs something with an .invoke() method. app.py builds the real one.

Flask is pointed at the project's templates/ and static/ explicitly, because
this module lives in src/ and would otherwise look for them there.
"""

import math
import os
import pathlib
import time
from typing import Any, Protocol

from flask import Flask, Response, g, make_response, render_template, request

from src import assets, middleware
from src.config import LOG_LEVEL, MAX_CONTENT_BYTES, MAX_QUESTION_CHARS
from src.errors import UNAVAILABLE, failure_reply
from src.messages import NO_ANSWER, NO_QUESTION, TOO_BUSY, TOO_LONG
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

# Crawlers cannot reach /get now that it is POST-only, but indexing the page
# invites traffic that spends the shared rate limit on nobody's question.
ROBOTS_TXT = "User-agent: *\nDisallow: /\n"

# Advertised when the chain could not be built. Long enough not to invite a
# retry storm against a service that needs a human or a restart.
DEGRADED_RETRY_AFTER = 60



def text_reply(body: str, status: int = 200) -> Response:
    """Reply with plain text; the front end renders it as text, not markup."""
    return Response(body, status=status, mimetype="text/plain")


def create_app(
    rag_chain: RagChain,
    limiter: TokenBucket | None = None,
    index_size: int | None = None,
    unavailable: str | None = None,
) -> Flask:
    """Build the Flask app around an object exposing .invoke({"input": ...}).

    ``limiter`` is an optional TokenBucket spending one token per answered
    question. None means no limit, which is what the tests use.

    ``index_size`` is the vector count read once at startup, reported by
    /healthz. None means it could not be determined.

    ``unavailable`` names the reason the chain could not be built, if it could
    not be. The app still serves, so the container does not crash-loop and
    /healthz can say what is wrong, but questions are refused.
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

    assets.install(app)

    # Checked by Flask before the body is read, so a huge upload is refused
    # rather than buffered. MAX_QUESTION_CHARS only applies after parsing, by
    # which point the bytes are already in memory -- and the worker has four
    # threads sharing a free tier's RAM.
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_BYTES

    middleware.install(app, text_reply, APP_REVISION)

    @app.route("/")
    def index() -> Response:
        """Serve the chat page."""
        # The template mirrors the server limit in a maxlength attribute, so
        # the two cannot drift apart.
        page = make_response(
            render_template("chat.html", max_question_chars=MAX_QUESTION_CHARS)
        )
        # The static assets are fingerprinted and cached for a year; the page
        # that references them must not be, or a deploy changing a filename
        # would never reach anyone holding the old HTML.
        page.headers["Cache-Control"] = "no-store"
        return page

    @app.route("/favicon.ico")
    def favicon() -> Response:
        """Browsers ask for this path whatever the page declares.

        Redirecting to the declared icon keeps one copy of the file and stops
        a 404 per visitor in the log, where it reads like a broken route.
        """
        return app.send_static_file("medicalbotpic.jpeg")

    @app.route("/robots.txt")
    def robots() -> Response:
        """Ask crawlers to stay away; this is a demo, not a resource to index."""
        return text_reply(ROBOTS_TXT)

    @app.route("/livez")
    def livez() -> Response:
        """Liveness: is this process serving at all.

        Deliberately knows nothing about the index or the chain. Restarting a
        container cannot repopulate an index or fix a revoked key, so a
        readiness problem must not read as "kill me" to whatever is watching.
        """
        return text_reply("alive")

    @app.route("/healthz")
    def healthz() -> Response:
        """Liveness probe: a reply means the chain was built and the app is up.

        Also names the revision, so a deploy can be confirmed without a log,
        and flags an empty index -- the failure that otherwise looks like
        success, since every request still returns a fluent answer.
        """
        if unavailable:
            return text_reply(f"degraded {APP_REVISION} {unavailable}", 503)
        if index_size == 0:
            return text_reply(f"degraded {APP_REVISION} index-empty", 503)
        vectors = "unknown" if index_size is None else str(index_size)
        return text_reply(f"ok {APP_REVISION} vectors={vectors}")

    @app.route("/get", methods=["POST"])
    def chat() -> Response:
        """Answer one question and return the reply as plain text.

        POST only. Answering costs a Groq call and spends the shared rate
        limit, which is a side effect -- so it does not belong behind a verb
        that crawlers follow and browsers prefetch.
        """
        if unavailable:
            app.logger.error("[%s] refusing: %s", g.request_id, unavailable)
            refusal = text_reply(UNAVAILABLE, 503)
            # Whatever broke at startup is not going to clear on its own in
            # the next second, and a client with no hint retries immediately.
            # A minute is a guess, but an honest one: the alternative is the
            # client guessing, and it guesses zero.
            refusal.headers["Retry-After"] = str(DEGRADED_RETRY_AFTER)
            return refusal

        msg = (request.form.get("msg") or "").strip()
        if not msg:
            return text_reply(NO_QUESTION, 400)
        if len(msg) > MAX_QUESTION_CHARS:
            return text_reply(TOO_LONG.format(limit=MAX_QUESTION_CHARS), 413)

        # Checked after validation so a rejected question costs no budget.
        if limiter is not None and not limiter.take():
            app.logger.warning("rate limit reached, question refused")
            refusal = text_reply(TOO_BUSY, 429)
            # Correct for a 429, and it saves a client guessing: at this rate
            # the next token is a known number of seconds away.
            wait = math.ceil(limiter.seconds_until())
            refusal.headers["Retry-After"] = str(max(1, wait))
            refusal.headers["X-RateLimit-Limit"] = str(int(limiter.capacity))
            refusal.headers["X-RateLimit-Remaining"] = "0"
            return refusal

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
        reply = text_reply(answer or NO_ANSWER)
        if limiter is not None:
            # A client that can see the budget can pace itself instead of
            # discovering the limit by being refused.
            reply.headers["X-RateLimit-Limit"] = str(int(limiter.capacity))
            remaining = int(limiter.tokens_available())
            reply.headers["X-RateLimit-Remaining"] = str(remaining)
        return reply

    return app
