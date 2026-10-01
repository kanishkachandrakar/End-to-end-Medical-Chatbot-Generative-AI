"""Turning a failed request into something worth showing a visitor.

Kept free of heavy imports for the same reason as src.text: app.py cannot be
imported without the embedding model and a Pinecone connection, so logic living
there cannot be tested.
"""

RATE_LIMITED = "I am being rate limited right now -- please try again in a moment."
TIMED_OUT = "That took too long to answer -- please try again."
UNAVAILABLE = "Sorry, I could not answer that right now. Please try again."


def _status_of(exc: BaseException) -> int | None:
    """Dig an HTTP status out of an exception, however the client wrapped it."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def failure_reply(exc: BaseException) -> tuple[str, int]:
    """Map a chain failure to the (message, HTTP status) the caller should send.

    Rate limiting is the one failure a visitor can act on -- by waiting -- so it
    is worth separating from everything else; on Groq's free tier it is also the
    likeliest. The status is read off the exception rather than matched against
    groq's exception classes, so a reorganised client library does not silently
    turn every rate limit back into a generic error.
    """
    if _status_of(exc) == 429:
        return RATE_LIMITED, 429
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return TIMED_OUT, 504
    return UNAVAILABLE, 502
