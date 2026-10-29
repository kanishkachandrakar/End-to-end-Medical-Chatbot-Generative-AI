"""Turning a failed request into something worth showing a visitor.

Kept free of heavy imports for the same reason as src.text: app.py cannot be
imported without the embedding model and a Pinecone connection, so logic living
there cannot be tested.
"""

from src.messages import RATE_LIMITED, TIMED_OUT, UNAVAILABLE

__all__ = [
    "RATE_LIMITED",
    "TIMED_OUT",
    "UNAVAILABLE",
    "failure_reply",
    "retry_after",
]

# What to tell a client when the provider rate-limited us and did not say for
# how long. Short, because a provider's throttle usually clears in seconds and
# a client told to wait a minute waits a minute.
DEFAULT_RETRY_AFTER = 5


def _status_of(exc: BaseException) -> int | None:
    """Dig an HTTP status out of an exception, however the client wrapped it."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def retry_after(exc: BaseException) -> int | None:
    """Seconds to wait, read off the provider's own response if it said.

    The app's own 429 carries a Retry-After computed from the token bucket;
    a 429 passed through from Groq carried none, so the same status told a
    client when to come back on one path and left it guessing on the other.

    Groq's is an httpx response underneath, so the header is on
    exc.response.headers. A provider that sends a date rather than a count
    is ignored rather than mis-parsed -- the default below is better than a
    number read wrongly.
    """
    # One try around the whole access. getattr's default only covers an
    # AttributeError, so a response object whose .headers raises anything else
    # came straight back out of here -- and this runs on the failure path,
    # where adding a second failure is the one thing it must not do.
    try:
        raw = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
    except Exception:
        raw = None

    if raw is not None:
        try:
            seconds = int(float(str(raw).strip()))
        except (TypeError, ValueError):
            # The HTTP-date form is legal and is not a count of seconds.
            # Falling back beats reading "Wed," as a number.
            seconds = None
        if seconds is not None and seconds >= 0:
            return max(1, seconds)

    return DEFAULT_RETRY_AFTER if _status_of(exc) == 429 else None


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
