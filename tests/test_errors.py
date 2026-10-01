"""What a visitor is told when a request fails."""

import pytest

from src.errors import RATE_LIMITED, TIMED_OUT, UNAVAILABLE, failure_reply


class GroqRateLimit(Exception):
    """Shaped like groq.RateLimitError, which carries status_code directly."""

    status_code = 429


class _Response:
    status_code = 429


class WrappedRateLimit(Exception):
    """Shaped like an httpx-style error that keeps the status on .response."""

    response = _Response()


class APITimeoutError(Exception):
    """Named like the provider's timeout class but unrelated to TimeoutError."""


def test_rate_limit_on_the_exception():
    assert failure_reply(GroqRateLimit()) == (RATE_LIMITED, 429)


def test_rate_limit_nested_on_a_response():
    assert failure_reply(WrappedRateLimit()) == (RATE_LIMITED, 429)


def test_builtin_timeout():
    assert failure_reply(TimeoutError()) == (TIMED_OUT, 504)


def test_timeout_recognised_by_class_name():
    """The provider's timeout does not subclass TimeoutError, so match the name."""
    assert failure_reply(APITimeoutError()) == (TIMED_OUT, 504)


@pytest.mark.parametrize("exc", [ValueError("boom"), KeyError("answer"), Exception()])
def test_anything_else_is_a_generic_502(exc):
    assert failure_reply(exc) == (UNAVAILABLE, 502)


def test_a_non_429_status_is_not_special_cased():
    class ServerError(Exception):
        status_code = 500

    assert failure_reply(ServerError()) == (UNAVAILABLE, 502)


def test_a_non_integer_status_is_ignored():
    """A Mock or a string here must not crash the error path itself."""

    class Odd(Exception):
        status_code = "429"

    assert failure_reply(Odd()) == (UNAVAILABLE, 502)


def test_every_message_is_plain_text_for_the_chat_bubble():
    """The front end renders these with .text(); no markup should appear."""
    for message in (RATE_LIMITED, TIMED_OUT, UNAVAILABLE):
        assert "<" not in message and ">" not in message
        assert message.endswith(".")
