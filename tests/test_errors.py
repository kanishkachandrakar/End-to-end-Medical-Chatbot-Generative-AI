"""What a visitor is told when a request fails."""

import pytest

from src.errors import (
    DEFAULT_RETRY_AFTER,
    RATE_LIMITED,
    TIMED_OUT,
    UNAVAILABLE,
    failure_reply,
    retry_after,
)


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


class TestRetryAfter:
    """Our own 429 said when to come back; a passed-through one did not.

    Same status, and whether a client was told depended on which side of the
    app produced it -- the token bucket, or the provider the bucket exists to
    protect.
    """

    def _raising(self, **attrs):
        exc = RuntimeError("upstream said no")
        for name, value in attrs.items():
            setattr(exc, name, value)
        return exc

    class _Response:
        def __init__(self, headers):
            self.headers = headers
            self.status_code = 429

    def test_the_providers_own_number_is_used(self):
        exc = self._raising(response=self._Response({"retry-after": "17"}))
        assert retry_after(exc) == 17

    def test_a_fractional_value_is_accepted(self):
        """httpx hands back whatever the header said; some send 1.5."""
        exc = self._raising(response=self._Response({"retry-after": "1.5"}))
        assert retry_after(exc) == 1

    def test_a_429_with_no_header_gets_the_default(self):
        assert retry_after(self._raising(status_code=429)) == DEFAULT_RETRY_AFTER

    def test_a_date_valued_header_falls_back_rather_than_mis_parsing(self):
        """The HTTP-date form is legal and is not a count of seconds.

        Reading "Wed," as a number would be worse than not reading it.
        """
        exc = self._raising(
            response=self._Response({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        )
        assert retry_after(exc) == DEFAULT_RETRY_AFTER

    def test_zero_is_raised_to_one(self):
        """A client told to wait no time at all retries immediately."""
        exc = self._raising(response=self._Response({"retry-after": "0"}))
        assert retry_after(exc) == 1

    def test_a_failure_that_is_not_a_rate_limit_gets_nothing(self):
        """Retry-After on a 502 would be a guess dressed as information."""
        assert retry_after(self._raising()) is None
        assert retry_after(TimeoutError("slow")) is None

    def test_an_unreadable_headers_object_does_not_raise(self):
        """It is on the failure path; it must not add a second failure."""

        class Hostile:
            @property
            def headers(self):
                raise RuntimeError("no")

        exc = self._raising(status_code=429)
        exc.response = Hostile()
        assert retry_after(exc) == DEFAULT_RETRY_AFTER
