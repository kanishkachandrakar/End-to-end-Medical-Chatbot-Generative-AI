"""The budget that stops a loop draining the Groq quota."""

import threading

import pytest

from src.ratelimit import TokenBucket, per_minute


class FakeClock:
    """A clock the test moves by hand, so no test has to wait."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_a_full_burst_is_allowed():
    bucket = TokenBucket(capacity=3, rate=1, clock=FakeClock())
    assert [bucket.take() for _ in range(3)] == [True, True, True]


def test_the_request_after_the_burst_is_refused():
    bucket = TokenBucket(capacity=2, rate=1, clock=FakeClock())
    bucket.take()
    bucket.take()
    assert bucket.take() is False


def test_tokens_come_back_over_time():
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, rate=1, clock=clock)
    bucket.take()
    bucket.take()
    assert bucket.take() is False
    clock.advance(1.0)
    assert bucket.take() is True


def test_refill_is_proportional_not_all_or_nothing():
    clock = FakeClock()
    bucket = TokenBucket(capacity=10, rate=2, clock=clock)
    for _ in range(10):
        bucket.take()
    clock.advance(0.25)  # 0.5 tokens at 2/s
    assert bucket.take() is False
    clock.advance(0.25)  # now 1.0
    assert bucket.take() is True


def test_idling_does_not_accumulate_beyond_the_burst_size():
    """An hour of quiet must not buy an hour's worth of requests at once."""
    clock = FakeClock()
    bucket = TokenBucket(capacity=3, rate=1, clock=clock)
    clock.advance(3600)
    assert [bucket.take() for _ in range(4)] == [True, True, True, False]


def test_a_clock_that_goes_backwards_does_not_credit_the_bucket():
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, rate=1, clock=clock)
    assert bucket.take() is True
    clock.advance(-50)
    assert bucket.take() is False


def test_concurrent_takers_cannot_overspend():
    """Four gunicorn threads share one bucket, so take() must be atomic."""
    bucket = TokenBucket(capacity=50, rate=0.0001, clock=FakeClock())
    granted = []
    lock = threading.Lock()

    def worker():
        for _ in range(50):
            if bucket.take():
                with lock:
                    granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == 50, f"handed out {len(granted)} of 50 tokens"


@pytest.mark.parametrize("bad", [0, -1])
def test_a_nonsensical_capacity_is_rejected(bad):
    with pytest.raises(ValueError):
        TokenBucket(capacity=bad, rate=1)


def test_a_nonsensical_rate_is_rejected():
    with pytest.raises(ValueError):
        TokenBucket(capacity=1, rate=0)


def test_per_minute_allows_the_stated_number_in_a_minute():
    clock = FakeClock()
    bucket = per_minute(30, clock=clock)
    assert sum(bucket.take() for _ in range(30)) == 30
    assert bucket.take() is False
    clock.advance(60)
    assert sum(bucket.take() for _ in range(30)) == 30


@pytest.mark.parametrize("disabled", [0, -5])
def test_per_minute_returns_none_when_disabled(disabled):
    assert per_minute(disabled) is None


class TestSecondsUntil:
    """What a Retry-After header is computed from."""

    def test_zero_when_a_token_is_available(self):
        assert TokenBucket(capacity=2, rate=1, clock=FakeClock()).seconds_until() == 0.0

    def test_the_wait_after_the_burst_is_one_over_the_rate(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=1, rate=2, clock=clock)
        bucket.take()
        assert bucket.seconds_until() == pytest.approx(0.5)

    def test_the_wait_shrinks_as_time_passes(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=1, rate=1, clock=clock)
        bucket.take()
        before = bucket.seconds_until()
        clock.advance(0.5)
        assert bucket.seconds_until() < before

    def test_asking_does_not_spend_a_token(self):
        """A client told to wait must not be charged for being told."""
        bucket = TokenBucket(capacity=1, rate=0.001, clock=FakeClock())
        bucket.seconds_until()
        bucket.seconds_until()
        assert bucket.take() is True

    def test_it_accounts_for_more_than_one_token(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=1, rate=1, clock=clock)
        bucket.take()
        assert bucket.seconds_until(3) == pytest.approx(3.0)


class TestTokensAvailable:
    """What the X-RateLimit-Remaining header reports."""

    def test_a_fresh_bucket_is_full(self):
        bucket = TokenBucket(capacity=5, rate=1, clock=FakeClock())
        assert bucket.tokens_available() == 5

    def test_it_drops_as_tokens_are_spent(self):
        bucket = TokenBucket(capacity=3, rate=0.001, clock=FakeClock())
        bucket.take()
        assert bucket.tokens_available() == pytest.approx(2)

    def test_asking_does_not_spend_one(self):
        bucket = TokenBucket(capacity=1, rate=0.001, clock=FakeClock())
        bucket.tokens_available()
        bucket.tokens_available()
        assert bucket.take() is True

    def test_it_accounts_for_refill(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=2, rate=1, clock=clock)
        bucket.take()
        bucket.take()
        clock.advance(1)
        assert bucket.tokens_available() == pytest.approx(1)

    def test_it_never_exceeds_the_capacity(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=2, rate=1, clock=clock)
        clock.advance(3600)
        assert bucket.tokens_available() == 2
