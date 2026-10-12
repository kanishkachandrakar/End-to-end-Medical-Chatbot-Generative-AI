"""One shared budget for answering questions.

The limit is deliberately global rather than per visitor. Behind the Spaces
proxy every request arrives from the same address, so keying on remote_addr
would put the whole world in one bucket anyway, and keying on a forwarded header
would be trivially spoofed. A global cap makes no promise it cannot keep: it
protects the Groq quota from being drained by a loop, and it says so.

The cost is that one busy visitor can make others wait. For a free demo that is
the right trade; the limit is generous and configurable.
"""

import threading
import time


class TokenBucket:
    """Allows a burst of ``capacity``, then one request per ``1/rate`` seconds."""

    def __init__(self, capacity: float, rate: float, clock=time.monotonic):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.capacity = float(capacity)
        self.rate = float(rate)
        self._clock = clock
        self._tokens = float(capacity)
        self._updated = clock()
        self._lock = threading.Lock()

    def tokens_available(self) -> float:
        """Tokens that could be spent now, without spending one.

        Reported in a header so a client can pace itself rather than learning
        the limit by being refused.
        """
        with self._lock:
            elapsed = max(0.0, self._clock() - self._updated)
            return min(self.capacity, self._tokens + elapsed * self.rate)

    def seconds_until(self, tokens: float = 1) -> float:
        """How long until ``tokens`` would be available. 0 if they are now.

        Rounded up by the caller into a Retry-After, so a client is told when
        to come back rather than left to guess.
        """
        with self._lock:
            now = self._clock()
            elapsed = max(0.0, now - self._updated)
            available = min(self.capacity, self._tokens + elapsed * self.rate)
            if available >= tokens:
                return 0.0
            return (tokens - available) / self.rate

    def take(self, tokens: float = 1) -> bool:
        """Spend a token if one is available. Thread-safe; never blocks."""
        with self._lock:
            now = self._clock()
            # A clock that went backwards would otherwise credit the bucket.
            elapsed = max(0.0, now - self._updated)
            self._updated = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False


def per_minute(limit: int, clock=time.monotonic) -> TokenBucket | None:
    """Build a bucket allowing ``limit`` requests a minute, or None to disable."""
    if limit <= 0:
        return None
    return TokenBucket(capacity=limit, rate=limit / 60.0, clock=clock)
