"""Thread-safe rate limiter with jitter and adaptive backoff.

Designed to minimize the risk of IP blocking when downloading
from yfinance / Yahoo Finance at scale.
"""

from __future__ import annotations

import logging
import random
import threading
import time

logger = logging.getLogger("market_data.downloader.rate_limiter")


class RateLimiter:
    """Adaptive rate limiter with random jitter (thread-safe).

    Features:
        - Fixed minimum interval between requests
        - Random jitter (±30%) to avoid bot-like patterns
        - Adaptive slowdown: automatically reduces rate on throttle signals
        - Cooldown recovery: gradually restores original rate after a period
          without throttling

    Usage:
        limiter = RateLimiter(max_per_second=2.0)
        limiter.acquire()          # blocks until a slot is available
        limiter.report_throttle()  # call when you detect a 429 / block
    """

    # How much to multiply the interval on each throttle event
    _BACKOFF_FACTOR = 2.0
    # Maximum slowdown factor (won't go slower than original * this)
    _MAX_SLOWDOWN = 16.0
    # Jitter range: ±30% of the base interval
    _JITTER_FRACTION = 0.3
    # After this many seconds without a throttle, start recovering
    _RECOVERY_AFTER_SECS = 120.0
    # Recovery reduces the slowdown multiplier by this factor each time
    _RECOVERY_FACTOR = 0.5

    def __init__(self, max_per_second: float = 2.0) -> None:
        if max_per_second <= 0:
            msg = "max_per_second must be positive"
            raise ValueError(msg)

        self._base_interval = 1.0 / max_per_second
        self._slowdown_multiplier = 1.0
        self._last_call_time = 0.0
        self._last_throttle_time = 0.0
        self._throttle_count = 0
        self._lock = threading.Lock()

    @property
    def current_rate(self) -> float:
        """Current effective requests per second."""
        return 1.0 / (self._base_interval * self._slowdown_multiplier)

    @property
    def effective_interval(self) -> float:
        """Current effective interval between requests (seconds)."""
        return self._base_interval * self._slowdown_multiplier

    def acquire(self) -> None:
        """Block until a rate-limit slot is available, then consume it.

        Applies the current interval (with jitter) between consecutive calls.
        If enough time has passed since the last throttle event, gradually
        recovers toward the original rate.
        """
        with self._lock:
            self._maybe_recover()

            now = time.monotonic()
            interval = self._jittered_interval()
            elapsed = now - self._last_call_time
            wait_time = interval - elapsed

            if wait_time > 0:
                time.sleep(wait_time)

            self._last_call_time = time.monotonic()

    def report_throttle(self) -> None:
        """Signal that a throttle / rate-limit was detected.

        Call this when you receive HTTP 429, connection resets, or other
        indicators that the server is pushing back.  The limiter will
        exponentially increase the interval between requests.
        """
        with self._lock:
            self._throttle_count += 1
            self._last_throttle_time = time.monotonic()

            old = self._slowdown_multiplier
            self._slowdown_multiplier = min(
                self._slowdown_multiplier * self._BACKOFF_FACTOR,
                self._MAX_SLOWDOWN,
            )
            new_rate = 1.0 / (self._base_interval * self._slowdown_multiplier)
            logger.warning(
                "Throttle detected (#%d) — slowing down: %.2f → %.2f req/s "
                "(multiplier %.1fx → %.1fx)",
                self._throttle_count,
                1.0 / (self._base_interval * old),
                new_rate,
                old,
                self._slowdown_multiplier,
            )

    def _maybe_recover(self) -> None:
        """Gradually recover toward the original rate if no recent throttling."""
        if self._slowdown_multiplier <= 1.0:
            return
        if self._last_throttle_time == 0.0:
            return

        time_since_throttle = time.monotonic() - self._last_throttle_time
        if time_since_throttle > self._RECOVERY_AFTER_SECS:
            old = self._slowdown_multiplier
            self._slowdown_multiplier = max(
                1.0,
                self._slowdown_multiplier * self._RECOVERY_FACTOR,
            )
            if self._slowdown_multiplier < old:
                logger.info(
                    "Rate limiter recovering: multiplier %.1fx → %.1fx (%.1f req/s)",
                    old,
                    self._slowdown_multiplier,
                    1.0 / (self._base_interval * self._slowdown_multiplier),
                )
            # Reset the throttle timer so we don't recover too fast
            self._last_throttle_time = time.monotonic()

    def _jittered_interval(self) -> float:
        """Return the current interval with random jitter applied."""
        base = self._base_interval * self._slowdown_multiplier
        jitter = base * self._JITTER_FRACTION
        return base + random.uniform(-jitter, jitter)  # noqa: S311

    @property
    def throttle_count(self) -> int:
        """Total number of throttle events recorded."""
        return self._throttle_count

    def __enter__(self) -> RateLimiter:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        pass
