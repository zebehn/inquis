"""Rate limiter for VLM API calls.

This module implements a token bucket rate limiter to prevent
exceeding API rate limits and to provide smooth request pacing.
"""

import time
import threading
from typing import Optional
from pydantic import BaseModel, Field


class RateLimiterConfig(BaseModel):
    """Configuration for rate limiter.

    Attributes:
        requests_per_second: Maximum requests per second (refill rate)
        burst_capacity: Maximum burst size (bucket capacity)
        enabled: Whether rate limiting is enabled
    """

    requests_per_second: float = Field(default=10.0, gt=0.0, le=100.0)
    burst_capacity: int = Field(default=20, ge=1, le=200)
    enabled: bool = Field(default=True)

    class Config:
        """Pydantic config."""
        frozen = True


class RateLimiter:
    """Token bucket rate limiter for API calls.

    This class implements a thread-safe token bucket algorithm that
    limits the rate of API requests while allowing for bursts.

    Example:
        >>> config = RateLimiterConfig(requests_per_second=5.0, burst_capacity=10)
        >>> limiter = RateLimiter(config)
        >>> limiter.acquire()  # Blocks until token is available
        >>> # Make API call
    """

    def __init__(self, config: RateLimiterConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limiter configuration
        """
        self.config = config
        self._tokens = float(config.burst_capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire a token for making an API call.

        Blocks until a token is available or timeout expires.

        Args:
            timeout: Maximum time to wait in seconds (None for infinite)

        Returns:
            True if token acquired, False if timeout expired

        Raises:
            TimeoutError: If timeout expires before token is available
        """
        if not self.config.enabled:
            return True

        start_time = time.monotonic()

        while True:
            with self._lock:
                self._refill_tokens()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

                # Calculate wait time for next token
                wait_time = self._calculate_wait_time()

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise TimeoutError("Rate limiter timeout expired")

                # Sleep for minimum of wait_time or remaining timeout
                sleep_time = min(wait_time, timeout - elapsed)
            else:
                sleep_time = wait_time

            time.sleep(sleep_time)

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False if no tokens available
        """
        if not self.config.enabled:
            return True

        with self._lock:
            self._refill_tokens()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            return False

    def _refill_tokens(self):
        """Refill tokens based on elapsed time.

        This method must be called while holding the lock.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill

        # Add tokens based on elapsed time and refill rate
        tokens_to_add = elapsed * self.config.requests_per_second
        self._tokens = min(
            self._tokens + tokens_to_add,
            float(self.config.burst_capacity)
        )

        self._last_refill = now

    def _calculate_wait_time(self) -> float:
        """Calculate time to wait for next token.

        This method must be called while holding the lock.

        Returns:
            Wait time in seconds
        """
        tokens_needed = 1.0 - self._tokens
        return tokens_needed / self.config.requests_per_second

    def get_available_tokens(self) -> float:
        """Get current number of available tokens.

        Returns:
            Number of available tokens
        """
        with self._lock:
            self._refill_tokens()
            return self._tokens

    def reset(self):
        """Reset rate limiter to full capacity."""
        with self._lock:
            self._tokens = float(self.config.burst_capacity)
            self._last_refill = time.monotonic()
