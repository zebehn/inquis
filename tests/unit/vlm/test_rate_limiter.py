"""Unit tests for VLM rate limiter.

Tests token bucket algorithm, rate limiting enforcement, and burst handling.
"""

import pytest
import time
from pydantic import ValidationError

from src.vlm.rate_limiter import RateLimiter, RateLimiterConfig


class TestRateLimiterConfig:
    """Tests for RateLimiterConfig model."""

    def test_valid_config_with_defaults(self):
        """Test creating config with default values."""
        config = RateLimiterConfig()

        assert config.requests_per_second == 10.0
        assert config.burst_capacity == 20
        assert config.enabled is True

    def test_valid_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = RateLimiterConfig(
            requests_per_second=5.0,
            burst_capacity=10,
            enabled=False
        )

        assert config.requests_per_second == 5.0
        assert config.burst_capacity == 10
        assert config.enabled is False

    def test_config_validates_requests_per_second_range(self):
        """Test that requests_per_second must be positive and <= 100."""
        # Valid range
        config_min = RateLimiterConfig(requests_per_second=0.1)
        config_max = RateLimiterConfig(requests_per_second=100.0)
        assert config_min.requests_per_second == 0.1
        assert config_max.requests_per_second == 100.0

        # Invalid: too low
        with pytest.raises(ValidationError):
            RateLimiterConfig(requests_per_second=0.0)

        # Invalid: too high
        with pytest.raises(ValidationError):
            RateLimiterConfig(requests_per_second=150.0)

    def test_config_validates_burst_capacity_range(self):
        """Test that burst_capacity must be >= 1 and <= 200."""
        # Valid range
        config_min = RateLimiterConfig(burst_capacity=1)
        config_max = RateLimiterConfig(burst_capacity=200)
        assert config_min.burst_capacity == 1
        assert config_max.burst_capacity == 200

        # Invalid: too low
        with pytest.raises(ValidationError):
            RateLimiterConfig(burst_capacity=0)

        # Invalid: too high
        with pytest.raises(ValidationError):
            RateLimiterConfig(burst_capacity=300)

    def test_config_is_immutable(self):
        """Test that config is frozen after creation."""
        config = RateLimiterConfig()

        with pytest.raises(ValidationError):
            config.requests_per_second = 20.0


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_limiter_initialization(self):
        """Test initializing rate limiter."""
        config = RateLimiterConfig(requests_per_second=5.0, burst_capacity=10)
        limiter = RateLimiter(config)

        assert limiter.config == config
        assert limiter.get_available_tokens() == 10.0

    def test_acquire_single_token(self):
        """Test acquiring a single token."""
        config = RateLimiterConfig(requests_per_second=10.0, burst_capacity=10)
        limiter = RateLimiter(config)

        # Should succeed immediately
        result = limiter.acquire()
        assert result is True
        # Allow small variance due to time precision
        assert 8.9 <= limiter.get_available_tokens() <= 9.1

    def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens in succession."""
        config = RateLimiterConfig(requests_per_second=100.0, burst_capacity=5)
        limiter = RateLimiter(config)

        # Acquire 5 tokens (full capacity)
        for i in range(5):
            assert limiter.acquire() is True

        # Should have 0 tokens left
        assert limiter.get_available_tokens() < 1.0

    def test_try_acquire_without_blocking(self):
        """Test try_acquire returns False when no tokens available."""
        config = RateLimiterConfig(requests_per_second=10.0, burst_capacity=2)
        limiter = RateLimiter(config)

        # Acquire all tokens
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True

        # Should fail without blocking
        assert limiter.try_acquire() is False

    def test_token_refill_over_time(self):
        """Test that tokens are refilled over time."""
        config = RateLimiterConfig(requests_per_second=10.0, burst_capacity=10)
        limiter = RateLimiter(config)

        # Acquire all tokens
        for _ in range(10):
            limiter.try_acquire()

        assert limiter.get_available_tokens() < 1.0

        # Wait for tokens to refill (0.5 seconds = 5 tokens at 10/sec)
        time.sleep(0.5)

        available = limiter.get_available_tokens()
        assert 4.0 <= available <= 6.0  # Allow some timing variance

    def test_acquire_blocks_until_token_available(self):
        """Test that acquire blocks until token is available."""
        config = RateLimiterConfig(requests_per_second=10.0, burst_capacity=1)
        limiter = RateLimiter(config)

        # Acquire the only token
        limiter.acquire()

        # This should block briefly until token refills
        start = time.time()
        limiter.acquire()
        elapsed = time.time() - start

        # Should have waited approximately 0.1 seconds (1/10 seconds)
        assert 0.05 <= elapsed <= 0.2

    def test_acquire_with_timeout_success(self):
        """Test acquire with timeout that succeeds."""
        config = RateLimiterConfig(requests_per_second=10.0, burst_capacity=2)
        limiter = RateLimiter(config)

        # Acquire first token immediately
        result = limiter.acquire(timeout=1.0)
        assert result is True

    def test_acquire_with_timeout_expires(self):
        """Test acquire with timeout that expires."""
        config = RateLimiterConfig(requests_per_second=1.0, burst_capacity=1)
        limiter = RateLimiter(config)

        # Acquire the only token
        limiter.acquire()

        # Try to acquire with very short timeout (token refills after 1 second)
        with pytest.raises(TimeoutError) as exc_info:
            limiter.acquire(timeout=0.1)

        assert "timeout expired" in str(exc_info.value).lower()

    def test_burst_handling(self):
        """Test that burst capacity allows multiple quick requests."""
        config = RateLimiterConfig(requests_per_second=5.0, burst_capacity=10)
        limiter = RateLimiter(config)

        start = time.time()

        # Should be able to make 10 requests immediately (burst)
        for _ in range(10):
            limiter.acquire()

        elapsed = time.time() - start

        # Should complete quickly (< 0.5 seconds) due to burst capacity
        assert elapsed < 0.5

    def test_sustained_rate_limiting(self):
        """Test that rate limiting enforces sustained rate."""
        config = RateLimiterConfig(requests_per_second=10.0, burst_capacity=2)
        limiter = RateLimiter(config)

        start = time.time()

        # Make 12 requests (2 burst + 10 more)
        for _ in range(12):
            limiter.acquire()

        elapsed = time.time() - start

        # Should take approximately 1 second (10 requests at 10/sec after burst)
        assert 0.8 <= elapsed <= 1.3

    def test_disabled_rate_limiter(self):
        """Test that disabled rate limiter doesn't block."""
        config = RateLimiterConfig(enabled=False)
        limiter = RateLimiter(config)

        start = time.time()

        # Should be able to make many requests immediately
        for _ in range(100):
            assert limiter.acquire() is True
            assert limiter.try_acquire() is True

        elapsed = time.time() - start

        # Should complete very quickly (< 0.1 seconds)
        assert elapsed < 0.1

    def test_reset_rate_limiter(self):
        """Test resetting rate limiter to full capacity."""
        config = RateLimiterConfig(requests_per_second=10.0, burst_capacity=10)
        limiter = RateLimiter(config)

        # Acquire some tokens
        for _ in range(7):
            limiter.acquire()

        assert limiter.get_available_tokens() < 5.0

        # Reset
        limiter.reset()

        # Should be back to full capacity
        assert limiter.get_available_tokens() == 10.0

    def test_tokens_do_not_exceed_capacity(self):
        """Test that available tokens never exceed burst capacity."""
        config = RateLimiterConfig(requests_per_second=10.0, burst_capacity=5)
        limiter = RateLimiter(config)

        # Wait for tokens to accumulate
        time.sleep(1.0)  # Should refill 10 tokens, but capped at 5

        available = limiter.get_available_tokens()
        assert available <= 5.0

    def test_concurrent_acquire(self):
        """Test that rate limiter is thread-safe."""
        import threading

        config = RateLimiterConfig(requests_per_second=100.0, burst_capacity=50)
        limiter = RateLimiter(config)

        acquired_count = [0]
        lock = threading.Lock()

        def acquire_token():
            if limiter.try_acquire():
                with lock:
                    acquired_count[0] += 1

        # Try to acquire 100 tokens from multiple threads very quickly
        threads = []
        for _ in range(100):
            t = threading.Thread(target=acquire_token)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have acquired approximately 50 tokens (burst capacity)
        # Allow small variance for token refill during execution
        assert 48 <= acquired_count[0] <= 52
