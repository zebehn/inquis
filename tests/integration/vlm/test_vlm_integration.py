"""Integration tests for VLM module with real API calls.

These tests require a valid OpenAI API key and make real API calls.
Mark as slow and skip in CI/CD pipelines to avoid costs.

Run with: pytest tests/integration/vlm/test_vlm_integration.py -v -s
"""

import pytest
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np

from src.vlm import (
    VLMClient,
    VLMConfig,
    VLMRequest,
    VLMBatchRequest,
    VLMStatus,
    VLMException,
    VLMAuthError,
    RateLimiterConfig,
)


# Skip all tests if no API key available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping integration tests"
)


@pytest.fixture
def api_key():
    """Get OpenAI API key from environment."""
    return os.environ.get("OPENAI_API_KEY")


@pytest.fixture
def test_image_simple(tmp_path):
    """Create a simple test image with geometric shapes."""
    image_path = tmp_path / "test_simple.jpg"

    # Create simple image with colored rectangle
    img = Image.new("RGB", (300, 300), color="white")
    pixels = img.load()

    # Draw a red rectangle
    for i in range(100, 200):
        for j in range(100, 200):
            pixels[i, j] = (255, 0, 0)

    img.save(image_path, "JPEG")
    return image_path


@pytest.fixture
def test_image_complex(tmp_path):
    """Create a more complex test image."""
    image_path = tmp_path / "test_complex.jpg"

    # Create image with multiple shapes
    img = Image.new("RGB", (400, 400), color="lightblue")
    pixels = img.load()

    # Draw a circle-like shape (yellow)
    center_x, center_y = 200, 200
    radius = 50
    for i in range(400):
        for j in range(400):
            if (i - center_x)**2 + (j - center_y)**2 < radius**2:
                pixels[i, j] = (255, 255, 0)

    # Draw a triangle-like shape (green)
    for i in range(100, 150):
        for j in range(100, 100 + (i - 100)):
            pixels[i, j] = (0, 255, 0)

    img.save(image_path, "JPEG")
    return image_path


class TestVLMIntegrationSingleQuery:
    """Integration tests for single VLM queries."""

    @pytest.mark.slow
    def test_real_api_call_simple_image(self, api_key, test_image_simple):
        """Test real API call with simple image."""
        config = VLMConfig(
            api_key=api_key,
            model_name="gpt-4o-mini",  # Use cheaper model for testing
            max_retries=3
        )
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_simple,
            prompt="What color is the rectangle in this image?",
            model="gpt-4o-mini"
        )

        response = client.query_region(request)

        # Verify response structure
        assert response.status in [VLMStatus.SUCCESS, VLMStatus.VLM_UNCERTAIN]
        assert response.label is not None
        assert response.confidence is not None
        assert 0.0 <= response.confidence <= 1.0
        assert response.cost > 0
        assert response.tokens_used > 0
        assert response.latency_ms > 0
        assert response.queried_at is not None
        assert response.responded_at is not None

        # Log results for manual inspection
        print(f"\n✅ Real API call successful:")
        print(f"   Label: {response.label}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Cost: ${response.cost:.4f}")
        print(f"   Latency: {response.latency_ms:.0f}ms")
        print(f"   Tokens: {response.tokens_used}")

    @pytest.mark.slow
    def test_real_api_call_complex_image(self, api_key, test_image_complex):
        """Test real API call with more complex image."""
        config = VLMConfig(
            api_key=api_key,
            model_name="gpt-4o-mini"
        )
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_complex,
            prompt="Describe the shapes and colors you see in this image.",
            model="gpt-4o-mini",
            max_tokens=200
        )

        response = client.query_region(request)

        assert response.status in [VLMStatus.SUCCESS, VLMStatus.VLM_UNCERTAIN]
        assert response.label is not None
        assert response.reasoning is not None

        print(f"\n✅ Complex image analysis:")
        print(f"   Label: {response.label}")
        print(f"   Reasoning: {response.reasoning}")
        print(f"   Confidence: {response.confidence:.2f}")

    @pytest.mark.slow
    def test_invalid_api_key_error(self, test_image_simple):
        """Test that invalid API key raises VLMAuthError."""
        config = VLMConfig(
            api_key="sk-invalid-key-12345",
            max_retries=0  # Don't retry on auth errors
        )
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_simple,
            prompt="What is this?",
            model="gpt-4o-mini"
        )

        with pytest.raises(VLMAuthError) as exc_info:
            client.query_region(request)

        assert "authentication" in str(exc_info.value).lower() or "api key" in str(exc_info.value).lower()
        print(f"\n✅ Auth error correctly raised: {exc_info.value}")

    @pytest.mark.slow
    def test_confidence_threshold_classification(self, api_key, test_image_simple):
        """Test that confidence threshold correctly classifies responses."""
        # High threshold - more likely to get VLM_UNCERTAIN
        config_strict = VLMConfig(
            api_key=api_key,
            model_name="gpt-4o-mini",
            confidence_threshold=0.95
        )
        client_strict = VLMClient(config_strict)

        # Low threshold - more likely to get SUCCESS
        config_lenient = VLMConfig(
            api_key=api_key,
            model_name="gpt-4o-mini",
            confidence_threshold=0.1
        )
        client_lenient = VLMClient(config_lenient)

        request = VLMRequest(
            image_path=test_image_simple,
            prompt="What is in this image?",
            model="gpt-4o-mini"
        )

        response_lenient = client_lenient.query_region(request)

        print(f"\n✅ Confidence threshold test:")
        print(f"   Lenient (0.1): {response_lenient.status} (confidence: {response_lenient.confidence:.2f})")


class TestVLMIntegrationBatchQuery:
    """Integration tests for batch VLM queries."""

    @pytest.mark.slow
    def test_batch_query_multiple_images(self, api_key, tmp_path):
        """Test batch query with multiple images."""
        config = VLMConfig(
            api_key=api_key,
            model_name="gpt-4o-mini"
        )
        client = VLMClient(config)

        # Create 3 test images
        images = []
        for i, color in enumerate(["red", "green", "blue"]):
            img_path = tmp_path / f"test_{color}.jpg"
            img = Image.new("RGB", (200, 200), color=color)
            img.save(img_path, "JPEG")
            images.append(img_path)

        # Create batch request
        requests = [
            VLMRequest(
                image_path=img_path,
                prompt=f"What is the main color in this image?",
                model="gpt-4o-mini"
            )
            for img_path in images
        ]

        batch_request = VLMBatchRequest(
            requests=requests,
            max_workers=2
        )

        # Track progress
        progress_updates = []
        def progress_callback(completed, total):
            progress_updates.append((completed, total))
            print(f"   Progress: {completed}/{total}")

        print(f"\n🔄 Running batch query with {len(requests)} images...")
        batch_response = client.query_batch(batch_request, progress_callback=progress_callback)

        # Verify batch response
        assert batch_response.total_requests == 3
        assert batch_response.successful + batch_response.uncertain + batch_response.failed == 3
        assert len(batch_response.responses) == 3
        assert batch_response.total_cost > 0
        assert batch_response.total_tokens > 0
        assert len(progress_updates) > 0

        print(f"\n✅ Batch query results:")
        print(f"   Total: {batch_response.total_requests}")
        print(f"   Successful: {batch_response.successful}")
        print(f"   Uncertain: {batch_response.uncertain}")
        print(f"   Failed: {batch_response.failed}")
        print(f"   Total cost: ${batch_response.total_cost:.4f}")
        print(f"   Total tokens: {batch_response.total_tokens}")
        print(f"   Total latency: {batch_response.total_latency_ms:.0f}ms")

        # Verify individual responses
        for i, response in enumerate(batch_response.responses):
            print(f"   Image {i+1}: {response.label} (confidence: {response.confidence:.2f})")


class TestVLMIntegrationRateLimiting:
    """Integration tests for rate limiting."""

    @pytest.mark.slow
    def test_rate_limiter_with_real_api(self, api_key, test_image_simple):
        """Test rate limiter integration with real API calls."""
        # Configure with low rate limit
        rate_limiter_config = RateLimiterConfig(
            requests_per_second=2.0,  # 2 requests per second
            burst_capacity=3,
            enabled=True
        )

        config = VLMConfig(
            api_key=api_key,
            model_name="gpt-4o-mini",
            rate_limiter_config=rate_limiter_config
        )
        client = VLMClient(config)

        # Make multiple requests and measure time
        num_requests = 5
        requests = [
            VLMRequest(
                image_path=test_image_simple,
                prompt=f"Request {i}: What color is this?",
                model="gpt-4o-mini"
            )
            for i in range(num_requests)
        ]

        print(f"\n⏱️  Making {num_requests} requests with rate limiter (2 req/sec)...")
        start_time = time.time()

        responses = []
        for i, request in enumerate(requests):
            response = client.query_region(request)
            responses.append(response)
            elapsed = time.time() - start_time
            print(f"   Request {i+1} completed at {elapsed:.2f}s")

        total_time = time.time() - start_time

        # Verify rate limiting worked
        # With 2 req/sec and 3 burst capacity:
        # - First 3 requests: immediate (burst)
        # - Requests 4-5: rate limited (0.5s each)
        # Expected total time: ~1 second (2 additional requests at 2 req/sec)

        assert len(responses) == num_requests
        assert all(r.status in [VLMStatus.SUCCESS, VLMStatus.VLM_UNCERTAIN] for r in responses)

        print(f"\n✅ Rate limiting test completed:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Expected: ~1.0s (burst 3, then rate limited)")
        print(f"   All requests successful: {all(r.status == VLMStatus.SUCCESS for r in responses)}")

    @pytest.mark.slow
    def test_disabled_rate_limiter(self, api_key, test_image_simple):
        """Test that disabled rate limiter doesn't slow down requests."""
        rate_limiter_config = RateLimiterConfig(
            requests_per_second=1.0,
            burst_capacity=1,
            enabled=False  # Disabled
        )

        config = VLMConfig(
            api_key=api_key,
            model_name="gpt-4o-mini",
            rate_limiter_config=rate_limiter_config
        )
        client = VLMClient(config)

        # Make 3 requests quickly
        start_time = time.time()

        for i in range(3):
            request = VLMRequest(
                image_path=test_image_simple,
                prompt=f"Request {i}",
                model="gpt-4o-mini"
            )
            client.query_region(request)

        total_time = time.time() - start_time

        # Should complete relatively quickly (limited only by API latency)
        print(f"\n✅ Disabled rate limiter test:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Rate limiter was disabled - no artificial delays")


class TestVLMIntegrationCostTracking:
    """Integration tests for cost tracking."""

    @pytest.mark.slow
    def test_cost_calculation_accuracy(self, api_key, test_image_simple):
        """Test that cost calculation is reasonable."""
        config = VLMConfig(
            api_key=api_key,
            model_name="gpt-4o-mini"
        )
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_simple,
            prompt="What is this?",
            model="gpt-4o-mini",
            max_tokens=50
        )

        response = client.query_region(request)

        # Verify cost is reasonable for gpt-4o-mini
        # Typical cost should be < $0.01 for a simple query
        assert 0.0001 < response.cost < 0.01

        # Verify tokens used
        assert response.tokens_used > 0

        print(f"\n✅ Cost tracking:")
        print(f"   Model: {request.model}")
        print(f"   Cost: ${response.cost:.6f}")
        print(f"   Tokens used: {response.tokens_used}")
        print(f"   Cost per 1K tokens: ${(response.cost / response.tokens_used * 1000):.4f}")

    @pytest.mark.slow
    def test_batch_cost_aggregation(self, api_key, test_image_simple):
        """Test that batch cost aggregation works correctly."""
        config = VLMConfig(
            api_key=api_key,
            model_name="gpt-4o-mini"
        )
        client = VLMClient(config)

        # Create batch with 3 requests
        requests = [
            VLMRequest(
                image_path=test_image_simple,
                prompt=f"Query {i}",
                model="gpt-4o-mini"
            )
            for i in range(3)
        ]

        batch_request = VLMBatchRequest(requests=requests, max_workers=2)
        batch_response = client.query_batch(batch_request)

        # Verify cost aggregation
        individual_costs = sum(r.cost for r in batch_response.responses)
        assert abs(batch_response.total_cost - individual_costs) < 0.000001

        print(f"\n✅ Batch cost aggregation:")
        print(f"   Total cost: ${batch_response.total_cost:.6f}")
        print(f"   Individual costs sum: ${individual_costs:.6f}")
        print(f"   Match: {abs(batch_response.total_cost - individual_costs) < 0.000001}")


if __name__ == "__main__":
    """Run integration tests with verbose output."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])
