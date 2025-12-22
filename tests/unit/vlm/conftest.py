"""Pytest fixtures for VLM module unit tests.

This module provides reusable test fixtures for mocking VLM API responses,
creating test images, and setting up authentication.
"""

import pytest
import tempfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from unittest.mock import Mock, MagicMock
import numpy as np
import cv2


@pytest.fixture
def test_image_path(tmp_path):
    """Create a temporary test image file.

    Returns:
        Path: Path to temporary test image (300x300 RGB)
    """
    # Create a simple test image
    test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    # Save to temporary file
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))

    return image_path


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing.

    Returns:
        str: Mock API key
    """
    return "sk-test-mock-api-key-12345"


@pytest.fixture
def mock_openai_response():
    """Create a mock successful OpenAI API response.

    Returns:
        Mock: Mock response object with OpenAI structure
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"label": "excavator", "confidence": 0.89, "reasoning": "Large construction vehicle"}'
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 1000
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 1050

    return mock_response


@pytest.fixture
def mock_openai_error_response():
    """Create a mock error OpenAI API response.

    Returns:
        Mock: Mock error response
    """
    from openai import APIError

    error = APIError("Rate limit exceeded")
    error.status_code = 429
    error.response = Mock()
    error.response.headers = {"retry-after": "60"}

    return error


@pytest.fixture
def mock_vlm_request(test_image_path):
    """Create a mock VLM request object.

    Args:
        test_image_path: Fixture providing test image path

    Returns:
        VLMRequest: Mock request object
    """
    from src.vlm.models import VLMRequest

    return VLMRequest(
        region_id=uuid4(),
        image_path=test_image_path,
        prompt="Identify the object in this image",
        model="gpt-4o",
        max_tokens=500,
        temperature=0.2
    )


@pytest.fixture
def mock_vlm_response():
    """Create a mock successful VLM response object.

    Returns:
        VLMResponse: Mock response object
    """
    from src.vlm.models import VLMResponse, VLMStatus

    return VLMResponse(
        request_id=uuid4(),
        status=VLMStatus.SUCCESS,
        label="excavator",
        confidence=0.89,
        reasoning="Large construction vehicle with distinctive features",
        raw_response={"label": "excavator", "confidence": 0.89, "reasoning": "Large construction vehicle with distinctive features"},
        cost=0.0047,
        tokens_used=1050,
        latency_ms=1710.0,
        queried_at=datetime.now(),
        responded_at=datetime.now()
    )


@pytest.fixture
def mock_vlm_client(mock_api_key):
    """Create a mock VLM client for testing.

    Args:
        mock_api_key: Fixture providing mock API key

    Returns:
        Mock: Mock VLMClient object
    """
    mock_client = Mock()
    mock_client.config = Mock()
    mock_client.config.api_key = mock_api_key
    mock_client.config.model_name = "gpt-4o"
    mock_client.config.timeout = 30
    mock_client.config.max_retries = 5

    return mock_client


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client.

    Returns:
        Mock: Mock OpenAI client with chat.completions.create method
    """
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()

    return mock_client


@pytest.fixture
def sample_batch_requests(test_image_path):
    """Create a list of sample VLM requests for batch testing.

    Args:
        test_image_path: Fixture providing test image path

    Returns:
        List[VLMRequest]: List of mock requests
    """
    from src.vlm.models import VLMRequest

    return [
        VLMRequest(
            region_id=uuid4(),
            image_path=test_image_path,
            prompt=f"Identify object {i}",
            model="gpt-4o"
        )
        for i in range(5)
    ]


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state between tests.

    This fixture automatically runs before each test to ensure
    rate limiter state is clean.
    """
    # This will be implemented when rate limiter is created
    yield
    # Cleanup after test
