"""Unit tests for VLM client single region queries.

Tests successful queries, error handling, and response parsing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.vlm.client import VLMClient, VLMConfig
from src.vlm.models import VLMRequest, VLMStatus
from src.vlm.exceptions import (
    VLMAuthError,
    VLMRateLimitError,
    VLMResponseError,
    VLMNetworkError,
)


class TestVLMClientSingleQuery:
    """Tests for VLMClient.query_region() method."""

    @patch("src.vlm.client.OpenAI")
    def test_successful_query(self, mock_openai_class, mock_api_key, test_image_path):
        """Test successful VLM query returns proper response."""
        # Setup mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"label": "excavator", "confidence": 0.89, "reasoning": "Construction vehicle"}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 1050

        mock_client.chat.completions.create.return_value = mock_response

        # Create client and request
        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_path,
            prompt="Identify the object",
            model="gpt-4o"
        )

        # Execute query
        response = client.query_region(request)

        # Verify response
        assert response.status == VLMStatus.SUCCESS
        assert response.label == "excavator"
        assert response.confidence == 0.89
        assert response.reasoning == "Construction vehicle"
        assert response.tokens_used == 1050
        assert response.cost > 0
        assert response.latency_ms > 0
        assert response.queried_at is not None
        assert response.responded_at is not None

    @patch("src.vlm.client.OpenAI")
    def test_uncertain_query(self, mock_openai_class, mock_api_key, test_image_path):
        """Test query with low confidence returns VLM_UNCERTAIN status."""
        # Setup mock with low confidence response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"label": "unknown", "confidence": 0.3, "reasoning": "Unclear object"}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 30
        mock_response.usage.total_tokens = 1030

        mock_client.chat.completions.create.return_value = mock_response

        # Create client with confidence threshold 0.7
        config = VLMConfig(api_key=mock_api_key, confidence_threshold=0.7)
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_path,
            prompt="What is this?",
            model="gpt-4o"
        )

        # Execute query
        response = client.query_region(request)

        # Verify uncertain status
        assert response.status == VLMStatus.VLM_UNCERTAIN
        assert response.label == "unknown"
        assert response.confidence == 0.3

    @patch("src.vlm.client.OpenAI")
    def test_query_with_rate_limit_error(self, mock_openai_class, mock_api_key, test_image_path):
        """Test query handles rate limit errors."""
        from openai import RateLimitError as OpenAIRateLimitError

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Create proper mock response for RateLimitError
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "5"}

        # Simulate rate limit error with proper signature
        rate_limit_error = OpenAIRateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        mock_client.chat.completions.create.side_effect = rate_limit_error

        config = VLMConfig(api_key=mock_api_key, max_retries=0)
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_path,
            prompt="Identify object",
            model="gpt-4o"
        )

        # Should raise VLMRateLimitError
        with pytest.raises(VLMRateLimitError) as exc_info:
            client.query_region(request)

        assert "Rate limit exceeded" in str(exc_info.value)

    @patch("src.vlm.client.OpenAI")
    def test_query_with_auth_error(self, mock_openai_class, mock_api_key, test_image_path):
        """Test query handles authentication errors."""
        from openai import APIError

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Create proper mock request for APIError
        mock_request = Mock()
        mock_request.method = "POST"
        mock_request.url = "https://api.openai.com/v1/chat/completions"

        # Simulate auth error with proper signature
        auth_error = APIError(
            message="Invalid API key",
            request=mock_request,
            body={"error": {"message": "Invalid API key"}}
        )
        mock_client.chat.completions.create.side_effect = auth_error

        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_path,
            prompt="Identify object",
            model="gpt-4o"
        )

        # Should raise VLMAuthError
        with pytest.raises(VLMAuthError) as exc_info:
            client.query_region(request)

        assert "authentication" in str(exc_info.value).lower() or "api key" in str(exc_info.value).lower()

    @patch("src.vlm.client.OpenAI")
    def test_query_with_malformed_json_response(self, mock_openai_class, mock_api_key, test_image_path):
        """Test query handles malformed JSON responses."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Return invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'This is not valid JSON'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 1010

        mock_client.chat.completions.create.return_value = mock_response

        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_path,
            prompt="Identify object",
            model="gpt-4o"
        )

        # Should raise VLMResponseError
        with pytest.raises(VLMResponseError) as exc_info:
            client.query_region(request)

        assert "JSON" in str(exc_info.value)

    @patch("src.vlm.client.OpenAI")
    def test_query_with_missing_label_in_response(self, mock_openai_class, mock_api_key, test_image_path):
        """Test query handles responses missing required 'label' field."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Return JSON without 'label' field
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"confidence": 0.9, "reasoning": "Clear"}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 1020

        mock_client.chat.completions.create.return_value = mock_response

        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        request = VLMRequest(
            image_path=test_image_path,
            prompt="Identify object",
            model="gpt-4o"
        )

        # Should raise VLMResponseError
        with pytest.raises(VLMResponseError) as exc_info:
            client.query_region(request)

        assert "label" in str(exc_info.value)

    def test_query_with_nonexistent_image(self, mock_api_key):
        """Test query with nonexistent image file raises error."""
        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        # This should fail during request validation
        with pytest.raises(ValueError):
            request = VLMRequest(
                image_path=Path("/nonexistent/image.jpg"),
                prompt="Identify object",
                model="gpt-4o"
            )
