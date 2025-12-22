"""Unit tests for VLM client batch queries.

Tests parallel execution, partial failures, and batch processing logic.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.vlm.client import VLMClient, VLMConfig
from src.vlm.models import VLMBatchRequest, VLMStatus


class TestVLMClientBatchQuery:
    """Tests for VLMClient.query_batch() method."""

    @patch("src.vlm.client.OpenAI")
    def test_successful_batch_query(self, mock_openai_class, mock_api_key, sample_batch_requests):
        """Test successful batch query processes all requests."""
        # Setup mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"label": "object", "confidence": 0.85, "reasoning": "Clear"}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 30
        mock_response.usage.total_tokens = 1030

        mock_client.chat.completions.create.return_value = mock_response

        # Create client and batch request
        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        batch_request = VLMBatchRequest(
            requests=sample_batch_requests,
            max_workers=3
        )

        # Execute batch query
        batch_response = client.query_batch(batch_request)

        # Verify batch response
        assert batch_response.total_requests == len(sample_batch_requests)
        assert batch_response.successful == len(sample_batch_requests)
        assert batch_response.failed == 0
        assert len(batch_response.responses) == len(sample_batch_requests)
        assert batch_response.total_cost > 0
        assert batch_response.total_tokens > 0
        assert batch_response.started_at is not None
        assert batch_response.completed_at is not None

    @patch("src.vlm.client.OpenAI")
    def test_batch_query_with_mixed_results(self, mock_openai_class, mock_api_key, sample_batch_requests):
        """Test batch query with mix of successful and uncertain responses."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Alternate between high and low confidence responses
        responses = [
            '{"label": "object1", "confidence": 0.9, "reasoning": "Clear"}',
            '{"label": "unknown", "confidence": 0.3, "reasoning": "Unclear"}',
            '{"label": "object3", "confidence": 0.85, "reasoning": "Clear"}',
            '{"label": "unknown", "confidence": 0.4, "reasoning": "Unclear"}',
            '{"label": "object5", "confidence": 0.95, "reasoning": "Very clear"}',
        ]

        call_count = [0]

        def mock_create(*args, **kwargs):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = responses[call_count[0] % len(responses)]
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 1000
            mock_response.usage.completion_tokens = 30
            mock_response.usage.total_tokens = 1030
            call_count[0] += 1
            return mock_response

        mock_client.chat.completions.create.side_effect = mock_create

        config = VLMConfig(api_key=mock_api_key, confidence_threshold=0.7)
        client = VLMClient(config)

        batch_request = VLMBatchRequest(
            requests=sample_batch_requests,
            max_workers=2
        )

        # Execute batch query
        batch_response = client.query_batch(batch_request)

        # Verify mixed results
        assert batch_response.total_requests == len(sample_batch_requests)
        assert batch_response.successful > 0
        assert batch_response.uncertain > 0
        assert batch_response.successful + batch_response.uncertain == len(sample_batch_requests)

    @patch("src.vlm.client.OpenAI")
    def test_batch_query_with_partial_failures(self, mock_openai_class, mock_api_key, sample_batch_requests):
        """Test batch query continues on partial failures when fail_fast=False."""
        from openai import APIError

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # First call succeeds, second fails, third succeeds
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise APIError("Network error")

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"label": "object", "confidence": 0.85, "reasoning": "Clear"}'
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 1000
            mock_response.usage.completion_tokens = 30
            mock_response.usage.total_tokens = 1030
            return mock_response

        mock_client.chat.completions.create.side_effect = mock_create

        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        batch_request = VLMBatchRequest(
            requests=sample_batch_requests,
            max_workers=1,  # Sequential to control order
            fail_fast=False  # Continue on error
        )

        # Execute batch query
        batch_response = client.query_batch(batch_request)

        # Verify partial failure handling
        assert batch_response.total_requests == len(sample_batch_requests)
        assert batch_response.failed >= 1
        assert batch_response.successful + batch_response.failed == len(sample_batch_requests)
        assert len(batch_response.responses) == len(sample_batch_requests)

    @patch("src.vlm.client.OpenAI")
    def test_batch_query_with_fail_fast(self, mock_openai_class, mock_api_key, sample_batch_requests):
        """Test batch query stops on first error when fail_fast=True."""
        from openai import APIError

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # First call succeeds, second fails
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise APIError("Network error")

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"label": "object", "confidence": 0.85, "reasoning": "Clear"}'
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 1000
            mock_response.usage.completion_tokens = 30
            mock_response.usage.total_tokens = 1030
            return mock_response

        mock_client.chat.completions.create.side_effect = mock_create

        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        batch_request = VLMBatchRequest(
            requests=sample_batch_requests,
            max_workers=1,  # Sequential
            fail_fast=True  # Stop on error
        )

        # Execute batch query
        batch_response = client.query_batch(batch_request)

        # Verify early termination
        assert len(batch_response.responses) <= len(sample_batch_requests)
        assert batch_response.failed >= 1

    @patch("src.vlm.client.OpenAI")
    def test_batch_query_with_progress_callback(self, mock_openai_class, mock_api_key, sample_batch_requests):
        """Test batch query calls progress callback."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"label": "object", "confidence": 0.85, "reasoning": "Clear"}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 30
        mock_response.usage.total_tokens = 1030

        mock_client.chat.completions.create.return_value = mock_response

        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        batch_request = VLMBatchRequest(
            requests=sample_batch_requests,
            max_workers=2
        )

        # Track progress callbacks
        progress_updates = []

        def progress_callback(completed, total):
            progress_updates.append((completed, total))

        # Execute batch query with callback
        batch_response = client.query_batch(batch_request, progress_callback=progress_callback)

        # Verify progress callbacks were made
        assert len(progress_updates) > 0
        assert all(total == len(sample_batch_requests) for _, total in progress_updates)
        assert progress_updates[-1][0] == len(sample_batch_requests)  # Final callback

    def test_batch_request_validates_empty_list(self, mock_api_key):
        """Test that batch request rejects empty request list."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VLMBatchRequest(requests=[])

    def test_batch_request_validates_max_workers_range(self, sample_batch_requests):
        """Test that batch request validates max_workers range."""
        from pydantic import ValidationError

        # Valid range
        batch_valid_min = VLMBatchRequest(requests=sample_batch_requests, max_workers=1)
        batch_valid_max = VLMBatchRequest(requests=sample_batch_requests, max_workers=50)
        assert batch_valid_min.max_workers == 1
        assert batch_valid_max.max_workers == 50

        # Out of range
        with pytest.raises(ValidationError):
            VLMBatchRequest(requests=sample_batch_requests, max_workers=0)

        with pytest.raises(ValidationError):
            VLMBatchRequest(requests=sample_batch_requests, max_workers=100)
