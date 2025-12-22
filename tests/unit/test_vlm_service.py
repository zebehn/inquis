"""Unit tests for VLMService.

TDD Tests:
- T071: VLMService.query_region() - Send image to GPT-5.2 API
- T072: VLMService.parse_response() - Extract label, confidence, reasoning
- T073: VLMService.evaluate_confidence() - Detect VLM_UNCERTAIN responses
- T074: VLMService manual label fallback workflow
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from uuid import uuid4
from pathlib import Path
from datetime import datetime
import base64

from src.services.vlm_service import VLMService
from src.models.vlm_query import VLMQuery, VLMQueryStatus, UserAction
from src.models.uncertain_region import UncertainRegion, RegionStatus


class TestVLMService:
    """Test suite for VLMService."""

    @pytest.fixture
    def vlm_service(self):
        """Create VLMService instance for testing."""
        return VLMService(api_key="test_api_key_12345")

    @pytest.fixture
    def sample_region(self, tmp_path):
        """Create a sample UncertainRegion for testing."""
        # Create dummy cropped image and mask paths
        cropped_path = tmp_path / "cropped_region.jpg"
        mask_path = tmp_path / "mask_region.png"
        cropped_path.write_bytes(b"fake_cropped_image")
        mask_path.write_bytes(b"fake_mask")

        return UncertainRegion(
            id=uuid4(),
            session_id=uuid4(),
            frame_id=uuid4(),
            frame_index=10,
            bbox=[100, 100, 200, 200],
            uncertainty_score=0.7,
            cropped_image_path=cropped_path,
            mask_path=mask_path,
            status=RegionStatus.PENDING_REVIEW,
            created_at=datetime.now(),
        )

    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """Create a temporary test image file."""
        image_path = tmp_path / "test_region.jpg"
        # Create a dummy image file
        image_path.write_bytes(b"fake_image_data")
        return image_path

    # T071: Test VLMService.query_region()
    @patch("src.vlm.VLMClient.query_region")
    def test_query_region_success_high_confidence(self, mock_vlm_query, sample_region, sample_image_path):
        """Test successful VLM query with high confidence response.

        TDD: T071 [US3] - Test VLMService.query_region() with successful response
        """
        from src.vlm import VLMResponse, VLMStatus

        # Mock VLMClient.query_region response
        mock_vlm_response = VLMResponse(
            request_id=sample_region.id,
            status=VLMStatus.SUCCESS,
            label="traffic_cone",
            confidence=0.95,
            reasoning="The object has the distinctive orange and white striped pattern of a traffic cone.",
            raw_response={"label": "traffic_cone", "confidence": 0.95, "reasoning": "The object has the distinctive orange and white striped pattern of a traffic cone."},
            cost=0.005,
            tokens_used=150,
            latency_ms=1500.0,
            queried_at=datetime.now(),
            responded_at=datetime.now(),
        )
        mock_vlm_query.return_value = mock_vlm_response

        # Create service
        vlm_service = VLMService(api_key="test_api_key_12345")

        # Execute query
        result = vlm_service.query_region(
            region=sample_region,
            image_path=sample_image_path,
            prompt="What object is shown in this image?"
        )

        # Assertions
        assert isinstance(result, VLMQuery)
        assert result.region_id == sample_region.id
        assert result.image_path == sample_image_path
        assert result.status == VLMQueryStatus.SUCCESS
        assert result.response["label"] == "traffic_cone"
        assert result.response["confidence"] == 0.95
        assert result.token_count == 150
        assert result.cost > 0  # Should calculate cost
        assert result.latency > 0
        assert result.responded_at is not None

        # Verify VLMClient was called
        mock_vlm_query.assert_called_once()

    @patch("src.vlm.VLMClient.query_region")
    def test_query_region_low_confidence_vlm_uncertain(self, mock_vlm_query, sample_region, sample_image_path):
        """Test VLM query with low confidence triggers VLM_UNCERTAIN status.

        TDD: T071, T073 [US3] - Test VLM_UNCERTAIN detection with low confidence
        """
        from src.vlm import VLMResponse, VLMStatus

        # Mock VLMClient.query_region response with low confidence
        mock_vlm_response = VLMResponse(
            request_id=sample_region.id,
            status=VLMStatus.VLM_UNCERTAIN,
            label="unknown_object",
            confidence=0.25,
            reasoning="The object is partially occluded and lighting conditions make identification uncertain.",
            raw_response={"label": "unknown_object", "confidence": 0.25, "reasoning": "The object is partially occluded and lighting conditions make identification uncertain."},
            cost=0.003,
            tokens_used=120,
            latency_ms=1200.0,
            queried_at=datetime.now(),
            responded_at=datetime.now(),
        )
        mock_vlm_query.return_value = mock_vlm_response

        # Create service
        vlm_service = VLMService(api_key="test_api_key_12345")

        # Execute query
        result = vlm_service.query_region(
            region=sample_region,
            image_path=sample_image_path,
            prompt="What object is shown in this image?"
        )

        # Assertions
        assert isinstance(result, VLMQuery)
        assert result.status == VLMQueryStatus.VLM_UNCERTAIN
        assert result.response["confidence"] < 0.5  # Low confidence threshold
        assert result.user_action is None  # Awaiting manual input
        assert result.user_modified_label is None
        assert result.requires_manual_input() is True

    @patch("src.vlm.VLMClient.query_region")
    def test_query_region_api_failure(self, mock_vlm_query, sample_region, sample_image_path):
        """Test VLM query handles API failures gracefully.

        TDD: T071 [US3] - Test error handling for API failures
        """
        from src.vlm import VLMException

        # Mock VLMClient to raise exception
        mock_vlm_query.side_effect = VLMException("API connection failed")

        # Create service
        vlm_service = VLMService(api_key="test_api_key_12345")

        # Execute query
        result = vlm_service.query_region(
            region=sample_region,
            image_path=sample_image_path,
            prompt="What object is shown in this image?"
        )

        # Assertions
        assert isinstance(result, VLMQuery)
        assert result.status == VLMQueryStatus.FAILED
        assert "API connection failed" in result.error_message
        assert result.token_count == 0
        assert result.cost == 0.0
        assert result.responded_at is None

    @patch("src.vlm.VLMClient.query_region")
    def test_query_region_rate_limited(self, mock_vlm_query, sample_region, sample_image_path):
        """Test VLM query handles rate limiting.

        TDD: T071 [US3] - Test rate limiting error handling
        """
        from src.vlm import VLMRateLimitError

        # Mock VLMClient to raise rate limit exception
        mock_vlm_query.side_effect = VLMRateLimitError(
            "Rate limit exceeded",
            retry_after=60
        )

        # Create service
        vlm_service = VLMService(api_key="test_api_key_12345")

        # Execute query
        result = vlm_service.query_region(
            region=sample_region,
            image_path=sample_image_path,
            prompt="What object is shown in this image?"
        )

        # Assertions
        assert isinstance(result, VLMQuery)
        assert result.status == VLMQueryStatus.RATE_LIMITED
        assert "rate limit" in result.error_message.lower()
        assert result.token_count == 0
        assert result.responded_at is None

    # T072: Test VLMService.parse_response()
    def test_parse_response_valid_json(self, vlm_service):
        """Test parsing valid VLM JSON response.

        TDD: T072 [US3] - Test VLMService.parse_response() with valid JSON
        """
        raw_response = '{"label": "bicycle", "confidence": 0.88, "reasoning": "Two wheels, handlebars, and pedals are clearly visible."}'

        parsed = vlm_service.parse_response(raw_response)

        assert parsed["label"] == "bicycle"
        assert parsed["confidence"] == 0.88
        assert "reasoning" in parsed
        assert parsed["raw_response"] == raw_response

    def test_parse_response_malformed_json(self, vlm_service):
        """Test parsing malformed JSON response falls back gracefully.

        TDD: T072 [US3] - Test error handling for malformed JSON
        """
        raw_response = '{"label": "car", "confidence": 0.9'  # Missing closing brace

        parsed = vlm_service.parse_response(raw_response)

        assert parsed["label"] == "unknown"
        assert parsed["confidence"] == 0.0
        assert parsed["reasoning"] == "Failed to parse VLM response"
        assert parsed["raw_response"] == raw_response

    def test_parse_response_missing_required_fields(self, vlm_service):
        """Test parsing response with missing required fields.

        TDD: T072 [US3] - Test handling of incomplete responses
        """
        raw_response = '{"label": "truck"}'  # Missing confidence and reasoning

        parsed = vlm_service.parse_response(raw_response)

        assert parsed["label"] == "truck"
        assert parsed["confidence"] == 0.0  # Default value
        assert parsed["reasoning"] == ""  # Default empty string
        assert parsed["raw_response"] == raw_response

    # T073: Test VLMService.evaluate_confidence()
    def test_evaluate_confidence_high(self, vlm_service):
        """Test confidence evaluation for high confidence responses.

        TDD: T073 [US3] - Test VLM_UNCERTAIN detection logic
        """
        response = {
            "label": "car",
            "confidence": 0.92,
            "reasoning": "Clear view of sedan with visible wheels and body."
        }

        status = vlm_service.evaluate_confidence(response)

        assert status == VLMQueryStatus.SUCCESS

    def test_evaluate_confidence_low(self, vlm_service):
        """Test confidence evaluation for low confidence responses.

        TDD: T073 [US3] - Test VLM_UNCERTAIN detection with low confidence
        """
        response = {
            "label": "unclear",
            "confidence": 0.35,
            "reasoning": "Object is too blurred to identify confidently."
        }

        status = vlm_service.evaluate_confidence(response)

        assert status == VLMQueryStatus.VLM_UNCERTAIN

    def test_evaluate_confidence_threshold_boundary(self, vlm_service):
        """Test confidence evaluation at threshold boundary.

        TDD: T073 [US3] - Test confidence threshold boundary (default 0.5)
        """
        # Just below threshold
        response_low = {"label": "item", "confidence": 0.49, "reasoning": "Uncertain"}
        status_low = vlm_service.evaluate_confidence(response_low)
        assert status_low == VLMQueryStatus.VLM_UNCERTAIN

        # At threshold
        response_at = {"label": "item", "confidence": 0.5, "reasoning": "Acceptable"}
        status_at = vlm_service.evaluate_confidence(response_at)
        assert status_at == VLMQueryStatus.SUCCESS

        # Above threshold
        response_high = {"label": "item", "confidence": 0.51, "reasoning": "Confident"}
        status_high = vlm_service.evaluate_confidence(response_high)
        assert status_high == VLMQueryStatus.SUCCESS

    def test_evaluate_confidence_ambiguous_language(self, vlm_service):
        """Test confidence evaluation detects ambiguous language in reasoning.

        TDD: T073 [US3] - Test VLM_UNCERTAIN detection via ambiguous language
        """
        response = {
            "label": "object",
            "confidence": 0.6,
            "reasoning": "I'm not sure what this is. It could be a box or maybe a container."
        }

        # Should detect uncertainty from ambiguous language keywords
        status = vlm_service.evaluate_confidence(response)

        assert status == VLMQueryStatus.VLM_UNCERTAIN

    # T074: Test manual label fallback workflow
    def test_apply_manual_label(self, vlm_service, sample_region):
        """Test applying manual label to VLM_UNCERTAIN query.

        TDD: T074 [US3] - Test manual label fallback workflow
        """
        # Create a VLM_UNCERTAIN query
        uncertain_query = VLMQuery(
            id=uuid4(),
            region_id=sample_region.id,
            image_path=Path("/data/test/region.jpg"),
            prompt="What is this object?",
            model_name="gpt-5.2",
            response={
                "label": "unknown",
                "confidence": 0.3,
                "reasoning": "Cannot identify object.",
                "raw_response": "..."
            },
            token_count=100,
            cost=0.005,
            latency=0.8,
            status=VLMQueryStatus.VLM_UNCERTAIN,
            queried_at=datetime.now(),
            responded_at=datetime.now(),
        )

        # Apply manual label
        updated_query = vlm_service.apply_manual_label(
            query=uncertain_query,
            manual_label="traffic_barrier"
        )

        # Assertions
        assert updated_query.user_action == UserAction.MODIFIED
        assert updated_query.user_modified_label == "traffic_barrier"
        assert updated_query.get_final_label() == "traffic_barrier"
        assert updated_query.requires_manual_input() is False

    def test_accept_vlm_suggestion(self, vlm_service, sample_region):
        """Test accepting VLM suggestion for successful query.

        TDD: T074 [US3] - Test user acceptance workflow
        """
        # Create a successful VLM query
        success_query = VLMQuery(
            id=uuid4(),
            region_id=sample_region.id,
            image_path=Path("/data/test/region.jpg"),
            prompt="What is this object?",
            model_name="gpt-5.2",
            response={
                "label": "bicycle",
                "confidence": 0.89,
                "reasoning": "Clear view of bicycle frame and wheels.",
                "raw_response": "..."
            },
            token_count=120,
            cost=0.006,
            latency=0.9,
            status=VLMQueryStatus.SUCCESS,
            queried_at=datetime.now(),
            responded_at=datetime.now(),
        )

        # Accept suggestion
        updated_query = vlm_service.accept_suggestion(query=success_query)

        # Assertions
        assert updated_query.user_action == UserAction.ACCEPTED
        assert updated_query.get_final_label() == "bicycle"

    def test_reject_vlm_suggestion(self, vlm_service, sample_region):
        """Test rejecting VLM suggestion.

        TDD: T074 [US3] - Test user rejection workflow
        """
        # Create a successful VLM query
        success_query = VLMQuery(
            id=uuid4(),
            region_id=sample_region.id,
            image_path=Path("/data/test/region.jpg"),
            prompt="What is this object?",
            model_name="gpt-5.2",
            response={
                "label": "box",
                "confidence": 0.75,
                "reasoning": "Rectangular shape visible.",
                "raw_response": "..."
            },
            token_count=100,
            cost=0.005,
            latency=0.7,
            status=VLMQueryStatus.SUCCESS,
            queried_at=datetime.now(),
            responded_at=datetime.now(),
        )

        # Reject suggestion and provide alternative
        updated_query = vlm_service.reject_suggestion(
            query=success_query,
            manual_label="traffic_cone"
        )

        # Assertions
        assert updated_query.user_action == UserAction.REJECTED
        assert updated_query.user_modified_label == "traffic_cone"
        assert updated_query.get_final_label() == "traffic_cone"

    def test_calculate_cost(self, vlm_service):
        """Test cost calculation for GPT-5.2 API usage.

        TDD: T071 [US3] - Test cost calculation logic
        """
        # GPT-5.2 pricing (example rates)
        input_tokens = 1000
        output_tokens = 500

        cost = vlm_service.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model="gpt-5.2"
        )

        # Cost should be positive and reasonable
        assert cost > 0
        assert cost < 1.0  # Should be less than $1 for this token count

    def test_encode_image_base64(self, vlm_service, tmp_path):
        """Test image encoding to base64 for API transmission.

        TDD: T071 [US3] - Test image encoding utility
        """
        # Create a test image file
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image_content_12345")

        # Encode image
        encoded = vlm_service.encode_image_base64(test_image)

        # Assertions
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == b"fake_image_content_12345"
