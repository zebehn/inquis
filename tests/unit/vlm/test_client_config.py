"""Unit tests for VLM client configuration.

Tests configuration validation, default values, and error handling.
"""

import pytest
from pydantic import ValidationError

from src.vlm.client import VLMConfig, VLMClient
from src.vlm.exceptions import VLMAuthError


class TestVLMConfig:
    """Tests for VLMConfig model."""

    def test_valid_config_with_defaults(self, mock_api_key):
        """Test creating config with valid API key and default values."""
        config = VLMConfig(api_key=mock_api_key)

        assert config.api_key == mock_api_key
        assert config.model_name == "gpt-4o"
        assert config.timeout == 30
        assert config.max_retries == 5
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 60.0
        assert config.confidence_threshold == 0.7

    def test_valid_config_with_custom_values(self, mock_api_key):
        """Test creating config with custom values."""
        config = VLMConfig(
            api_key=mock_api_key,
            model_name="gpt-4o-mini",
            timeout=60,
            max_retries=3,
            retry_base_delay=2.0,
            retry_max_delay=120.0,
            confidence_threshold=0.8
        )

        assert config.model_name == "gpt-4o-mini"
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.retry_base_delay == 2.0
        assert config.retry_max_delay == 120.0
        assert config.confidence_threshold == 0.8

    def test_config_requires_api_key(self):
        """Test that config requires non-empty API key."""
        with pytest.raises(ValidationError) as exc_info:
            VLMConfig(api_key="")

        assert "api_key" in str(exc_info.value)

    def test_config_validates_timeout_range(self, mock_api_key):
        """Test that timeout must be within valid range."""
        # Too low
        with pytest.raises(ValidationError):
            VLMConfig(api_key=mock_api_key, timeout=0)

        # Too high
        with pytest.raises(ValidationError):
            VLMConfig(api_key=mock_api_key, timeout=400)

        # Valid boundaries
        config_min = VLMConfig(api_key=mock_api_key, timeout=1)
        config_max = VLMConfig(api_key=mock_api_key, timeout=300)
        assert config_min.timeout == 1
        assert config_max.timeout == 300

    def test_config_validates_max_retries_range(self, mock_api_key):
        """Test that max_retries must be within valid range."""
        # Valid values
        config_zero = VLMConfig(api_key=mock_api_key, max_retries=0)
        config_max = VLMConfig(api_key=mock_api_key, max_retries=10)
        assert config_zero.max_retries == 0
        assert config_max.max_retries == 10

        # Too high
        with pytest.raises(ValidationError):
            VLMConfig(api_key=mock_api_key, max_retries=20)

    def test_config_validates_confidence_threshold_range(self, mock_api_key):
        """Test that confidence_threshold must be between 0 and 1."""
        # Valid boundaries
        config_min = VLMConfig(api_key=mock_api_key, confidence_threshold=0.0)
        config_max = VLMConfig(api_key=mock_api_key, confidence_threshold=1.0)
        assert config_min.confidence_threshold == 0.0
        assert config_max.confidence_threshold == 1.0

        # Out of range
        with pytest.raises(ValidationError):
            VLMConfig(api_key=mock_api_key, confidence_threshold=1.5)

    def test_config_is_immutable(self, mock_api_key):
        """Test that config is frozen after creation."""
        config = VLMConfig(api_key=mock_api_key)

        with pytest.raises(ValidationError):
            config.model_name = "different-model"


class TestVLMClient:
    """Tests for VLMClient initialization."""

    def test_client_initialization_with_valid_config(self, mock_api_key):
        """Test initializing client with valid configuration."""
        config = VLMConfig(api_key=mock_api_key)
        client = VLMClient(config)

        assert client.config == config
        assert client._client is not None

    def test_client_stores_config(self, mock_api_key):
        """Test that client stores configuration reference."""
        config = VLMConfig(
            api_key=mock_api_key,
            model_name="gpt-4o-mini",
            timeout=45
        )
        client = VLMClient(config)

        assert client.config.model_name == "gpt-4o-mini"
        assert client.config.timeout == 45
