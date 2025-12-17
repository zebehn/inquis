"""Unit tests for configuration manager.

TDD: These tests are written FIRST and must FAIL before implementation.
"""

import pytest
from pathlib import Path
from src.core.config import ConfigManager


class TestConfigManager:
    """Test configuration loading and validation."""

    def test_load_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
sam2:
  checkpoint: ./models/sam2.pt
  confidence_threshold: 0.75

vlm:
  model: gpt-5.2
  max_tokens: 300
""")

        config = ConfigManager(config_path=config_file)

        assert config.sam2.checkpoint == "./models/sam2.pt"
        assert config.sam2.confidence_threshold == 0.75
        assert config.vlm.model == "gpt-5.2"
        assert config.vlm.max_tokens == 300

    def test_load_config_with_env_override(self, tmp_path, monkeypatch):
        """Test environment variables override YAML config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
sam2:
  confidence_threshold: 0.75
""")

        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.85")
        config = ConfigManager(config_path=config_file)

        assert config.sam2.confidence_threshold == 0.85

    def test_default_config_values(self):
        """Test default configuration values are set."""
        config = ConfigManager()

        assert config.sam2.device in ["cuda", "cpu", "mps"]
        assert config.sam2.confidence_threshold > 0
        assert config.sam2.uncertainty_threshold > 0

    def test_validate_config_paths(self, tmp_path):
        """Test configuration validates required paths."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
sam2:
  checkpoint: /nonexistent/path.pt
""")

        with pytest.raises(ValueError, match="checkpoint.*not exist"):
            config = ConfigManager(config_path=config_file)
            config.validate()

    def test_get_nested_config_value(self):
        """Test retrieving nested configuration values."""
        config = ConfigManager()

        value = config.get("sam2.confidence_threshold")
        assert isinstance(value, float)
        assert 0 < value < 1

    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ConfigManager()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "sam2" in config_dict
        assert "vlm" in config_dict
        assert "training" in config_dict

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test invalid YAML raises appropriate error."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content:")

        with pytest.raises(ValueError, match="Invalid YAML"):
            ConfigManager(config_path=config_file)
