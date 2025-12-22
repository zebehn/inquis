"""Configuration manager for loading and validating application config.

Provides centralized configuration management with:
- YAML file loading
- Environment variable overrides
- Nested config access
- Path validation
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field


@dataclass
class SAM2Config:
    """SAM2 segmentation model configuration."""

    checkpoint: str = "./data/models/base/sam2_hiera_large.pt"
    config: str = "sam2_hiera_l.yaml"
    device: str = "cuda"
    confidence_threshold: float = 0.75
    uncertainty_threshold: float = 0.25


@dataclass
class ZImageConfig:
    """Z-Image generation model configuration."""

    model_path: str = "./data/models/base/zimage"
    model_variant: str = "turbo"
    device: str = "cuda"
    inference_steps: int = 8
    quality_threshold: float = 0.7
    num_images_per_label: int = 20


@dataclass
class VLMConfig:
    """Vision-Language Model configuration."""

    model: str = "gpt-4o"  # OpenAI vision model (gpt-4o, gpt-4o-mini, gpt-4-turbo)
    max_tokens: int = 300
    temperature: float = 0.1
    retry_max_attempts: int = 3
    retry_backoff_factor: int = 2
    prompt_template: str = "What object is in this image? Provide: 1) Specific label 2) Confidence 3) Reasoning"


@dataclass
class TrainingConfig:
    """Model training configuration."""

    batch_size: int = 4
    learning_rate: float = 0.0001
    epochs: int = 5
    lora_rank: int = 8
    validation_split: float = 0.1
    early_stopping_patience: int = 3


@dataclass
class StorageConfig:
    """Storage paths configuration."""

    data_dir: str = "./data"
    sessions_dir: str = "./data/sessions"
    models_dir: str = "./data/models"
    datasets_dir: str = "./data/datasets"
    max_session_size_gb: int = 50
    cleanup_after_days: int = 30


@dataclass
class VideoConfig:
    """Video processing configuration."""

    max_duration_seconds: int = 600
    min_fps: int = 15
    max_fps: int = 60
    supported_formats: list = field(default_factory=lambda: ["mp4", "avi", "mov"])


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""

    batch_process_frames: int = 4
    max_concurrent_vlm_queries: int = 5
    enable_mixed_precision: bool = True
    cache_embeddings: bool = True
    gpu_memory_fraction: float = 0.9


@dataclass
class GUIConfig:
    """GUI configuration."""

    title: str = "Visual Perception Agent"
    layout: str = "wide"
    theme: str = "light"
    max_upload_size_mb: int = 500


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/perception_agent.log"
    max_file_size_mb: int = 10
    backup_count: int = 5


class ConfigManager:
    """Manager for application configuration."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to YAML config file. If None, uses defaults.
        """
        # Initialize with defaults
        self.sam2 = SAM2Config()
        self.zimage = ZImageConfig()
        self.vlm = VLMConfig()
        self.training = TrainingConfig()
        self.storage = StorageConfig()
        self.video = VideoConfig()
        self.performance = PerformanceConfig()
        self.gui = GUIConfig()
        self.logging = LoggingConfig()

        # Load from file if provided
        if config_path is not None:
            self._load_from_yaml(config_path)

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _load_from_yaml(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Raises:
            ValueError: If YAML is invalid
        """
        config_path = Path(config_path)

        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

        # Update dataclass instances with loaded values
        if "sam2" in config_dict:
            self._update_dataclass(self.sam2, config_dict["sam2"])
        if "zimage" in config_dict:
            self._update_dataclass(self.zimage, config_dict["zimage"])
        if "vlm" in config_dict:
            self._update_dataclass(self.vlm, config_dict["vlm"])
        if "training" in config_dict:
            self._update_dataclass(self.training, config_dict["training"])
        if "storage" in config_dict:
            self._update_dataclass(self.storage, config_dict["storage"])
        if "video" in config_dict:
            self._update_dataclass(self.video, config_dict["video"])
        if "performance" in config_dict:
            self._update_dataclass(self.performance, config_dict["performance"])
        if "gui" in config_dict:
            self._update_dataclass(self.gui, config_dict["gui"])
        if "logging" in config_dict:
            self._update_dataclass(self.logging, config_dict["logging"])

    def _update_dataclass(self, instance: Any, values: Dict[str, Any]) -> None:
        """Update dataclass instance with dictionary values.

        Args:
            instance: Dataclass instance to update
            values: Dictionary of field values
        """
        for key, value in values.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # SAM2 overrides
        if "CONFIDENCE_THRESHOLD" in os.environ:
            self.sam2.confidence_threshold = float(os.environ["CONFIDENCE_THRESHOLD"])
        if "DEVICE" in os.environ:
            self.sam2.device = os.environ["DEVICE"]
            self.zimage.device = os.environ["DEVICE"]

        # Storage overrides
        if "DATA_DIR" in os.environ:
            self.storage.data_dir = os.environ["DATA_DIR"]

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate SAM2 checkpoint path
        checkpoint_path = Path(self.sam2.checkpoint)
        # Check if path looks like it should exist (absolute or has .pt extension)
        if checkpoint_path.is_absolute() or checkpoint_path.suffix == ".pt":
            if not checkpoint_path.exists():
                raise ValueError(
                    f"SAM2 checkpoint does not exist: {checkpoint_path}"
                )

        # Validate thresholds
        if not 0 < self.sam2.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 < self.sam2.uncertainty_threshold < 1:
            raise ValueError("uncertainty_threshold must be between 0 and 1")

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "sam2.confidence_threshold")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        value = self

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as nested dictionary
        """
        return {
            "sam2": self._dataclass_to_dict(self.sam2),
            "zimage": self._dataclass_to_dict(self.zimage),
            "vlm": self._dataclass_to_dict(self.vlm),
            "training": self._dataclass_to_dict(self.training),
            "storage": self._dataclass_to_dict(self.storage),
            "video": self._dataclass_to_dict(self.video),
            "performance": self._dataclass_to_dict(self.performance),
            "gui": self._dataclass_to_dict(self.gui),
            "logging": self._dataclass_to_dict(self.logging),
        }

    def _dataclass_to_dict(self, instance: Any) -> Dict[str, Any]:
        """Convert dataclass instance to dictionary.

        Args:
            instance: Dataclass instance

        Returns:
            Dictionary representation
        """
        return {k: v for k, v in instance.__dict__.items()}
