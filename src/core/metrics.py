"""Metrics tracking infrastructure for the visual perception agent.

Provides functionality for:
- Tracking model performance metrics
- Recording processing statistics
- Session-level metrics aggregation
- Metric history and comparison
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class SegmentationMetrics:
    """Metrics for segmentation performance."""

    timestamp: str
    total_frames: int
    total_instances: int
    avg_confidence: float
    uncertain_instances: int
    uncertain_percentage: float
    processing_time_seconds: float
    frames_per_second: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_frames": self.total_frames,
            "total_instances": self.total_instances,
            "avg_confidence": self.avg_confidence,
            "uncertain_instances": self.uncertain_instances,
            "uncertain_percentage": self.uncertain_percentage,
            "processing_time_seconds": self.processing_time_seconds,
            "frames_per_second": self.frames_per_second,
        }


@dataclass
class TrainingMetrics:
    """Metrics for model training."""

    timestamp: str
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    samples_trained: int
    duration_seconds: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
            "samples_trained": self.samples_trained,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class VLMMetrics:
    """Metrics for VLM query performance."""

    timestamp: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_response_time_seconds: float
    avg_confidence: float
    total_cost_usd: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "avg_response_time_seconds": self.avg_response_time_seconds,
            "avg_confidence": self.avg_confidence,
            "total_cost_usd": self.total_cost_usd,
        }


class MetricsTracker:
    """Tracker for aggregating and persisting metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.segmentation_history: List[SegmentationMetrics] = []
        self.training_history: List[TrainingMetrics] = []
        self.vlm_history: List[VLMMetrics] = []

    def record_segmentation(self, metrics: SegmentationMetrics) -> None:
        """Record segmentation metrics.

        Args:
            metrics: Segmentation metrics to record
        """
        self.segmentation_history.append(metrics)

    def record_training(self, metrics: TrainingMetrics) -> None:
        """Record training metrics.

        Args:
            metrics: Training metrics to record
        """
        self.training_history.append(metrics)

    def record_vlm(self, metrics: VLMMetrics) -> None:
        """Record VLM metrics.

        Args:
            metrics: VLM metrics to record
        """
        self.vlm_history.append(metrics)

    def get_latest_segmentation(self) -> Optional[SegmentationMetrics]:
        """Get most recent segmentation metrics.

        Returns:
            Latest segmentation metrics or None
        """
        return self.segmentation_history[-1] if self.segmentation_history else None

    def get_latest_training(self) -> Optional[TrainingMetrics]:
        """Get most recent training metrics.

        Returns:
            Latest training metrics or None
        """
        return self.training_history[-1] if self.training_history else None

    def get_latest_vlm(self) -> Optional[VLMMetrics]:
        """Get most recent VLM metrics.

        Returns:
            Latest VLM metrics or None
        """
        return self.vlm_history[-1] if self.vlm_history else None

    def compute_improvement(self) -> Optional[Dict]:
        """Compute improvement metrics between first and last segmentation.

        Returns:
            Dictionary with improvement metrics or None if insufficient data
        """
        if len(self.segmentation_history) < 2:
            return None

        first = self.segmentation_history[0]
        last = self.segmentation_history[-1]

        uncertainty_reduction = (
            first.uncertain_percentage - last.uncertain_percentage
        )

        confidence_improvement = last.avg_confidence - first.avg_confidence

        return {
            "uncertainty_reduction_percent": uncertainty_reduction,
            "confidence_improvement": confidence_improvement,
            "iterations": len(self.segmentation_history),
            "first_timestamp": first.timestamp,
            "last_timestamp": last.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save metrics to JSON file.

        Args:
            path: Path to save metrics JSON
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "segmentation": [m.to_dict() for m in self.segmentation_history],
            "training": [m.to_dict() for m in self.training_history],
            "vlm": [m.to_dict() for m in self.vlm_history],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load metrics from JSON file.

        Args:
            path: Path to metrics JSON file
        """
        with open(path) as f:
            data = json.load(f)

        # Load segmentation metrics
        self.segmentation_history = [
            SegmentationMetrics(**m) for m in data.get("segmentation", [])
        ]

        # Load training metrics
        self.training_history = [
            TrainingMetrics(**m) for m in data.get("training", [])
        ]

        # Load VLM metrics
        self.vlm_history = [VLMMetrics(**m) for m in data.get("vlm", [])]

    def clear(self) -> None:
        """Clear all metrics history."""
        self.segmentation_history.clear()
        self.training_history.clear()
        self.vlm_history.clear()
