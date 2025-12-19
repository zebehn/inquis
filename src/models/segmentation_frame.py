"""SegmentationFrame Pydantic model for frame segmentation results."""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, field_validator, Field


class InstanceMask(BaseModel):
    """Represents a single instance mask with metadata.

    Attributes:
        mask_path: Path to binary mask file (.npz)
        class_label: Predicted class name (from segmentation model)
        confidence: Prediction confidence [0, 1]
        bbox: Bounding box [x, y, width, height]
        area: Mask area in pixels
        semantic_label: Optional VLM-provided semantic label (enhanced)
        vlm_query_id: Optional reference to VLM query
        semantic_label_source: Source of semantic label ("vlm", "manual", None)
    """

    mask_path: Path
    class_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[int] = Field(min_length=4, max_length=4)
    area: int = Field(gt=0)
    semantic_label: Optional[str] = None  # VLM-enhanced label
    vlm_query_id: Optional[UUID] = None
    semantic_label_source: Optional[str] = None  # "vlm", "manual", None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[int]) -> List[int]:
        """Validate bbox values are non-negative."""
        if any(val < 0 for val in v):
            raise ValueError("bbox values must be non-negative")
        return v

    class Config:
        """Pydantic model configuration."""

        json_encoders = {
            Path: str,
        }


class SegmentationFrame(BaseModel):
    """Stores segmentation results for a single video frame.

    Attributes:
        id: Unique identifier
        session_id: Foreign key to VideoSession
        frame_index: Frame number (0-indexed)
        timestamp: Frame timestamp in video (seconds)
        image_path: Path to extracted frame image
        masks: List of detected instance masks
        uncertainty_map_path: Optional path to uncertainty heatmap
        processing_time: Time taken to process frame (seconds)
        model_version_id: Which model version was used
        processed_at: When frame was processed
    """

    id: UUID
    session_id: UUID
    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    image_path: Path
    masks: List[InstanceMask] = Field(default_factory=list)
    uncertainty_map_path: Optional[Path] = None
    processing_time: float = Field(gt=0.0)
    model_version_id: UUID
    processed_at: datetime

    @field_validator("frame_index")
    @classmethod
    def validate_frame_index(cls, v: int) -> int:
        """Validate frame index is non-negative."""
        if v < 0:
            raise ValueError("frame_index must be >= 0")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        """Validate timestamp is non-negative."""
        if v < 0:
            raise ValueError("timestamp must be >= 0")
        return v

    class Config:
        """Pydantic model configuration."""

        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }
