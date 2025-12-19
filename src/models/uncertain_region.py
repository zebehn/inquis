"""UncertainRegion Pydantic model for uncertain segmentation regions.

TDD: T050 [US2] - Create UncertainRegion Pydantic model
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, field_validator, Field
from enum import Enum


class RegionStatus(str, Enum):
    """Status of an uncertain region in the labeling workflow."""

    PENDING_REVIEW = "pending_review"  # Detected but not yet reviewed
    QUERIED = "queried"  # VLM query sent
    CONFIRMED = "confirmed"  # Label confirmed by user
    REJECTED = "rejected"  # Region rejected as invalid


class UncertainRegion(BaseModel):
    """Represents an uncertain segmentation region requiring review.

    Attributes:
        id: Unique region identifier
        session_id: Associated video session ID
        frame_id: Associated frame ID
        frame_index: Frame index in video
        bbox: Bounding box [x, y, width, height]
        uncertainty_score: Uncertainty score [0, 1]
        cropped_image_path: Path to cropped region image
        mask_path: Path to cropped mask file
        top_predictions: List of top prediction candidates with confidences
        status: Current status in labeling workflow
        vlm_query_id: Optional VLM query ID if queried
        confirmed_label: Optional confirmed label from user
        created_at: Timestamp when region was detected
        reviewed_at: Optional timestamp when region was reviewed
    """

    id: UUID
    session_id: UUID
    frame_id: UUID
    frame_index: int = Field(ge=0)
    bbox: List[int] = Field(min_length=4, max_length=4)
    uncertainty_score: float = Field(ge=0.0, le=1.0)
    cropped_image_path: Path
    mask_path: Path
    top_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    status: RegionStatus
    vlm_query_id: Optional[UUID] = None
    confirmed_label: Optional[str] = None
    created_at: datetime
    reviewed_at: Optional[datetime] = None

    @field_validator("uncertainty_score")
    @classmethod
    def validate_uncertainty_score(cls, v: float) -> float:
        """Validate uncertainty score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("uncertainty_score must be between 0.0 and 1.0")
        return v

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[int]) -> List[int]:
        """Validate bbox has exactly 4 values."""
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 values [x, y, width, height]")
        return v

    @field_validator("frame_index")
    @classmethod
    def validate_frame_index(cls, v: int) -> int:
        """Validate frame index is non-negative."""
        if v < 0:
            raise ValueError("frame_index must be non-negative")
        return v

    class Config:
        """Pydantic model configuration."""

        json_encoders = {
            Path: str,
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
