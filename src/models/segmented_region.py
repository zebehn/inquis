"""SegmentedRegion Pydantic model for segmented regions with tracking support.

TDD: T006 - Create SegmentedRegion model with tracking fields for region tracking optimization
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, field_validator, Field
from enum import Enum


class TrackingStatus(str, Enum):
    """Status of region tracking across frames."""

    NEW = "new"  # New region, not tracked from previous frame
    TRACKED = "tracked"  # Successfully tracked from previous frame (IoU >= threshold)
    OCCLUDED = "occluded"  # Temporarily occluded (within 3-frame tolerance)
    SPLIT = "split"  # Region split into multiple regions
    MERGED = "merged"  # Region merged from multiple regions
    ENDED = "ended"  # Region no longer appears (ended after occlusion tolerance)


class SemanticLabelSource(str, Enum):
    """Source of semantic label for region."""

    VLM = "vlm"  # Label from VLM query
    TRACKED = "tracked"  # Label inherited from tracked parent region
    MANUAL = "manual"  # Manual label from user
    UNCERTAIN = "uncertain"  # VLM uncertain, needs manual labeling


class SegmentedRegion(BaseModel):
    """Represents a segmented region from SAM2 with semantic labeling.

    Base model for all segmented regions, containing segmentation output
    (mask, bbox, confidence) and semantic labeling information.

    Attributes:
        id: Unique region identifier
        session_id: Associated video session ID
        frame_id: Associated frame ID
        frame_index: Frame index in video
        bbox: Bounding box [x, y, width, height]
        mask_path: Path to binary mask file (.npz)
        area: Mask area in pixels
        confidence: Segmentation confidence [0, 1] (predicted_iou from SAM2)
        semantic_label: Semantic class label (e.g., "car", "person")
        semantic_label_source: Source of semantic label
        vlm_query_id: Optional VLM query ID if labeled by VLM
        created_at: When region was detected
    """

    id: UUID
    session_id: UUID
    frame_id: UUID
    frame_index: int = Field(ge=0)
    bbox: List[int] = Field(min_length=4, max_length=4)
    mask_path: Path
    area: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)  # Segmentation confidence (predicted_iou)
    semantic_label: Optional[str] = None  # Semantic class label
    semantic_label_source: SemanticLabelSource = SemanticLabelSource.VLM
    vlm_query_id: Optional[UUID] = None
    created_at: datetime

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[int]) -> List[int]:
        """Validate bbox has exactly 4 non-negative values."""
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 values [x, y, width, height]")
        if any(val < 0 for val in v):
            raise ValueError("bbox values must be non-negative")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
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


class TrackedRegion(SegmentedRegion):
    """SegmentedRegion with region tracking metadata for cost optimization.

    Extends SegmentedRegion with tracking fields to enable cross-frame region
    tracking, allowing VLM query skipping for regions that match across frames
    (60-70% cost reduction for stable scenes).

    Additional Attributes (beyond SegmentedRegion):
        tracking_status: Status of region tracking across frames
        parent_region_id: UUID of region from previous frame (if tracked)
        tracked_iou: IoU score with parent region (if tracked)
        frames_since_last_query: Number of frames since last VLM query
        tracking_chain_length: Number of consecutive frames this region was tracked
    """

    tracking_status: TrackingStatus = TrackingStatus.NEW
    parent_region_id: Optional[UUID] = None  # Region ID from previous frame
    tracked_iou: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # IoU with parent
    frames_since_last_query: int = Field(ge=0, default=0)  # Frames since VLM query
    tracking_chain_length: int = Field(ge=0, default=0)  # Consecutive tracked frames

    @field_validator("tracked_iou")
    @classmethod
    def validate_tracked_iou(cls, v: Optional[float]) -> Optional[float]:
        """Validate tracked IoU is between 0 and 1."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("tracked_iou must be between 0.0 and 1.0")
        return v

    def is_tracked(self) -> bool:
        """Check if region is successfully tracked from previous frame.

        Returns:
            True if tracking_status is TRACKED
        """
        return self.tracking_status == TrackingStatus.TRACKED

    def is_new(self) -> bool:
        """Check if region is new (not tracked).

        Returns:
            True if tracking_status is NEW
        """
        return self.tracking_status == TrackingStatus.NEW

    def is_occluded(self) -> bool:
        """Check if region is temporarily occluded.

        Returns:
            True if tracking_status is OCCLUDED
        """
        return self.tracking_status == TrackingStatus.OCCLUDED

    def needs_vlm_query(self) -> bool:
        """Check if region needs VLM query.

        Returns:
            True if region is NEW or has no semantic label
        """
        return (
            self.tracking_status == TrackingStatus.NEW
            or self.semantic_label is None
            or self.semantic_label_source == SemanticLabelSource.UNCERTAIN
        )

    def inherit_label_from_parent(self, parent: "TrackedRegion") -> None:
        """Inherit semantic label from tracked parent region.

        Args:
            parent: Parent region from previous frame
        """
        self.semantic_label = parent.semantic_label
        self.semantic_label_source = SemanticLabelSource.TRACKED
        self.parent_region_id = parent.id
        self.tracking_status = TrackingStatus.TRACKED
        self.tracking_chain_length = parent.tracking_chain_length + 1
        self.frames_since_last_query = parent.frames_since_last_query + 1
