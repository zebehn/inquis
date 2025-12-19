"""SemanticUncertaintyPattern Pydantic model for clustering similar VLM_UNCERTAIN regions.

TDD: T007 - Create SemanticUncertaintyPattern model for pattern detection
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
from uuid import UUID
from pydantic import BaseModel, field_validator, Field
from enum import Enum


class PatternStatus(str, Enum):
    """Status of a semantic uncertainty pattern."""

    UNRESOLVED = "unresolved"  # Pattern detected, not yet labeled
    IN_PROGRESS = "in_progress"  # User is reviewing/labeling
    RESOLVED = "resolved"  # Pattern labeled and confirmed


class SemanticUncertaintyPattern(BaseModel):
    """Represents a cluster of visually similar VLM_UNCERTAIN regions.

    Patterns are detected through DBSCAN clustering of uncertain regions based on
    visual similarity (bbox, mask, color features). They enable batch manual labeling
    of similar objects the VLM struggles with.

    Attributes:
        id: Unique pattern identifier
        job_id: Associated labeling job ID
        cluster_id: DBSCAN cluster label (integer)
        region_ids: List of region UUIDs in this pattern (min 2)
        region_count: Number of regions in pattern
        frames_affected: List of frame indices containing these regions
        sample_image_paths: Paths to representative sample images (max 5)
        avg_bbox_size: Average bounding box size (width, height)
        avg_similarity_score: Average intra-cluster similarity [0, 1]
        status: Pattern resolution status
        confirmed_label: Manual label applied to all regions (if resolved)
        created_at: When pattern was detected
        resolved_at: When pattern was resolved (if resolved)
    """

    id: UUID
    job_id: UUID
    cluster_id: int  # DBSCAN cluster label
    region_ids: List[UUID] = Field(min_length=2)  # Minimum 2 regions per pattern
    region_count: int = Field(ge=2)  # Must match len(region_ids)
    frames_affected: List[int] = Field(default_factory=list)  # Frame indices
    sample_image_paths: List[Path] = Field(default_factory=list, max_length=5)  # Max 5 samples
    avg_bbox_size: Tuple[float, float]  # (width, height)
    avg_similarity_score: float = Field(ge=0.0, le=1.0)  # Average similarity
    status: PatternStatus
    confirmed_label: Optional[str] = None  # Manual label (if resolved)
    created_at: datetime
    resolved_at: Optional[datetime] = None

    @field_validator("region_ids")
    @classmethod
    def validate_region_ids(cls, v: List[UUID]) -> List[UUID]:
        """Validate at least 2 regions in pattern."""
        if len(v) < 2:
            raise ValueError("Pattern must contain at least 2 regions")
        return v

    @field_validator("region_count")
    @classmethod
    def validate_region_count(cls, v: int) -> int:
        """Validate region count is at least 2."""
        if v < 2:
            raise ValueError("region_count must be >= 2")
        return v

    @field_validator("sample_image_paths")
    @classmethod
    def validate_sample_image_paths(cls, v: List[Path]) -> List[Path]:
        """Validate max 5 sample images."""
        if len(v) > 5:
            raise ValueError("Maximum 5 sample images allowed")
        return v

    @field_validator("avg_similarity_score")
    @classmethod
    def validate_avg_similarity_score(cls, v: float) -> float:
        """Validate similarity score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("avg_similarity_score must be between 0.0 and 1.0")
        return v

    @field_validator("avg_bbox_size")
    @classmethod
    def validate_avg_bbox_size(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Validate bbox size is positive."""
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError("avg_bbox_size dimensions must be positive")
        return v

    def is_resolved(self) -> bool:
        """Check if pattern has been resolved with manual label.

        Returns:
            True if status is RESOLVED
        """
        return self.status == PatternStatus.RESOLVED

    def is_unresolved(self) -> bool:
        """Check if pattern still needs manual labeling.

        Returns:
            True if status is UNRESOLVED
        """
        return self.status == PatternStatus.UNRESOLVED

    def resolve(self, label: str, resolved_at: Optional[datetime] = None) -> None:
        """Mark pattern as resolved with confirmed label.

        Args:
            label: Manual semantic label to apply
            resolved_at: Optional resolution timestamp (defaults to now)
        """
        self.status = PatternStatus.RESOLVED
        self.confirmed_label = label
        self.resolved_at = resolved_at or datetime.now()

    class Config:
        """Pydantic model configuration."""

        json_encoders = {
            Path: str,
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
