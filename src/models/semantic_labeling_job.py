"""SemanticLabelingJob Pydantic model for automatic labeling jobs.

TDD: T005 - Create SemanticLabelingJob model with pause/resume state management
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, field_validator, Field
from enum import Enum


class JobStatus(str, Enum):
    """Status of a semantic labeling job."""

    PENDING = "pending"  # Job created, not yet started
    RUNNING = "running"  # Job actively processing regions
    PAUSED = "paused"  # Job paused by user
    PAUSED_BUDGET_LIMIT = "paused_budget_limit"  # Auto-paused due to budget limit
    COMPLETED = "completed"  # Job completed successfully
    FAILED = "failed"  # Job failed with error
    CANCELLED = "cancelled"  # Job cancelled by user


class JobProgress(BaseModel):
    """Progress tracking for labeling job."""

    frames_total: int = Field(ge=0)
    frames_processed: int = Field(ge=0, default=0)
    frames_pending: int = Field(ge=0)
    regions_total: int = Field(ge=0)
    regions_completed: int = Field(ge=0, default=0)
    regions_pending: int = Field(ge=0)
    regions_failed: int = Field(ge=0, default=0)
    progress_percentage: float = Field(ge=0.0, le=100.0, default=0.0)


class CostTracking(BaseModel):
    """Cost tracking for VLM queries."""

    total_cost: float = Field(ge=0.0, default=0.0)
    total_tokens: int = Field(ge=0, default=0)
    budget_limit: Optional[float] = Field(ge=0.0, default=None)
    budget_consumed_percentage: float = Field(ge=0.0, default=0.0)
    queries_successful: int = Field(ge=0, default=0)
    queries_failed: int = Field(ge=0, default=0)
    queries_uncertain: int = Field(ge=0, default=0)
    average_cost_per_region: float = Field(ge=0.0, default=0.0)
    estimated_remaining_cost: Optional[float] = Field(ge=0.0, default=None)


class JobConfiguration(BaseModel):
    """Configuration parameters for labeling job."""

    frame_sampling: int = Field(ge=1, default=1)  # Process every Nth frame
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.5)  # VLM confidence threshold
    model_name: str = Field(default="gpt-5.2")  # VLM model to use
    enable_tracking: bool = Field(default=True)  # Enable region tracking optimization
    tracking_iou_threshold: float = Field(ge=0.0, le=1.0, default=0.7)  # IoU threshold for tracking


class JobTimestamps(BaseModel):
    """Timestamp tracking for job lifecycle."""

    created_at: datetime
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class SemanticLabelingJob(BaseModel):
    """Represents an automatic semantic labeling job with pause/resume support.

    This model tracks the complete state of a long-running VLM labeling job,
    including progress, cost tracking, and checkpoint data for pause/resume.

    Attributes:
        id: Unique job identifier
        session_id: Associated video session ID
        video_path: Path to video file being processed
        status: Current job status
        error_message: Error details if status is FAILED

        progress: Progress tracking metrics
        cost_tracking: Cost and budget tracking
        configuration: Job configuration parameters
        timestamps: Job lifecycle timestamps

        regions_completed: List of region IDs that have been successfully labeled
        regions_pending: List of region IDs still awaiting labeling
        in_flight_region: Optional region ID currently being processed

        last_checkpoint_at: Timestamp of last checkpoint save
        checkpoint_version: Incremental version number for checkpoint file rotation
    """

    id: UUID
    session_id: UUID
    video_path: Path
    status: JobStatus
    error_message: Optional[str] = None

    # Progress and tracking
    progress: JobProgress
    cost_tracking: CostTracking
    configuration: JobConfiguration
    timestamps: JobTimestamps

    # Checkpoint state for pause/resume
    regions_completed: List[UUID] = Field(default_factory=list)
    regions_pending: List[UUID] = Field(default_factory=list)
    in_flight_region: Optional[UUID] = None

    # Checkpoint metadata
    last_checkpoint_at: datetime
    checkpoint_version: int = Field(ge=0, default=0)

    @field_validator("status")
    @classmethod
    def validate_status_transitions(cls, v: JobStatus) -> JobStatus:
        """Validate job status is a valid JobStatus enum value."""
        if not isinstance(v, JobStatus):
            raise ValueError(f"Invalid job status: {v}")
        return v

    def is_pausable(self) -> bool:
        """Check if job can be paused.

        Returns:
            True if job status is RUNNING
        """
        return self.status == JobStatus.RUNNING

    def is_resumable(self) -> bool:
        """Check if job can be resumed.

        Returns:
            True if job status is PAUSED or PAUSED_BUDGET_LIMIT
        """
        return self.status in [JobStatus.PAUSED, JobStatus.PAUSED_BUDGET_LIMIT]

    def is_startable(self) -> bool:
        """Check if job can be started.

        Returns:
            True if job status is PENDING
        """
        return self.status == JobStatus.PENDING

    def is_terminal(self) -> bool:
        """Check if job is in a terminal state (cannot transition).

        Returns:
            True if job status is COMPLETED, FAILED, or CANCELLED
        """
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]

    def update_progress_percentage(self) -> None:
        """Calculate and update progress percentage based on regions completed."""
        if self.progress.regions_total > 0:
            self.progress.progress_percentage = (
                self.progress.regions_completed / self.progress.regions_total
            ) * 100.0
        else:
            self.progress.progress_percentage = 0.0

    def update_budget_consumed_percentage(self) -> None:
        """Calculate and update budget consumed percentage."""
        if self.cost_tracking.budget_limit and self.cost_tracking.budget_limit > 0:
            self.cost_tracking.budget_consumed_percentage = (
                self.cost_tracking.total_cost / self.cost_tracking.budget_limit
            ) * 100.0
        else:
            self.cost_tracking.budget_consumed_percentage = 0.0

    def update_average_cost_per_region(self) -> None:
        """Calculate and update average cost per region."""
        if self.cost_tracking.queries_successful > 0:
            self.cost_tracking.average_cost_per_region = (
                self.cost_tracking.total_cost / self.cost_tracking.queries_successful
            )
        else:
            self.cost_tracking.average_cost_per_region = 0.0

    def estimate_remaining_cost(self) -> None:
        """Estimate remaining cost based on average cost per region."""
        if self.cost_tracking.average_cost_per_region > 0 and self.progress.regions_pending > 0:
            self.cost_tracking.estimated_remaining_cost = (
                self.cost_tracking.average_cost_per_region * self.progress.regions_pending
            )
        else:
            self.cost_tracking.estimated_remaining_cost = None

    class Config:
        """Pydantic model configuration."""

        json_encoders = {
            Path: str,
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
