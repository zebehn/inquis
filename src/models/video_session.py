"""VideoSession Pydantic model for video analysis sessions."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, field_validator, Field


class SessionStatus(str, Enum):
    """Video session processing status."""

    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoSession(BaseModel):
    """Represents a video analysis session from upload through processing.

    Attributes:
        id: Unique identifier
        filename: Original video filename
        filepath: Path to stored video file
        upload_timestamp: When video was uploaded
        status: Current processing status
        metadata: Video metadata (resolution, frame_count, duration, fps, codec)
        processing_progress: Progress from 0.0 to 1.0
        error_message: Error details if status is failed
        created_at: Session creation time
        updated_at: Last modification time
    """

    id: UUID
    filename: str
    filepath: Path
    upload_timestamp: datetime
    status: SessionStatus
    metadata: Dict[str, Any]
    processing_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @field_validator("filename")
    @classmethod
    def validate_filename_extension(cls, v: str) -> str:
        """Validate that filename has a supported video extension."""
        valid_extensions = [".mp4", ".avi", ".mov"]
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Invalid file extension. Must be one of {valid_extensions}"
            )
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video metadata constraints."""
        # Validate duration (max 600 seconds)
        if "duration" in v and v["duration"] > 600.0:
            raise ValueError("Video duration must be ≤ 600 seconds (10 minutes)")

        # Validate FPS (15-60)
        if "fps" in v:
            if v["fps"] < 15.0 or v["fps"] > 60.0:
                raise ValueError("Video fps must be between 15 and 60")

        # Validate frame_count is positive
        if "frame_count" in v and v["frame_count"] <= 0:
            raise ValueError("frame_count must be > 0")

        return v

    @field_validator("processing_progress")
    @classmethod
    def validate_progress(cls, v: float) -> float:
        """Validate processing progress is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("processing_progress must be between 0.0 and 1.0")
        return v

    class Config:
        """Pydantic model configuration."""

        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }
