"""Pydantic models for VLM requests and responses.

This module defines structured data models for VLM API interactions,
providing validation and type safety.
"""

from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class VLMStatus(str, Enum):
    """Status of VLM query execution."""
    SUCCESS = "success"
    VLM_UNCERTAIN = "vlm_uncertain"  # Low confidence response
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


class VLMRequest(BaseModel):
    """Request model for single VLM query.

    Attributes:
        region_id: Unique identifier for the region being queried
        image_path: Path to image file to analyze
        prompt: Text prompt for the VLM
        model: VLM model name (e.g., "gpt-4o")
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
    """

    region_id: UUID = Field(default_factory=uuid4)
    image_path: Path
    prompt: str = Field(min_length=1)
    model: str = Field(default="gpt-4o")
    max_tokens: int = Field(default=500, ge=1, le=4000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("image_path")
    @classmethod
    def validate_image_exists(cls, v: Path) -> Path:
        """Validate that image file exists."""
        if not v.exists():
            raise ValueError(f"Image file does not exist: {v}")
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, v: str) -> str:
        """Validate that prompt is not empty."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v


class VLMResponse(BaseModel):
    """Response model from VLM query.

    Attributes:
        request_id: UUID of original request
        status: Query execution status
        label: Semantic label returned by VLM
        confidence: Confidence score (0.0-1.0)
        reasoning: VLM's explanation for the label
        raw_response: Original API response
        cost: Estimated cost in USD
        tokens_used: Total tokens consumed
        latency_ms: Query latency in milliseconds
        queried_at: Timestamp when query was sent
        responded_at: Timestamp when response received
        error_message: Error details if status=FAILED
    """

    request_id: UUID
    status: VLMStatus
    label: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    cost: float = Field(default=0.0, ge=0.0)
    tokens_used: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    queried_at: datetime = Field(default_factory=datetime.now)
    responded_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: Optional[float]) -> Optional[float]:
        """Validate confidence is between 0 and 1."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class VLMBatchRequest(BaseModel):
    """Request model for batch VLM queries.

    Attributes:
        requests: List of individual VLM requests
        max_workers: Number of parallel workers for execution
        fail_fast: If True, stop on first error; if False, continue processing
    """

    requests: List[VLMRequest] = Field(min_length=1)
    max_workers: int = Field(default=10, ge=1, le=50)
    fail_fast: bool = Field(default=False)

    @field_validator("requests")
    @classmethod
    def validate_requests_not_empty(cls, v: List[VLMRequest]) -> List[VLMRequest]:
        """Validate that requests list is not empty."""
        if not v:
            raise ValueError("Batch request must contain at least one request")
        return v


class VLMBatchResponse(BaseModel):
    """Response model from batch VLM queries.

    Attributes:
        responses: List of individual VLM responses
        total_requests: Total number of requests in batch
        successful: Number of successful responses
        failed: Number of failed responses
        uncertain: Number of uncertain responses
        total_cost: Total cost for batch in USD
        total_tokens: Total tokens consumed
        total_latency_ms: Total processing time
        started_at: Batch start timestamp
        completed_at: Batch completion timestamp
    """

    responses: List[VLMResponse]
    total_requests: int = Field(ge=0)
    successful: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    uncertain: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
