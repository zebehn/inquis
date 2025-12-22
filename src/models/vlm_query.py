"""VLMQuery model for vision-language model queries.

TDD: T080 [US3] - Implement VLMQuery Pydantic model with VLM_UNCERTAIN support
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from uuid import UUID
from pathlib import Path
from datetime import datetime
from enum import Enum


class VLMQueryStatus(str, Enum):
    """Status of VLM query execution."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    VLM_UNCERTAIN = "vlm_uncertain"  # VLM returned low confidence or ambiguous response


class UserAction(str, Enum):
    """User action taken on VLM suggestion."""

    ACCEPTED = "accepted"  # User accepted VLM suggestion
    REJECTED = "rejected"  # User rejected VLM suggestion
    MODIFIED = "modified"  # User modified VLM suggestion with manual input


class VLMQuery(BaseModel):
    """Represents a query sent to vision-language model for semantic labeling.

    Key Clarifications (2025-12-19):
    - VLM_UNCERTAIN status: Triggered when VLM returns low confidence or ambiguous response
    - Manual fallback: When VLM_UNCERTAIN, user provides manual text input via user_modified_label
    - Semantic label replacement: Confirmed labels replace generic IDs (object_N → class name)

    Fields:
        id: Unique identifier for this query
        region_id: Foreign key to UncertainRegion
        image_path: Path to image sent to VLM
        prompt: Prompt text sent to VLM
        model_name: VLM model used (e.g., "gpt-5.2")
        response: VLM response with label, confidence, reasoning, raw_response
        token_count: Tokens used in query (0 if failed)
        cost: Estimated cost in USD
        latency: Query latency in seconds
        status: Query execution status
        error_message: Error details if failed
        user_action: User's action on VLM suggestion (accepted/rejected/modified)
        user_modified_label: Manual label if user modified VLM suggestion
        queried_at: When query was sent
        responded_at: When response received (None if pending/failed)
    """

    id: UUID
    region_id: UUID
    image_path: Path
    prompt: str = Field(min_length=1)  # Must not be empty
    model_name: str = Field(default="gpt-4o")  # OpenAI vision model (gpt-4o, gpt-4o-mini, gpt-4-turbo)
    response: Dict[str, Any] = Field(default_factory=dict)  # {label, confidence, reasoning, raw_response}
    token_count: int = Field(ge=0)  # Must be >= 0
    cost: float = Field(ge=0.0)  # Must be >= 0
    latency: float = Field(default=0.0, ge=0.0)
    status: VLMQueryStatus
    error_message: Optional[str] = None
    user_action: Optional[UserAction] = None
    user_modified_label: Optional[str] = None  # Manual label when VLM_UNCERTAIN or user modifies
    queried_at: datetime
    responded_at: Optional[datetime] = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, v: str) -> str:
        """Validate that prompt is not empty."""
        if not v or not v.strip():
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("token_count")
    @classmethod
    def validate_token_count_for_success(cls, v: int, values) -> int:
        """Validate token count is positive for successful queries."""
        # Note: Pydantic v2 uses values parameter differently
        # For failed queries, token_count can be 0
        return v

    @field_validator("cost")
    @classmethod
    def validate_cost_non_negative(cls, v: float) -> float:
        """Validate that cost is non-negative."""
        if v < 0:
            raise ValueError("cost must be non-negative")
        return v

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            UUID: str,
            Path: str,
            datetime: lambda v: v.isoformat(),
        }

    def get_final_label(self) -> Optional[str]:
        """Get the final label (user-modified if available, otherwise VLM label).

        Returns:
            Final semantic label to use for this region
        """
        if self.user_modified_label:
            return self.user_modified_label

        if self.response and "label" in self.response:
            return self.response["label"]

        return None

    def is_vlm_uncertain(self) -> bool:
        """Check if VLM was uncertain about this region.

        Returns:
            True if status is VLM_UNCERTAIN
        """
        return self.status == VLMQueryStatus.VLM_UNCERTAIN

    def requires_manual_input(self) -> bool:
        """Check if this query requires manual user input.

        Returns:
            True if VLM_UNCERTAIN and no manual label provided yet
        """
        return self.is_vlm_uncertain() and self.user_modified_label is None

    def get_confidence(self) -> float:
        """Get VLM confidence score if available.

        Returns:
            VLM confidence score (0.0-1.0) or 0.0 if not available
        """
        if self.response and "confidence" in self.response:
            return float(self.response["confidence"])
        return 0.0
