"""Independent VLM (Vision-Language Model) client module.

This module provides a standalone, testable interface for interacting with
Vision-Language Model APIs (OpenAI, etc.) with built-in error handling,
rate limiting, and cost tracking.

Public API:
    VLMClient: Main client for making VLM queries
    VLMConfig: Configuration for VLM client
    VLMRequest: Request model for VLM queries
    VLMResponse: Response model from VLM queries
    VLMException: Base exception for VLM errors
"""

from src.vlm.client import VLMClient, VLMConfig
from src.vlm.models import VLMRequest, VLMResponse, VLMBatchRequest, VLMBatchResponse, VLMStatus
from src.vlm.exceptions import (
    VLMException,
    VLMAuthError,
    VLMRateLimitError,
    VLMResponseError,
    VLMNetworkError,
)
from src.vlm.rate_limiter import RateLimiter, RateLimiterConfig

__all__ = [
    "VLMClient",
    "VLMConfig",
    "VLMRequest",
    "VLMResponse",
    "VLMBatchRequest",
    "VLMBatchResponse",
    "VLMStatus",
    "VLMException",
    "VLMAuthError",
    "VLMRateLimitError",
    "VLMResponseError",
    "VLMNetworkError",
    "RateLimiter",
    "RateLimiterConfig",
]

__version__ = "1.0.0"
