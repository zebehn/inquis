"""VLM Service for querying vision-language models for semantic labeling.

TDD: T081-T086 [US3] - Implement VLMService with OpenAI API integration

Key Features:
- Query OpenAI vision models with cropped region images
- Parse and validate VLM responses
- Detect VLM_UNCERTAIN responses based on confidence and ambiguous language
- Support manual label fallback workflow
- Calculate API costs and track usage metrics

Integration Note:
- This service now uses the new independent VLM module (src/vlm/)
- Maintains backward compatibility with existing VLMQuery/VLMQueryStatus models
- Delegates core VLM logic to VLMClient for better separation of concerns
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
import re

from pydantic import ValidationError

from src.models.vlm_query import VLMQuery, VLMQueryStatus, UserAction
from src.models.uncertain_region import UncertainRegion
from src.vlm import (
    VLMClient,
    VLMConfig,
    VLMRequest,
    VLMResponse,
    VLMStatus,
    VLMException,
    VLMRateLimitError,
    RateLimiterConfig,
)


class VLMService:
    """Service for querying vision-language models to generate semantic labels."""

    # Confidence threshold for VLM_UNCERTAIN detection
    CONFIDENCE_THRESHOLD = 0.5

    # Keywords indicating uncertainty in VLM reasoning
    UNCERTAINTY_KEYWORDS = [
        "not sure",
        "uncertain",
        "unclear",
        "ambiguous",
        "difficult to",
        "hard to",
        "cannot identify",
        "cannot determine",
        "could be",
        "might be",
        "maybe",
        "possibly",
        "perhaps",
    ]

    # OpenAI Vision Model Pricing (as of 2025)
    PRICING = {
        "gpt-4o": {
            "input": 0.0025,  # $2.50 per 1M tokens = $0.0025 per 1K tokens
            "output": 0.01,  # $10.00 per 1M tokens = $0.01 per 1K tokens
        },
        "gpt-4o-mini": {
            "input": 0.00015,  # $0.15 per 1M tokens = $0.00015 per 1K tokens
            "output": 0.0006,  # $0.60 per 1M tokens = $0.0006 per 1K tokens
        },
        "gpt-4-turbo": {
            "input": 0.01,  # $10.00 per 1M tokens = $0.01 per 1K tokens
            "output": 0.03,  # $30.00 per 1M tokens = $0.03 per 1K tokens
        }
    }

    def __init__(
        self,
        api_key: str,
        confidence_threshold: float = 0.5,
        enable_rate_limiting: bool = False,
        requests_per_second: float = 10.0,
    ):
        """Initialize VLM service with VLM client.

        Args:
            api_key: OpenAI API key
            confidence_threshold: Confidence threshold for VLM_UNCERTAIN (default 0.5)
            enable_rate_limiting: Enable proactive rate limiting (default False)
            requests_per_second: Max requests per second if rate limiting enabled (default 10.0)
        """
        self.confidence_threshold = confidence_threshold

        # Configure rate limiter if enabled
        rate_limiter_config = None
        if enable_rate_limiting:
            rate_limiter_config = RateLimiterConfig(
                requests_per_second=requests_per_second,
                burst_capacity=int(requests_per_second * 2),
                enabled=True
            )

        # Initialize VLM client with new module
        config = VLMConfig(
            api_key=api_key,
            confidence_threshold=confidence_threshold,
            rate_limiter_config=rate_limiter_config
        )
        self.client = VLMClient(config)

    def query_region(
        self,
        region: UncertainRegion,
        image_path: Path,
        prompt: str,
        model: str = "gpt-4o",
    ) -> VLMQuery:
        """Query VLM with region image to generate semantic label.

        Args:
            region: UncertainRegion to label
            image_path: Path to cropped region image
            prompt: Prompt text for VLM
            model: VLM model name (default "gpt-4o")

        Returns:
            VLMQuery with response and metadata
        """
        query_id = uuid4()
        queried_at = datetime.now()

        try:
            # Create VLM request
            vlm_request = VLMRequest(
                region_id=region.id,
                image_path=image_path,
                prompt=prompt,
                model=model,
                max_tokens=500,
                temperature=0.2,
            )

            # Query VLM client (new module)
            vlm_response: VLMResponse = self.client.query_region(vlm_request)

            # Convert VLMResponse to VLMQuery for backward compatibility
            return self._convert_response_to_query(
                query_id=query_id,
                region=region,
                image_path=image_path,
                prompt=prompt,
                model=model,
                vlm_response=vlm_response,
                queried_at=queried_at,
            )

        except VLMRateLimitError as e:
            # Handle rate limiting
            return VLMQuery(
                id=query_id,
                region_id=region.id,
                image_path=image_path,
                prompt=prompt,
                model_name=model,
                response={},
                token_count=0,
                cost=0.0,
                latency=0.0,
                status=VLMQueryStatus.RATE_LIMITED,
                error_message=f"Rate limit exceeded: {str(e)}",
                queried_at=queried_at,
            )

        except VLMException as e:
            # Handle other VLM failures
            return VLMQuery(
                id=query_id,
                region_id=region.id,
                image_path=image_path,
                prompt=prompt,
                model_name=model,
                response={},
                token_count=0,
                cost=0.0,
                latency=0.0,
                status=VLMQueryStatus.FAILED,
                error_message=str(e),
                queried_at=queried_at,
            )

    def _convert_response_to_query(
        self,
        query_id,
        region: UncertainRegion,
        image_path: Path,
        prompt: str,
        model: str,
        vlm_response: VLMResponse,
        queried_at: datetime,
    ) -> VLMQuery:
        """Convert VLMResponse from new module to VLMQuery for backward compatibility.

        Args:
            query_id: Query UUID
            region: UncertainRegion
            image_path: Image path
            prompt: Prompt text
            model: Model name
            vlm_response: Response from VLM client
            queried_at: Query timestamp

        Returns:
            VLMQuery for legacy compatibility
        """
        # Map VLMStatus to VLMQueryStatus
        status_map = {
            VLMStatus.SUCCESS: VLMQueryStatus.SUCCESS,
            VLMStatus.VLM_UNCERTAIN: VLMQueryStatus.VLM_UNCERTAIN,
            VLMStatus.FAILED: VLMQueryStatus.FAILED,
            VLMStatus.RATE_LIMITED: VLMQueryStatus.RATE_LIMITED,
        }

        # Build response dict for legacy format
        response_dict = {
            "label": vlm_response.label or "unknown",
            "confidence": vlm_response.confidence or 0.0,
            "reasoning": vlm_response.reasoning or "",
            "raw_response": vlm_response.raw_response or {},
        }

        # Check for uncertainty keywords (legacy behavior)
        status = status_map.get(vlm_response.status, VLMQueryStatus.FAILED)
        if status == VLMQueryStatus.SUCCESS:
            # Double-check with uncertainty keywords
            status = self.evaluate_confidence(response_dict)

        return VLMQuery(
            id=query_id,
            region_id=region.id,
            image_path=image_path,
            prompt=prompt,
            model_name=model,
            response=response_dict,
            token_count=vlm_response.tokens_used,
            cost=vlm_response.cost,
            latency=vlm_response.latency_ms / 1000.0,  # Convert ms to seconds
            status=status,
            queried_at=queried_at,
            responded_at=vlm_response.responded_at or datetime.now(),
            error_message=vlm_response.error_message,
        )

    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse VLM response JSON.

        Note: This method is kept for backward compatibility but is no longer used internally.
        The new VLM module handles parsing internally.

        Args:
            raw_response: Raw JSON string from VLM

        Returns:
            Parsed response dictionary with label, confidence, reasoning, raw_response
        """
        try:
            # Try to parse as JSON
            parsed = json.loads(raw_response)

            # Extract fields with defaults
            return {
                "label": parsed.get("label", "unknown"),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", ""),
                "raw_response": raw_response,
            }

        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback for malformed JSON
            return {
                "label": "unknown",
                "confidence": 0.0,
                "reasoning": "Failed to parse VLM response",
                "raw_response": raw_response,
            }

    def evaluate_confidence(self, response: Dict[str, Any]) -> VLMQueryStatus:
        """Evaluate VLM response confidence to determine status.

        Checks both numeric confidence score and ambiguous language in reasoning.

        Args:
            response: Parsed VLM response dictionary

        Returns:
            VLMQueryStatus.SUCCESS or VLMQueryStatus.VLM_UNCERTAIN
        """
        confidence = response.get("confidence", 0.0)
        reasoning = response.get("reasoning", "").lower()

        # Check numeric confidence threshold
        if confidence < self.confidence_threshold:
            return VLMQueryStatus.VLM_UNCERTAIN

        # Check for uncertainty keywords in reasoning
        for keyword in self.UNCERTAINTY_KEYWORDS:
            if keyword in reasoning:
                return VLMQueryStatus.VLM_UNCERTAIN

        return VLMQueryStatus.SUCCESS

    def apply_manual_label(self, query: VLMQuery, manual_label: str) -> VLMQuery:
        """Apply manual label to VLM_UNCERTAIN query.

        Args:
            query: VLMQuery to update
            manual_label: User-provided semantic label

        Returns:
            Updated VLMQuery with manual label
        """
        query.user_action = UserAction.MODIFIED
        query.user_modified_label = manual_label
        return query

    def accept_suggestion(self, query: VLMQuery) -> VLMQuery:
        """Accept VLM suggestion.

        Args:
            query: VLMQuery to update

        Returns:
            Updated VLMQuery with accepted status
        """
        query.user_action = UserAction.ACCEPTED
        return query

    def reject_suggestion(self, query: VLMQuery, manual_label: str) -> VLMQuery:
        """Reject VLM suggestion and provide manual label.

        Args:
            query: VLMQuery to update
            manual_label: User-provided alternative label

        Returns:
            Updated VLMQuery with rejected status and manual label
        """
        query.user_action = UserAction.REJECTED
        query.user_modified_label = manual_label
        return query

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate API cost based on token usage.

        Note: This method is kept for backward compatibility but is no longer used internally.
        The new VLM module handles cost calculation internally.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name for pricing lookup

        Returns:
            Estimated cost in USD
        """
        if model not in self.PRICING:
            # Default pricing if model not found
            model = "gpt-4o"

        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]

        return input_cost + output_cost

    def encode_image_base64(self, image_path: Path) -> str:
        """Encode image to base64 for API transmission.

        Note: This method is kept for backward compatibility but is no longer used internally.
        The new VLM module handles image encoding internally.

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded image string
        """
        from src.vlm.utils import encode_image_base64 as vlm_encode
        return vlm_encode(image_path)

    # T011-T013: Batch VLM processing with parallel execution, rate limiting, and retry logic

    def query_regions_parallel(
        self,
        regions: List[Any],
        image_paths: List[Path],
        prompt: str,
        model: str = "gpt-4o",
        max_workers: int = 10,
    ) -> List[VLMQuery]:
        """Query multiple regions in parallel using VLM module's batch query.

        TDD: T011 - Extend VLMService with query_regions_parallel()

        Note: Now uses VLM module's optimized batch query with built-in retry logic.

        Args:
            regions: List of regions to query
            image_paths: List of cropped region image paths (parallel to regions)
            prompt: Prompt text for VLM
            model: VLM model name (default "gpt-4o")
            max_workers: Number of parallel workers (default 10)

        Returns:
            List of VLMQuery results (parallel to input regions)
        """
        from src.vlm import VLMBatchRequest

        # Validate inputs
        if len(regions) != len(image_paths):
            raise ValueError("regions and image_paths must have same length")

        # Create VLM requests
        vlm_requests = [
            VLMRequest(
                region_id=region.id,
                image_path=image_path,
                prompt=prompt,
                model=model,
                max_tokens=500,
                temperature=0.2,
            )
            for region, image_path in zip(regions, image_paths)
        ]

        # Create batch request
        batch_request = VLMBatchRequest(
            requests=vlm_requests,
            max_workers=max_workers,
            fail_fast=False  # Continue on errors
        )

        # Execute batch query
        batch_response = self.client.query_batch(batch_request)

        # Convert VLMResponse objects to VLMQuery objects
        vlm_queries = []
        queried_at = batch_response.started_at

        for i, (region, vlm_response) in enumerate(zip(regions, batch_response.responses)):
            vlm_query = self._convert_response_to_query(
                query_id=uuid4(),
                region=region,
                image_path=image_paths[i],
                prompt=prompt,
                model=model,
                vlm_response=vlm_response,
                queried_at=queried_at,
            )
            vlm_queries.append(vlm_query)

        return vlm_queries

    def _query_region_with_retry(
        self,
        region: Any,
        image_path: Path,
        prompt: str,
        model: str,
        max_retries: int = 5,
    ) -> VLMQuery:
        """Query region with exponential backoff retry logic.

        TDD: T012 - Add exponential backoff retry wrapper

        Note: This method is deprecated. The new VLM module handles retries internally.
        Kept for backward compatibility.

        Args:
            region: Region to query
            image_path: Path to cropped region image
            prompt: Prompt text for VLM
            model: VLM model name
            max_retries: Maximum retry attempts (default 5)

        Returns:
            VLMQuery result
        """
        # Delegate to query_region - VLM module handles retries internally
        return self.query_region(region, image_path, prompt, model)

    def query_regions_with_rate_limiting(
        self,
        regions: List[Any],
        image_paths: List[Path],
        prompt: str,
        model: str = "gpt-4o",
        max_workers: int = 10,
        rpm_limit: int = 450,
    ) -> List[VLMQuery]:
        """Query regions with proactive rate limiting.

        TDD: T013 - Add proactive rate limiting

        Note: This method now delegates to query_regions_parallel. Rate limiting is handled
        by the VLM module if enabled during VLMService initialization. The rpm_limit parameter
        is kept for backward compatibility but ignored if rate_limiter_config was provided
        during initialization.

        To enable rate limiting, initialize VLMService with:
            service = VLMService(api_key="...", enable_rate_limiting=True, requests_per_second=7.5)
            # 7.5 req/sec = 450 requests per minute

        Args:
            regions: List of regions to query
            image_paths: List of cropped region image paths
            prompt: Prompt text for VLM
            model: VLM model name (default "gpt-4o")
            max_workers: Number of parallel workers (default 10)
            rpm_limit: Requests per minute limit (default 450, kept for backward compatibility)

        Returns:
            List of VLMQuery results
        """
        # Delegate to query_regions_parallel
        # Rate limiting is handled by VLM module if configured
        return self.query_regions_parallel(
            regions=regions,
            image_paths=image_paths,
            prompt=prompt,
            model=model,
            max_workers=max_workers,
        )
