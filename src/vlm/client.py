"""VLM Client for interacting with Vision-Language Model APIs.

This module provides the main client interface for making VLM queries
with built-in error handling, retry logic, and cost tracking.
"""

import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from openai import OpenAI, APIError, RateLimitError as OpenAIRateLimitError
import json

from src.vlm.models import (
    VLMRequest,
    VLMResponse,
    VLMBatchRequest,
    VLMBatchResponse,
    VLMStatus,
)
from src.vlm.exceptions import (
    VLMException,
    VLMAuthError,
    VLMRateLimitError,
    VLMResponseError,
    VLMNetworkError,
)
from src.vlm.utils import encode_image_base64
from src.vlm.rate_limiter import RateLimiter, RateLimiterConfig


class VLMConfig(BaseModel):
    """Configuration for VLM client.

    Attributes:
        api_key: OpenAI API key
        model_name: Model to use (e.g., "gpt-4o", "gpt-4o-mini")
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_base_delay: Base delay for exponential backoff (seconds)
        retry_max_delay: Maximum delay between retries (seconds)
        confidence_threshold: Threshold for VLM_UNCERTAIN status
        rate_limiter_config: Optional rate limiter configuration
    """

    api_key: str = Field(min_length=1)
    model_name: str = Field(default="gpt-4o")
    timeout: int = Field(default=30, ge=1, le=300)
    max_retries: int = Field(default=5, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    retry_max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    rate_limiter_config: Optional[RateLimiterConfig] = Field(default=None)

    class Config:
        """Pydantic config."""
        frozen = True  # Make config immutable


class VLMClient:
    """Client for making VLM API queries.

    This class provides methods for single and batch VLM queries with
    automatic error handling, retry logic, and response parsing.

    Example:
        >>> config = VLMConfig(api_key="sk-...")
        >>> client = VLMClient(config)
        >>> request = VLMRequest(image_path=Path("image.jpg"), prompt="What is this?")
        >>> response = client.query_region(request)
        >>> print(response.label)
    """

    def __init__(self, config: VLMConfig):
        """Initialize VLM client.

        Args:
            config: Client configuration

        Raises:
            VLMAuthError: If API key is invalid
        """
        self.config = config
        self._client = OpenAI(api_key=config.api_key, timeout=config.timeout)

        # Initialize rate limiter if configured
        if config.rate_limiter_config:
            self._rate_limiter: Optional[RateLimiter] = RateLimiter(config.rate_limiter_config)
        else:
            self._rate_limiter = None

        # Validate API key by making a test call (optional, can be expensive)
        # For now, we'll validate on first use

    def query_region(self, request: VLMRequest) -> VLMResponse:
        """Query VLM for a single region.

        Args:
            request: VLM request with image and prompt

        Returns:
            VLM response with label, confidence, and metadata

        Raises:
            VLMAuthError: If authentication fails
            VLMRateLimitError: If rate limit is exceeded
            VLMResponseError: If response is malformed
            VLMNetworkError: If network error occurs
        """
        start_time = time.time()
        queried_at = datetime.now()

        try:
            # Encode image to base64
            image_base64 = encode_image_base64(request.image_path)

            # Create messages for VLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a computer vision expert. Analyze the image and provide a semantic label for the object shown. Respond in JSON format with: {\"label\": \"object_class\", \"confidence\": 0.0-1.0, \"reasoning\": \"explanation\"}",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ]

            # Call OpenAI API with retry logic
            response = self._call_with_retry(messages, request)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            parsed = self._parse_response(response)

            # Determine status based on confidence
            status = self._evaluate_status(parsed.get("confidence", 0.0))

            # Calculate cost (simplified, real pricing depends on model)
            cost = self._calculate_cost(
                response.usage.prompt_tokens if hasattr(response, "usage") else 0,
                response.usage.completion_tokens if hasattr(response, "usage") else 0,
                request.model
            )

            return VLMResponse(
                request_id=request.region_id,
                status=status,
                label=parsed.get("label"),
                confidence=parsed.get("confidence"),
                reasoning=parsed.get("reasoning"),
                raw_response=parsed,
                cost=cost,
                tokens_used=response.usage.total_tokens if hasattr(response, "usage") else 0,
                latency_ms=latency_ms,
                queried_at=queried_at,
                responded_at=datetime.now(),
            )

        except Exception as e:
            # Classify and raise appropriate VLM exception
            self._classify_and_raise_error(e)

    def query_batch(
        self,
        batch_request: VLMBatchRequest,
        progress_callback=None
    ) -> VLMBatchResponse:
        """Query VLM for multiple regions in parallel.

        Args:
            batch_request: Batch request with list of requests
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            Batch response with all individual responses

        Raises:
            VLMException: If batch processing fails
        """
        started_at = datetime.now()
        responses = []
        successful = 0
        failed = 0
        uncertain = 0
        total_cost = 0.0
        total_tokens = 0

        with ThreadPoolExecutor(max_workers=batch_request.max_workers) as executor:
            # Submit all requests
            future_to_request = {
                executor.submit(self.query_region, req): req
                for req in batch_request.requests
            }

            # Process completed requests
            for future in as_completed(future_to_request):
                try:
                    response = future.result()
                    responses.append(response)

                    # Update counters
                    if response.status == VLMStatus.SUCCESS:
                        successful += 1
                    elif response.status == VLMStatus.VLM_UNCERTAIN:
                        uncertain += 1
                    else:
                        failed += 1

                    total_cost += response.cost
                    total_tokens += response.tokens_used

                except Exception as e:
                    # Create failed response
                    request = future_to_request[future]
                    failed_response = VLMResponse(
                        request_id=request.region_id,
                        status=VLMStatus.FAILED,
                        error_message=str(e),
                        queried_at=started_at,
                    )
                    responses.append(failed_response)
                    failed += 1

                    # Stop if fail_fast is enabled
                    if batch_request.fail_fast:
                        break

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(len(responses), len(batch_request.requests))

        total_latency_ms = (datetime.now() - started_at).total_seconds() * 1000

        return VLMBatchResponse(
            responses=responses,
            total_requests=len(batch_request.requests),
            successful=successful,
            failed=failed,
            uncertain=uncertain,
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            started_at=started_at,
            completed_at=datetime.now(),
        )

    def _call_with_retry(self, messages: List[dict], request: VLMRequest):
        """Call OpenAI API with exponential backoff retry.

        Args:
            messages: Messages to send to API
            request: Original VLM request

        Returns:
            OpenAI API response

        Raises:
            VLMRateLimitError: If max retries exceeded
            VLMException: For other errors
        """
        for attempt in range(self.config.max_retries + 1):
            try:
                # Apply rate limiting before API call
                if self._rate_limiter:
                    self._rate_limiter.acquire()

                response = self._client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                return response

            except OpenAIRateLimitError as e:
                if attempt == self.config.max_retries:
                    raise

                # Calculate delay with exponential backoff
                delay = min(
                    self.config.retry_base_delay * (2 ** attempt),
                    self.config.retry_max_delay
                )

                # Add jitter (±25%)
                import random
                jitter = delay * random.uniform(-0.25, 0.25)
                delay += jitter

                time.sleep(delay)

            except Exception as e:
                # Don't retry on other errors
                raise

    def _parse_response(self, response) -> dict:
        """Parse VLM API response.

        Args:
            response: OpenAI API response

        Returns:
            Parsed response dictionary

        Raises:
            VLMResponseError: If response is malformed
        """
        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)

            # Validate required fields
            if "label" not in parsed:
                raise VLMResponseError("Response missing 'label' field", content)

            return parsed

        except json.JSONDecodeError as e:
            raise VLMResponseError(f"Invalid JSON response: {str(e)}", content, e)
        except (AttributeError, IndexError, KeyError) as e:
            raise VLMResponseError(f"Malformed response structure: {str(e)}", None, e)

    def _evaluate_status(self, confidence: float) -> VLMStatus:
        """Evaluate VLM status based on confidence.

        Args:
            confidence: Confidence score (0.0-1.0)

        Returns:
            VLM status (SUCCESS or VLM_UNCERTAIN)
        """
        if confidence >= self.config.confidence_threshold:
            return VLMStatus.SUCCESS
        return VLMStatus.VLM_UNCERTAIN

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate API cost based on token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name

        Returns:
            Estimated cost in USD
        """
        # Pricing per 1K tokens (as of 2025)
        pricing = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        }

        # Default to gpt-4o if model not found
        model_pricing = pricing.get(model, pricing["gpt-4o"])

        input_cost = (input_tokens / 1000.0) * model_pricing["input"]
        output_cost = (output_tokens / 1000.0) * model_pricing["output"]

        return input_cost + output_cost

    def _extract_retry_after(self, error: Exception) -> Optional[int]:
        """Extract retry-after value from rate limit error.

        Args:
            error: Rate limit error

        Returns:
            Retry-after seconds, or None if not available
        """
        try:
            if hasattr(error, "response") and error.response:
                return int(error.response.headers.get("retry-after", 0))
        except (AttributeError, ValueError):
            pass
        return None

    def _classify_and_raise_error(self, error: Exception):
        """Classify exception and raise appropriate VLM exception.

        Maps OpenAI SDK exceptions to VLM-specific exceptions for
        better error categorization and handling.

        Args:
            error: Original exception from OpenAI SDK or parsing

        Raises:
            VLMRateLimitError: If rate limit exceeded
            VLMAuthError: If authentication failed
            VLMResponseError: If response is malformed (re-raised)
            VLMNetworkError: If network/API error occurred
            VLMException: For unexpected errors
        """
        # Re-raise VLM-specific errors without wrapping
        if isinstance(error, VLMResponseError):
            raise

        # Rate limit errors
        if isinstance(error, OpenAIRateLimitError):
            raise VLMRateLimitError(
                f"Rate limit exceeded: {str(error)}",
                retry_after=self._extract_retry_after(error),
                original_error=error
            )

        # Authentication errors
        if isinstance(error, APIError):
            error_msg = str(error).lower()
            if "authentication" in error_msg or "api key" in error_msg:
                raise VLMAuthError(f"Authentication failed: {str(error)}", error)
            # Other API errors are network-related
            raise VLMNetworkError(f"API error: {str(error)}", error)

        # Unexpected errors
        raise VLMException(f"Unexpected error: {str(error)}", error)
