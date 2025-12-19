"""VLM Service for querying vision-language models for semantic labeling.

TDD: T081-T086 [US3] - Implement VLMService with OpenAI API integration

Key Features:
- Query OpenAI GPT-5.2 vision model with cropped region images
- Parse and validate VLM responses
- Detect VLM_UNCERTAIN responses based on confidence and ambiguous language
- Support manual label fallback workflow
- Calculate API costs and track usage metrics
"""

import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
import re

from openai import OpenAI, RateLimitError
from pydantic import ValidationError

from src.models.vlm_query import VLMQuery, VLMQueryStatus, UserAction
from src.models.uncertain_region import UncertainRegion


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

    # GPT-5.2 pricing (example rates - adjust based on actual API pricing)
    PRICING = {
        "gpt-5.2": {
            "input": 0.01,  # $0.01 per 1K input tokens
            "output": 0.03,  # $0.03 per 1K output tokens
        }
    }

    def __init__(self, api_key: str, confidence_threshold: float = 0.5):
        """Initialize VLM service with OpenAI API client.

        Args:
            api_key: OpenAI API key
            confidence_threshold: Confidence threshold for VLM_UNCERTAIN (default 0.5)
        """
        self.client = OpenAI(api_key=api_key)
        self.confidence_threshold = confidence_threshold

    def query_region(
        self,
        region: UncertainRegion,
        image_path: Path,
        prompt: str,
        model: str = "gpt-5.2",
    ) -> VLMQuery:
        """Query VLM with region image to generate semantic label.

        Args:
            region: UncertainRegion to label
            image_path: Path to cropped region image
            prompt: Prompt text for VLM
            model: VLM model name (default "gpt-5.2")

        Returns:
            VLMQuery with response and metadata
        """
        query_id = uuid4()
        queried_at = datetime.now()
        start_time = time.time()

        try:
            # Encode image to base64
            image_base64 = self.encode_image_base64(image_path)

            # Create messages for VLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a computer vision expert. Analyze the image and provide a semantic label for the object shown. Respond in JSON format with: {\"label\": \"object_class\", \"confidence\": 0.0-1.0, \"reasoning\": \"explanation\"}",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500,
                temperature=0.2,  # Lower temperature for more consistent outputs
            )

            # Calculate latency
            latency = time.time() - start_time

            # Parse response
            raw_response = response.choices[0].message.content
            parsed_response = self.parse_response(raw_response)

            # Evaluate confidence to determine status
            status = self.evaluate_confidence(parsed_response)

            # Calculate cost
            input_tokens = response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0
            output_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
            total_tokens = response.usage.total_tokens
            cost = self.calculate_cost(input_tokens, output_tokens, model)

            # Create VLMQuery
            vlm_query = VLMQuery(
                id=query_id,
                region_id=region.id,
                image_path=image_path,
                prompt=prompt,
                model_name=model,
                response=parsed_response,
                token_count=total_tokens,
                cost=cost,
                latency=latency,
                status=status,
                queried_at=queried_at,
                responded_at=datetime.now(),
            )

            return vlm_query

        except RateLimitError as e:
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
                latency=time.time() - start_time,
                status=VLMQueryStatus.RATE_LIMITED,
                error_message=f"Rate limit exceeded: {str(e)}",
                queried_at=queried_at,
            )

        except Exception as e:
            # Handle other API failures
            return VLMQuery(
                id=query_id,
                region_id=region.id,
                image_path=image_path,
                prompt=prompt,
                model_name=model,
                response={},
                token_count=0,
                cost=0.0,
                latency=time.time() - start_time,
                status=VLMQueryStatus.FAILED,
                error_message=str(e),
                queried_at=queried_at,
            )

    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse VLM response JSON.

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

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name for pricing lookup

        Returns:
            Estimated cost in USD
        """
        if model not in self.PRICING:
            # Default pricing if model not found
            model = "gpt-5.2"

        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]

        return input_cost + output_cost

    def encode_image_base64(self, image_path: Path) -> str:
        """Encode image to base64 for API transmission.

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # T011-T013: Batch VLM processing with parallel execution, rate limiting, and retry logic

    def query_regions_parallel(
        self,
        regions: List[Any],
        image_paths: List[Path],
        prompt: str,
        model: str = "gpt-5.2",
        max_workers: int = 10,
    ) -> List[VLMQuery]:
        """Query multiple regions in parallel using ThreadPoolExecutor.

        TDD: T011 - Extend VLMService with query_regions_parallel()

        Args:
            regions: List of regions to query
            image_paths: List of cropped region image paths (parallel to regions)
            prompt: Prompt text for VLM
            model: VLM model name (default "gpt-5.2")
            max_workers: Number of parallel workers (default 10)

        Returns:
            List of VLMQuery results (parallel to input regions)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Validate inputs
        if len(regions) != len(image_paths):
            raise ValueError("regions and image_paths must have same length")

        # Create tasks
        futures = {}
        results = [None] * len(regions)  # Preserve input order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            for idx, (region, image_path) in enumerate(zip(regions, image_paths)):
                future = executor.submit(
                    self._query_region_with_retry,
                    region,
                    image_path,
                    prompt,
                    model
                )
                futures[future] = idx

            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    vlm_query = future.result()
                    results[idx] = vlm_query
                except Exception as e:
                    # Create failed VLMQuery for exception
                    region = regions[idx]
                    results[idx] = VLMQuery(
                        id=uuid4(),
                        region_id=region.id,
                        image_path=image_paths[idx],
                        prompt=prompt,
                        model_name=model,
                        response={},
                        token_count=0,
                        cost=0.0,
                        latency=0.0,
                        status=VLMQueryStatus.FAILED,
                        error_message=f"Parallel execution error: {str(e)}",
                        queried_at=datetime.now(),
                    )

        return results

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

        Args:
            region: Region to query
            image_path: Path to cropped region image
            prompt: Prompt text for VLM
            model: VLM model name
            max_retries: Maximum retry attempts (default 5)

        Returns:
            VLMQuery result
        """
        base_delay = 1.0  # 1 second base delay
        max_delay = 60.0  # 60 second max delay
        jitter_factor = 0.25  # ±25% jitter

        for attempt in range(max_retries + 1):
            # Query region
            vlm_query = self.query_region(region, image_path, prompt, model)

            # Check if rate limited
            if vlm_query.status == VLMQueryStatus.RATE_LIMITED:
                if attempt < max_retries:
                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    # Add jitter (±25%)
                    jitter = delay * jitter_factor * (2 * (time.time() % 1) - 1)
                    delay_with_jitter = delay + jitter

                    # Wait before retry
                    time.sleep(delay_with_jitter)
                    continue
                else:
                    # Max retries exceeded
                    return vlm_query
            elif vlm_query.status == VLMQueryStatus.FAILED:
                if attempt < max_retries:
                    # Retry on failure (but with shorter backoff)
                    delay = min(base_delay * (1.5 ** attempt), max_delay / 2)
                    jitter = delay * jitter_factor * (2 * (time.time() % 1) - 1)
                    delay_with_jitter = delay + jitter
                    time.sleep(delay_with_jitter)
                    continue
                else:
                    return vlm_query
            else:
                # Success or VLM_UNCERTAIN - return immediately
                return vlm_query

        # Should not reach here, but return last query
        return vlm_query

    def query_regions_with_rate_limiting(
        self,
        regions: List[Any],
        image_paths: List[Path],
        prompt: str,
        model: str = "gpt-5.2",
        max_workers: int = 10,
        rpm_limit: int = 450,
    ) -> List[VLMQuery]:
        """Query regions with proactive rate limiting.

        TDD: T013 - Add proactive rate limiting

        Args:
            regions: List of regions to query
            image_paths: List of cropped region image paths
            prompt: Prompt text for VLM
            model: VLM model name (default "gpt-5.2")
            max_workers: Number of parallel workers (default 10)
            rpm_limit: Requests per minute limit (default 450 for Tier 1)

        Returns:
            List of VLMQuery results
        """
        # Calculate minimum delay between requests to stay under rpm_limit
        min_delay_seconds = 60.0 / rpm_limit  # e.g., 450 RPM = 0.133s between requests

        # Track last request time
        last_request_time = [0.0]  # Mutable list for closure access

        def rate_limited_query(region: Any, image_path: Path) -> VLMQuery:
            """Query with rate limiting."""
            # Wait if necessary to respect rate limit
            current_time = time.time()
            elapsed = current_time - last_request_time[0]
            if elapsed < min_delay_seconds:
                time.sleep(min_delay_seconds - elapsed)

            # Update last request time
            last_request_time[0] = time.time()

            # Execute query with retry
            return self._query_region_with_retry(region, image_path, prompt, model)

        # Use parallel execution with rate limiting
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Validate inputs
        if len(regions) != len(image_paths):
            raise ValueError("regions and image_paths must have same length")

        # Create tasks
        futures = {}
        results = [None] * len(regions)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            for idx, (region, image_path) in enumerate(zip(regions, image_paths)):
                future = executor.submit(rate_limited_query, region, image_path)
                futures[future] = idx

            # Collect results
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    vlm_query = future.result()
                    results[idx] = vlm_query
                except Exception as e:
                    # Create failed VLMQuery
                    region = regions[idx]
                    results[idx] = VLMQuery(
                        id=uuid4(),
                        region_id=region.id,
                        image_path=image_paths[idx],
                        prompt=prompt,
                        model_name=model,
                        response={},
                        token_count=0,
                        cost=0.0,
                        latency=0.0,
                        status=VLMQueryStatus.FAILED,
                        error_message=f"Rate limiting error: {str(e)}",
                        queried_at=datetime.now(),
                    )

        return results
