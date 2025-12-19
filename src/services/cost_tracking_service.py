"""Cost Tracking Service for VLM API cost calculation and budget enforcement.

TDD Green Phase: T033 [US2] - Implement CostTrackingService

This service provides:
- Real-time VLM API cost calculation based on token usage
- Budget limit monitoring and enforcement (95% threshold)
- Cost estimation for remaining regions
- Cost accumulation tracking across queries
"""

from typing import Optional

from src.models.semantic_labeling_job import SemanticLabelingJob


class CostTrackingService:
    """Service for tracking VLM API costs and enforcing budget limits."""

    # GPT-5.2 pricing (aligned with VLMService)
    PRICING = {
        "gpt-5.2": {
            "input": 0.01,  # $0.01 per 1K input tokens
            "output": 0.03,  # $0.03 per 1K output tokens
        },
        "gpt-4-vision": {
            "input": 0.01,
            "output": 0.03,
        },
        "claude-3": {
            "input": 0.01,
            "output": 0.03,
        },
    }

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """Calculate API cost based on token usage.

        TDD: T033 [US2] - Implement cost calculation

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name for pricing lookup

        Returns:
            Estimated cost in USD
        """
        # Default to gpt-5.2 if model not found
        if model not in self.PRICING:
            model = "gpt-5.2"

        pricing = self.PRICING[model]

        # Cost = (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]

        return input_cost + output_cost

    def update_job_cost(
        self,
        job: SemanticLabelingJob,
        query_cost: float,
        tokens: int,
        success: bool = True,
    ) -> None:
        """Update job with query cost and token usage.

        TDD: T033 [US2] - Implement cost accumulation

        Args:
            job: SemanticLabelingJob to update
            query_cost: Cost of the query in USD
            tokens: Total tokens used in query
            success: Whether query was successful (default True)
        """
        # Accumulate cost and tokens
        job.cost_tracking.total_cost += query_cost
        job.cost_tracking.total_tokens += tokens

        # Update query counts
        if success:
            job.cost_tracking.queries_successful += 1
        else:
            job.cost_tracking.queries_failed += 1

        # Update calculated fields
        job.update_budget_consumed_percentage()
        job.update_average_cost_per_region()

    def check_budget_limit(self, job: SemanticLabelingJob) -> bool:
        """Check if job is within budget limit.

        TDD: T033 [US2] - Implement budget enforcement (95% threshold)

        Args:
            job: SemanticLabelingJob to check

        Returns:
            True if within budget (<95%), False if at/over 95%
        """
        # If no budget limit set, always within budget
        if job.cost_tracking.budget_limit is None:
            job.cost_tracking.budget_consumed_percentage = 0.0
            return True

        # Calculate budget consumed percentage
        budget_consumed = (
            job.cost_tracking.total_cost / job.cost_tracking.budget_limit
        ) * 100.0

        # Update job with current percentage
        job.cost_tracking.budget_consumed_percentage = budget_consumed

        # Return False if at/over 95% threshold
        return budget_consumed < 95.0

    def estimate_remaining_cost(self, job: SemanticLabelingJob) -> Optional[float]:
        """Estimate remaining cost based on average cost per region.

        TDD: T033 [US2] - Implement remaining cost estimation

        Args:
            job: SemanticLabelingJob to estimate

        Returns:
            Estimated remaining cost in USD, or None if no data yet
        """
        # Cannot estimate if no regions completed yet
        if job.cost_tracking.queries_successful == 0:
            return None

        # Calculate average cost per region
        avg_cost_per_region = (
            job.cost_tracking.total_cost / job.cost_tracking.queries_successful
        )

        # Estimate remaining cost
        remaining_cost = avg_cost_per_region * job.progress.regions_pending

        # Update job with estimate
        job.cost_tracking.estimated_remaining_cost = remaining_cost

        return remaining_cost
