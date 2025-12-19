"""Unit tests for cost tracking and budget enforcement.

TDD Red Phase: T031-T032 [US2] - Write failing tests FIRST before implementation

These tests validate cost calculation and budget limit enforcement.
"""

import pytest
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from src.models.semantic_labeling_job import (
    SemanticLabelingJob,
    JobStatus,
    JobProgress,
    CostTracking,
    JobConfiguration,
    JobTimestamps,
)
from src.services.cost_tracking_service import CostTrackingService
from src.services.storage_service import StorageService


@pytest.fixture
def storage_service(tmp_path):
    """Create StorageService with temporary directory."""
    return StorageService(base_dir=tmp_path / "sessions")


@pytest.fixture
def cost_tracking_service():
    """Create CostTrackingService."""
    return CostTrackingService()


# T031 [P] [US2] - Test cost calculation


def test_calculate_cost_per_region():
    """Test cost calculation per region based on token usage.

    TDD: T031 [US2] - This test should FAIL until implementation is complete

    Validates:
    - Cost calculated correctly from input/output tokens
    - Pricing uses correct rates for model
    - Total cost accumulation
    """
    # ARRANGE
    cost_service = CostTrackingService()

    # Test data
    input_tokens = 1000
    output_tokens = 500
    model = "gpt-5.2"

    # ACT: Calculate cost
    cost = cost_service.calculate_cost(input_tokens, output_tokens, model)

    # ASSERT: Cost calculated correctly
    # Expected: (1000/1000 * $0.01) + (500/1000 * $0.03) = $0.01 + $0.015 = $0.025
    expected_cost = 0.025
    assert abs(cost - expected_cost) < 0.0001, \
        f"Expected cost ${expected_cost:.4f}, got ${cost:.4f}"

    # Test with different token counts
    cost2 = cost_service.calculate_cost(500, 250, model)
    expected_cost2 = 0.0125
    assert abs(cost2 - expected_cost2) < 0.0001, \
        f"Expected cost ${expected_cost2:.4f}, got ${cost2:.4f}"


def test_calculate_cost_with_different_models():
    """Test cost calculation uses correct pricing for different models.

    TDD: T031 [US2] - This test should FAIL until implementation is complete
    """
    # ARRANGE
    cost_service = CostTrackingService()

    # ACT & ASSERT: Test multiple models
    models_to_test = ["gpt-5.2", "gpt-4-vision", "claude-3"]

    for model in models_to_test:
        cost = cost_service.calculate_cost(1000, 500, model)
        assert cost > 0, f"Cost for {model} should be positive"


def test_accumulate_costs_over_multiple_queries():
    """Test cost accumulation across multiple VLM queries.

    TDD: T031 [US2] - This test should FAIL until implementation is complete
    """
    # ARRANGE
    cost_service = CostTrackingService()
    job = SemanticLabelingJob(
        id=uuid4(),
        session_id=uuid4(),
        video_path=Path("/tmp/test.mp4"),
        status=JobStatus.RUNNING,
        progress=JobProgress(
            frames_total=10,
            frames_pending=10,
            regions_total=50,
            regions_pending=50,
        ),
        cost_tracking=CostTracking(),
        configuration=JobConfiguration(),
        timestamps=JobTimestamps(created_at=datetime.now()),
        last_checkpoint_at=datetime.now(),
    )

    # ACT: Accumulate costs from 10 queries
    for i in range(10):
        query_cost = cost_service.calculate_cost(
            input_tokens=1000 + i * 100,
            output_tokens=500 + i * 50,
            model="gpt-5.2"
        )
        cost_service.update_job_cost(job, query_cost, 1500 + i * 150)

    # ASSERT: Total cost accumulated
    assert job.cost_tracking.total_cost > 0, "Total cost should be accumulated"
    assert job.cost_tracking.total_tokens > 0, "Total tokens should be accumulated"
    assert job.cost_tracking.queries_successful == 10, "Should track 10 successful queries"


# T032 [P] [US2] - Test budget enforcement


def test_budget_limit_enforcement():
    """Test budget limit checks and enforcement.

    TDD: T032 [US2] - This test should FAIL until implementation is complete

    Validates:
    - Budget limit enforcement at 95% threshold
    - check_budget_limit() returns True when within budget
    - check_budget_limit() returns False when at/over 95%
    - Budget consumed percentage calculated correctly
    """
    # ARRANGE
    cost_service = CostTrackingService()
    budget_limit = 10.0

    job = SemanticLabelingJob(
        id=uuid4(),
        session_id=uuid4(),
        video_path=Path("/tmp/test.mp4"),
        status=JobStatus.RUNNING,
        progress=JobProgress(
            frames_total=100,
            frames_pending=100,
            regions_total=500,
            regions_pending=500,
        ),
        cost_tracking=CostTracking(budget_limit=budget_limit),
        configuration=JobConfiguration(),
        timestamps=JobTimestamps(created_at=datetime.now()),
        last_checkpoint_at=datetime.now(),
    )

    # ACT & ASSERT: Test budget enforcement at different levels

    # 50% of budget - should continue
    job.cost_tracking.total_cost = 5.0
    within_budget = cost_service.check_budget_limit(job)
    assert within_budget is True, "Should be within budget at 50%"
    assert job.cost_tracking.budget_consumed_percentage == 50.0

    # 90% of budget - should continue
    job.cost_tracking.total_cost = 9.0
    within_budget = cost_service.check_budget_limit(job)
    assert within_budget is True, "Should be within budget at 90%"
    assert job.cost_tracking.budget_consumed_percentage == 90.0

    # 94% of budget - should continue
    job.cost_tracking.total_cost = 9.4
    within_budget = cost_service.check_budget_limit(job)
    assert within_budget is True, "Should be within budget at 94%"

    # 95% of budget - should pause
    job.cost_tracking.total_cost = 9.5
    within_budget = cost_service.check_budget_limit(job)
    assert within_budget is False, "Should exceed budget at 95%"
    assert job.cost_tracking.budget_consumed_percentage == 95.0

    # 100% of budget - should pause
    job.cost_tracking.total_cost = 10.0
    within_budget = cost_service.check_budget_limit(job)
    assert within_budget is False, "Should exceed budget at 100%"

    # Over budget - should pause
    job.cost_tracking.total_cost = 10.5
    within_budget = cost_service.check_budget_limit(job)
    assert within_budget is False, "Should exceed budget over 100%"


def test_budget_limit_with_no_limit_set():
    """Test budget enforcement when no budget limit is set.

    TDD: T032 [US2] - This test should FAIL until implementation is complete
    """
    # ARRANGE
    cost_service = CostTrackingService()

    job = SemanticLabelingJob(
        id=uuid4(),
        session_id=uuid4(),
        video_path=Path("/tmp/test.mp4"),
        status=JobStatus.RUNNING,
        progress=JobProgress(
            frames_total=100,
            frames_pending=100,
            regions_total=500,
            regions_pending=500,
        ),
        cost_tracking=CostTracking(budget_limit=None),  # No budget limit
        configuration=JobConfiguration(),
        timestamps=JobTimestamps(created_at=datetime.now()),
        last_checkpoint_at=datetime.now(),
    )

    # ACT: Check budget with no limit
    job.cost_tracking.total_cost = 1000.0  # Any amount
    within_budget = cost_service.check_budget_limit(job)

    # ASSERT: Always within budget when no limit set
    assert within_budget is True, "Should always be within budget when no limit set"
    assert job.cost_tracking.budget_consumed_percentage == 0.0, \
        "Budget consumed should be 0% when no limit"


def test_estimated_remaining_cost_calculation():
    """Test estimation of remaining cost based on average cost per region.

    TDD: T032 [US2] - This test should FAIL until implementation is complete
    """
    # ARRANGE
    cost_service = CostTrackingService()

    job = SemanticLabelingJob(
        id=uuid4(),
        session_id=uuid4(),
        video_path=Path("/tmp/test.mp4"),
        status=JobStatus.RUNNING,
        progress=JobProgress(
            frames_total=100,
            frames_processed=40,
            frames_pending=60,
            regions_total=500,
            regions_completed=200,
            regions_pending=300,
        ),
        cost_tracking=CostTracking(
            total_cost=5.0,
            queries_successful=200,
        ),
        configuration=JobConfiguration(),
        timestamps=JobTimestamps(created_at=datetime.now()),
        last_checkpoint_at=datetime.now(),
    )

    # ACT: Estimate remaining cost
    remaining_cost = cost_service.estimate_remaining_cost(job)

    # ASSERT: Remaining cost estimated correctly
    # Average cost per region: $5.0 / 200 = $0.025
    # Remaining regions: 300
    # Expected remaining cost: $0.025 * 300 = $7.50
    expected_remaining = 7.50
    assert remaining_cost is not None, "Should estimate remaining cost"
    assert abs(remaining_cost - expected_remaining) < 0.01, \
        f"Expected ${expected_remaining:.2f}, got ${remaining_cost:.2f}"


def test_cost_tracking_with_failed_queries():
    """Test cost tracking correctly handles failed queries.

    TDD: T032 [US2] - This test should FAIL until implementation is complete
    """
    # ARRANGE
    cost_service = CostTrackingService()

    job = SemanticLabelingJob(
        id=uuid4(),
        session_id=uuid4(),
        video_path=Path("/tmp/test.mp4"),
        status=JobStatus.RUNNING,
        progress=JobProgress(
            frames_total=10,
            frames_pending=10,
            regions_total=50,
            regions_pending=50,
        ),
        cost_tracking=CostTracking(),
        configuration=JobConfiguration(),
        timestamps=JobTimestamps(created_at=datetime.now()),
        last_checkpoint_at=datetime.now(),
    )

    # ACT: Track successful and failed queries
    # 8 successful queries
    for i in range(8):
        query_cost = cost_service.calculate_cost(1000, 500, "gpt-5.2")
        cost_service.update_job_cost(job, query_cost, 1500, success=True)

    # 2 failed queries (no cost)
    for i in range(2):
        cost_service.update_job_cost(job, 0.0, 0, success=False)

    # ASSERT: Failed queries tracked separately
    assert job.cost_tracking.queries_successful == 8, "Should track 8 successful queries"
    assert job.cost_tracking.queries_failed == 2, "Should track 2 failed queries"
    assert job.cost_tracking.total_cost > 0, "Should have cost from successful queries"
    assert job.cost_tracking.average_cost_per_region > 0, "Should calculate average from successful only"
