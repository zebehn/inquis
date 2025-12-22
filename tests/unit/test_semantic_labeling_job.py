"""Unit tests for SemanticLabelingJob model and service.

TDD Red Phase: T017-T018 [US1] - Write failing tests FIRST before implementation

These tests validate job creation and region queue building logic.
"""

import pytest
from uuid import uuid4, UUID
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
from src.services.semantic_labeling_service import SemanticLabelingService
from src.services.storage_service import StorageService
from src.services.vlm_service import VLMService


@pytest.fixture
def storage_service(tmp_path):
    """Create StorageService with temporary directory."""
    return StorageService(base_dir=tmp_path / "sessions")


@pytest.fixture
def vlm_service():
    """Create VLMService with test API key and rate limiting."""
    import os
    api_key = os.getenv("OPENAI_API_KEY", "test-api-key")
    return VLMService(
        api_key=api_key,
        confidence_threshold=0.5,
        enable_rate_limiting=True,
        requests_per_second=7.5,
    )


@pytest.fixture
def semantic_labeling_service(storage_service, vlm_service):
    """Create SemanticLabelingService."""
    return SemanticLabelingService(
        storage_service=storage_service,
        vlm_service=vlm_service
    )


@pytest.fixture
def test_session(storage_service):
    """Create test video session."""
    session_id = str(uuid4())
    storage_service.create_session(session_id)

    from src.models.video_session import VideoSession, SessionStatus
    session = VideoSession(
        id=session_id,
        filename="test_video.mp4",
        filepath=Path("/tmp/test_video.mp4"),
        upload_timestamp=datetime.now(),
        status=SessionStatus.COMPLETED,
        metadata={
            "duration": 10.0,
            "fps": 30.0,
            "frame_count": 100,
            "resolution": "1920x1080",
        },
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    storage_service.save_video_session(session)
    return session


# T017 [P] [US1] - Test job creation with valid parameters


def test_create_job_with_valid_params(
    semantic_labeling_service, test_session
):
    """Test creating a semantic labeling job with valid parameters.

    TDD: T017 [US1] - This test should FAIL until implementation is complete

    Validates:
    - Job is created in PENDING status
    - Job configuration is set correctly
    - Session validation passes
    - Job is persisted to storage
    """
    # ARRANGE
    video_path = Path(test_session.filepath)
    budget_limit = 10.0
    frame_sampling = 1
    confidence_threshold = 0.5
    enable_tracking = True
    tracking_iou_threshold = 0.7

    # ACT: Create job
    job = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=video_path,
        budget_limit=budget_limit,
        frame_sampling=frame_sampling,
        confidence_threshold=confidence_threshold,
        enable_tracking=enable_tracking,
        tracking_iou_threshold=tracking_iou_threshold,
    )

    # ASSERT: Job created with correct properties
    assert isinstance(job, SemanticLabelingJob), "Should return SemanticLabelingJob instance"
    assert isinstance(job.id, UUID), "Job should have UUID"
    assert job.session_id == test_session.id, "Job should be associated with session"
    assert job.video_path == video_path, "Job should have video path"
    assert job.status == JobStatus.PENDING, "New job should be in PENDING status"

    # Verify configuration
    assert job.cost_tracking.budget_limit == budget_limit, "Budget limit should match"
    assert job.configuration.frame_sampling == frame_sampling, "Frame sampling should match"
    assert job.configuration.confidence_threshold == confidence_threshold, "Confidence threshold should match"
    assert job.configuration.enable_tracking == enable_tracking, "Tracking should be enabled"
    assert job.configuration.tracking_iou_threshold == tracking_iou_threshold, "Tracking IoU threshold should match"

    # Verify initial progress
    assert job.progress.frames_total >= 0, "Should have frames_total calculated"
    assert job.progress.frames_processed == 0, "Should have 0 frames processed initially"
    assert job.progress.regions_completed == 0, "Should have 0 regions completed initially"
    assert job.progress.progress_percentage == 0.0, "Should have 0% progress initially"

    # Verify initial cost tracking
    assert job.cost_tracking.total_cost == 0.0, "Should have 0 cost initially"
    assert job.cost_tracking.total_tokens == 0, "Should have 0 tokens initially"
    assert job.cost_tracking.budget_limit == budget_limit, "Budget limit should be set"
    assert job.cost_tracking.queries_successful == 0, "Should have 0 successful queries initially"
    assert job.cost_tracking.queries_failed == 0, "Should have 0 failed queries initially"
    assert job.cost_tracking.queries_uncertain == 0, "Should have 0 uncertain queries initially"

    # Verify timestamps
    assert job.timestamps.created_at is not None, "Should have created_at timestamp"
    assert job.timestamps.started_at is None, "Should not have started_at initially"
    assert job.timestamps.completed_at is None, "Should not have completed_at initially"

    # Verify job can be loaded from storage
    loaded_job = semantic_labeling_service.get_job(job.id)
    assert loaded_job.id == job.id, "Loaded job should have same ID"
    assert loaded_job.status == JobStatus.PENDING, "Loaded job should have same status"


# T018 [P] [US1] - Test region queue building


def test_build_pending_regions_queue(
    semantic_labeling_service, test_session, storage_service
):
    """Test building pending regions queue for job.

    TDD: T018 [US1] - This test should FAIL until implementation is complete

    Validates:
    - All regions across sampled frames are enumerated
    - Region queue ignores predicted_iou thresholds (all regions queued)
    - Frame sampling is applied correctly
    - regions_total and regions_pending counts are accurate
    """
    # ARRANGE: Create mock segmentation data for frames
    from src.models.segmented_region import SegmentedRegion
    from src.models.segmentation_frame import SegmentationFrame, InstanceMask

    # Create 3 frames with varying region counts
    frame_data = [
        (0, 5),  # Frame 0: 5 regions
        (1, 8),  # Frame 1: 8 regions
        (2, 6),  # Frame 2: 6 regions
    ]

    for frame_idx, region_count in frame_data:
        frame = SegmentationFrame(
            id=uuid4(),
            session_id=test_session.id,
            frame_index=frame_idx,
            timestamp=frame_idx / 30.0,
            image_path=Path(f"/tmp/frame_{frame_idx}.jpg"),
            masks=[
                InstanceMask(
                    mask_path=Path(f"/tmp/mask_{frame_idx}_{i}.npz"),
                    class_label=f"object_{i}",
                    confidence=min(0.7 + (i * 0.03), 0.95),  # Varying confidence, capped at 0.95
                    bbox=[10 + i * 50, 10, 40, 40],
                    area=1600,
                )
                for i in range(region_count)
            ],
            processing_time=1.5,
            model_version_id=uuid4(),
            processed_at=datetime.now(),
        )
        storage_service.save_segmentation_frame(str(test_session.id), frame)

    # ACT: Create job with frame_sampling=1 (process all frames)
    job = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        frame_sampling=1,  # Process all frames
        enable_tracking=False,  # Disable tracking for this test
    )

    # ASSERT: Region queue built correctly
    assert job.progress.frames_total == 3, "Should have 3 frames to process"
    assert job.progress.regions_total == 19, "Should have 19 total regions (5+8+6)"
    assert len(job.regions_pending) == 19, "Pending queue should contain all 19 regions"
    assert len(job.regions_completed) == 0, "Completed queue should be empty initially"

    # Test with frame_sampling=2 (every 2nd frame)
    job2 = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        frame_sampling=2,  # Process every 2nd frame (frames 0, 2)
        enable_tracking=False,
    )

    assert job2.progress.frames_total == 2, "Should have 2 frames with sampling=2"
    assert job2.progress.regions_total == 11, "Should have 11 regions (frames 0 and 2: 5+6)"
    assert len(job2.regions_pending) == 11, "Pending queue should contain 11 regions"

    # Verify all pending regions are valid UUIDs
    for region_id in job.regions_pending:
        assert isinstance(region_id, UUID), "All region IDs should be UUIDs"


def test_job_state_transition_methods():
    """Test job state transition helper methods.

    TDD: T017 [US1] - Test job status validation methods

    Validates:
    - is_startable() returns True only for PENDING status
    - is_pausable() returns True only for RUNNING status
    - is_resumable() returns True for PAUSED and PAUSED_BUDGET_LIMIT
    - is_terminal() returns True for COMPLETED, FAILED, CANCELLED
    """
    # ARRANGE
    job_id = uuid4()
    session_id = uuid4()
    video_path = Path("/tmp/test.mp4")

    # ACT & ASSERT: Test PENDING status
    pending_job = SemanticLabelingJob(
        id=job_id,
        session_id=session_id,
        video_path=video_path,
        status=JobStatus.PENDING,
        progress=JobProgress(frames_total=10, frames_pending=10, regions_total=50, regions_pending=50),
        cost_tracking=CostTracking(),
        configuration=JobConfiguration(),
        timestamps=JobTimestamps(created_at=datetime.now()),
        last_checkpoint_at=datetime.now(),
    )

    assert pending_job.is_startable() is True, "PENDING job should be startable"
    assert pending_job.is_pausable() is False, "PENDING job should not be pausable"
    assert pending_job.is_resumable() is False, "PENDING job should not be resumable"
    assert pending_job.is_terminal() is False, "PENDING job is not terminal"

    # Test RUNNING status
    pending_job.status = JobStatus.RUNNING
    assert pending_job.is_startable() is False, "RUNNING job should not be startable"
    assert pending_job.is_pausable() is True, "RUNNING job should be pausable"
    assert pending_job.is_resumable() is False, "RUNNING job should not be resumable"
    assert pending_job.is_terminal() is False, "RUNNING job is not terminal"

    # Test PAUSED status
    pending_job.status = JobStatus.PAUSED
    assert pending_job.is_startable() is False, "PAUSED job should not be startable"
    assert pending_job.is_pausable() is False, "PAUSED job should not be pausable"
    assert pending_job.is_resumable() is True, "PAUSED job should be resumable"
    assert pending_job.is_terminal() is False, "PAUSED job is not terminal"

    # Test PAUSED_BUDGET_LIMIT status
    pending_job.status = JobStatus.PAUSED_BUDGET_LIMIT
    assert pending_job.is_resumable() is True, "PAUSED_BUDGET_LIMIT job should be resumable"
    assert pending_job.is_terminal() is False, "PAUSED_BUDGET_LIMIT job is not terminal"

    # Test COMPLETED status
    pending_job.status = JobStatus.COMPLETED
    assert pending_job.is_startable() is False, "COMPLETED job should not be startable"
    assert pending_job.is_pausable() is False, "COMPLETED job should not be pausable"
    assert pending_job.is_resumable() is False, "COMPLETED job should not be resumable"
    assert pending_job.is_terminal() is True, "COMPLETED job is terminal"

    # Test FAILED status
    pending_job.status = JobStatus.FAILED
    assert pending_job.is_terminal() is True, "FAILED job is terminal"

    # Test CANCELLED status
    pending_job.status = JobStatus.CANCELLED
    assert pending_job.is_terminal() is True, "CANCELLED job is terminal"


def test_job_progress_calculation_methods():
    """Test job progress calculation methods.

    TDD: T017 [US1] - Test progress and cost calculation methods

    Validates:
    - update_progress_percentage() calculates correctly
    - update_budget_consumed_percentage() calculates correctly
    - update_average_cost_per_region() calculates correctly
    - estimate_remaining_cost() calculates correctly
    """
    # ARRANGE
    job = SemanticLabelingJob(
        id=uuid4(),
        session_id=uuid4(),
        video_path=Path("/tmp/test.mp4"),
        status=JobStatus.RUNNING,
        progress=JobProgress(
            frames_total=10,
            frames_processed=5,
            frames_pending=5,
            regions_total=100,
            regions_completed=40,
            regions_pending=60,
        ),
        cost_tracking=CostTracking(
            total_cost=2.50,
            budget_limit=10.0,
            queries_successful=40,
        ),
        configuration=JobConfiguration(),
        timestamps=JobTimestamps(created_at=datetime.now()),
        last_checkpoint_at=datetime.now(),
    )

    # ACT & ASSERT: Progress percentage
    job.update_progress_percentage()
    assert job.progress.progress_percentage == 40.0, "Should calculate 40% progress (40/100 regions)"

    # Budget consumed percentage
    job.update_budget_consumed_percentage()
    assert job.cost_tracking.budget_consumed_percentage == 25.0, "Should calculate 25% budget consumed ($2.50/$10)"

    # Average cost per region
    job.update_average_cost_per_region()
    expected_avg = 2.50 / 40
    assert abs(job.cost_tracking.average_cost_per_region - expected_avg) < 0.001, \
        f"Should calculate average cost per region: {expected_avg}"

    # Estimated remaining cost
    job.estimate_remaining_cost()
    expected_remaining = expected_avg * 60  # 60 pending regions
    assert job.cost_tracking.estimated_remaining_cost is not None
    assert abs(job.cost_tracking.estimated_remaining_cost - expected_remaining) < 0.01, \
        f"Should estimate remaining cost: {expected_remaining}"
