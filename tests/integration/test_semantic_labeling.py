"""Integration tests for semantic labeling feature.

TDD Red Phase: T014-T016 [US1] - Write failing tests FIRST before implementation

These tests validate User Story 1: Automatic Semantic Labeling of All Regions
"""

import pytest
from uuid import uuid4
from pathlib import Path
from datetime import datetime

# Import models and services (will fail until implemented)
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
from src.models.vlm_query import VLMQueryStatus


@pytest.fixture
def storage_service(tmp_path):
    """Create StorageService with temporary directory."""
    return StorageService(base_dir=tmp_path / "sessions")


@pytest.fixture
def vlm_service():
    """Create VLMService with test API key."""
    import os
    api_key = os.getenv("OPENAI_API_KEY", "test-api-key")
    return VLMService(api_key=api_key, confidence_threshold=0.5)


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

    # Create mock session data
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
            "frame_count": 300,
            "resolution": "1920x1080",
        },
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    storage_service.save_video_session(session)
    return session


# T014 [P] [US1] - Test automatic region queuing regardless of quality


def test_all_regions_queued_regardless_of_quality(
    semantic_labeling_service, test_session, storage_service
):
    """Test that all regions are automatically queued for VLM labeling.

    Scenario 1.1 from quickstart.md:
    Given a video frame with 10 regions (8 high quality, 2 low quality),
    When the frame processing completes,
    Then all 10 regions are automatically queued for VLM labeling.

    TDD: T014 [US1] - This test should FAIL until implementation is complete
    """
    # ARRANGE: Create mock regions (8 high quality, 2 low quality)
    from src.models.segmented_region import SegmentedRegion

    regions = []
    for i in range(10):
        confidence = 0.85 if i < 8 else 0.65  # 8 high quality, 2 low quality
        region = SegmentedRegion(
            id=uuid4(),
            session_id=test_session.id,
            frame_id=uuid4(),
            frame_index=0,
            bbox=[10 + i * 50, 10, 40, 40],
            mask_path=Path(f"/tmp/mask_{i}.npz"),
            area=1600,
            confidence=confidence,
            created_at=datetime.now(),
        )
        regions.append(region)

    # ACT: Create labeling job for frame 0
    job = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        frame_sampling=1,
        enable_tracking=False,  # Disable tracking for single-frame test
    )

    # Start job
    semantic_labeling_service.start_job(job_id=job.id)

    # Wait for job completion (or timeout)
    import time
    max_wait = 120  # 2 minutes
    elapsed = 0
    while elapsed < max_wait:
        job_status = semantic_labeling_service.get_job_status(job.id)
        if job_status.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(1)
        elapsed += 1

    # ASSERT: All 10 regions queued and labeled
    final_job = semantic_labeling_service.get_job_status(job.id)

    assert final_job.status == JobStatus.COMPLETED, f"Job should complete successfully, got {final_job.status}"
    assert final_job.progress.regions_total == 10, "Should have 10 total regions"
    assert final_job.progress.regions_completed == 10, "Should complete all 10 regions"
    assert final_job.cost_tracking.queries_successful >= 10, "Should query VLM for all 10 regions"


# T015 [P] [US1] - Test high-quality regions still sent to VLM


def test_high_quality_regions_sent_to_vlm(
    semantic_labeling_service, test_session, storage_service
):
    """Test that high-quality segmentations are sent to VLM to detect semantic uncertainty.

    Scenario 1.2 from quickstart.md:
    Given all regions in a frame have high segmentation quality (>0.75),
    When the frame is processed,
    Then the system still sends all regions to the VLM for semantic classification.

    TDD: T015 [US1] - This test should FAIL until implementation is complete
    """
    # ARRANGE: Create mock regions (all high quality)
    from src.models.segmented_region import SegmentedRegion

    regions = []
    for i in range(5):
        region = SegmentedRegion(
            id=uuid4(),
            session_id=test_session.id,
            frame_id=uuid4(),
            frame_index=0,
            bbox=[10 + i * 50, 10, 40, 40],
            mask_path=Path(f"/tmp/mask_{i}.npz"),
            area=1600,
            confidence=0.85,  # All high quality (>0.75)
            created_at=datetime.now(),
        )
        regions.append(region)

    # ACT: Create and start labeling job
    job = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        frame_sampling=1,
        enable_tracking=False,
    )

    semantic_labeling_service.start_job(job_id=job.id)

    # Wait for completion
    import time
    max_wait = 60
    elapsed = 0
    while elapsed < max_wait:
        job_status = semantic_labeling_service.get_job_status(job.id)
        if job_status.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(1)
        elapsed += 1

    # ASSERT: High-quality regions sent to VLM
    final_job = semantic_labeling_service.get_job_status(job.id)

    assert final_job.progress.regions_completed == 5, "Should complete all 5 high-quality regions"
    assert final_job.cost_tracking.queries_successful >= 5, "Should query VLM for all regions despite high quality"

    # Verify semantic labels or uncertain status assigned
    total_labeled = (
        final_job.cost_tracking.queries_successful +
        final_job.cost_tracking.queries_uncertain
    )
    assert total_labeled == 5, "All regions should have semantic labels or VLM_UNCERTAIN status"


# T016 [P] [US1] - Test VLM_UNCERTAIN regions flagged for manual labeling


def test_vlm_uncertain_flagged_for_manual_labeling(
    semantic_labeling_service, test_session, storage_service
):
    """Test that low-confidence VLM responses are marked as VLM_UNCERTAIN.

    Scenario 1.3 from quickstart.md:
    Given VLM returns confident labels for 7 regions but low confidence for 3 regions,
    When the user reviews the frame,
    Then the 3 low-confidence regions are marked as VLM_UNCERTAIN and flagged for manual labeling.

    TDD: T016 [US1] - This test should FAIL until implementation is complete
    """
    # ARRANGE: Create mock regions
    from src.models.segmented_region import SegmentedRegion

    regions = []
    for i in range(10):
        region = SegmentedRegion(
            id=uuid4(),
            session_id=test_session.id,
            frame_id=uuid4(),
            frame_index=0,
            bbox=[10 + i * 50, 10, 40, 40],
            mask_path=Path(f"/tmp/mask_{i}.npz"),
            area=1600,
            confidence=0.80,
            created_at=datetime.now(),
        )
        regions.append(region)

    # ACT: Create and start labeling job with confidence threshold 0.5
    job = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        frame_sampling=1,
        confidence_threshold=0.5,  # Mark regions with VLM confidence < 0.5 as uncertain
        enable_tracking=False,
    )

    semantic_labeling_service.start_job(job_id=job.id)

    # Wait for completion
    import time
    max_wait = 120
    elapsed = 0
    while elapsed < max_wait:
        job_status = semantic_labeling_service.get_job_status(job.id)
        if job_status.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(1)
        elapsed += 1

    # ASSERT: Some regions marked as uncertain
    final_job = semantic_labeling_service.get_job_status(job.id)

    assert final_job.progress.regions_completed == 10, "Should complete all 10 regions"
    assert final_job.cost_tracking.queries_uncertain > 0, "Should detect some VLM_UNCERTAIN regions"

    # Verify uncertain regions have vlm_query.status == VLM_UNCERTAIN
    # (This requires loading actual region data - implementation detail)
    assert final_job.cost_tracking.queries_successful + final_job.cost_tracking.queries_uncertain == 10, \
        "All regions should be either successful or uncertain"


# ============================================================================
# User Story 2: Cost-Controlled Batch Processing Tests
# ============================================================================


# T027 [P] [US2] - Test cost estimation within 10% accuracy


def test_cost_estimation_within_10_percent_accuracy(
    semantic_labeling_service, test_session, storage_service
):
    """Test that cost estimation is accurate within 10% margin.

    Scenario 2.1 from quickstart.md:
    Given a video with 100 frames,
    When cost estimation is performed,
    Then the estimated cost should be within 10% of actual cost.

    TDD: T027 [US2] - This test should FAIL until implementation is complete
    """
    # ARRANGE: Create mock segmentation data for 100 frames
    from src.models.segmented_region import SegmentedRegion
    from src.models.segmentation_frame import SegmentationFrame, InstanceMask

    # Create 100 frames with 5-10 regions each
    import random
    random.seed(42)  # Deterministic for testing

    for frame_idx in range(100):
        region_count = random.randint(5, 10)
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
                    confidence=min(0.7 + (i * 0.03), 0.95),
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

    # ACT: Create job (PENDING state)
    job = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        frame_sampling=1,
        enable_tracking=True,  # Enable tracking for more accurate estimation
    )

    # Request cost estimate
    estimate = semantic_labeling_service.estimate_job_cost(job_id=job.id)

    # Record estimated cost
    estimated_cost = estimate["estimated_cost"]

    # Start and complete job
    semantic_labeling_service.start_job(job_id=job.id)

    # Wait for completion
    import time
    max_wait = 300  # 5 minutes
    elapsed = 0
    while elapsed < max_wait:
        job_status = semantic_labeling_service.get_job_status(job.id)
        if job_status.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(1)
        elapsed += 1

    # ASSERT: Actual cost within 10% of estimate
    final_job = semantic_labeling_service.get_job_status(job.id)
    actual_cost = final_job.cost_tracking.total_cost

    error_margin = abs(actual_cost - estimated_cost) / estimated_cost if estimated_cost > 0 else 0
    assert error_margin <= 0.10, f"Cost estimate error {error_margin:.1%} exceeds 10% threshold (estimated: ${estimated_cost:.2f}, actual: ${actual_cost:.2f})"

    # Verify estimate structure
    assert "estimated_cost" in estimate
    assert "min_cost" in estimate
    assert "max_cost" in estimate
    assert "sample_size" in estimate
    assert "scene_stability" in estimate
    assert estimate["scene_stability"] in ["stable", "moderate", "dynamic"]


# T028 [P] [US2] - Test budget limit auto-pause at 95%


def test_budget_limit_auto_pause_at_95_percent(
    semantic_labeling_service, test_session, storage_service
):
    """Test that processing pauses automatically when 95% of budget consumed.

    Scenario 2.2 from quickstart.md:
    Given a job with $5 budget limit,
    When cumulative costs reach $4.75 (95%),
    Then the system should pause automatically.

    TDD: T028 [US2] - This test should FAIL until implementation is complete
    """
    # ARRANGE: Create many regions to ensure budget hit
    from src.models.segmentation_frame import SegmentationFrame, InstanceMask

    # Create 50 frames with 30 regions each (high region count)
    for frame_idx in range(50):
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
                    confidence=0.85,
                    bbox=[10 + i * 20, 10, 15, 15],
                    area=225,
                )
                for i in range(30)
            ],
            processing_time=1.5,
            model_version_id=uuid4(),
            processed_at=datetime.now(),
        )
        storage_service.save_segmentation_frame(str(test_session.id), frame)

    # ACT: Create job with low budget to trigger pause
    job = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        budget_limit=5.00,  # $5 budget limit
        frame_sampling=1,
        enable_tracking=False,  # Disable tracking to ensure budget hit
    )

    # Start job
    semantic_labeling_service.start_job(job_id=job.id)

    # Wait for pause or completion
    import time
    max_wait = 300
    elapsed = 0
    while elapsed < max_wait:
        job_status = semantic_labeling_service.get_job_status(job.id)
        if job_status.status in [JobStatus.PAUSED_BUDGET_LIMIT, JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(1)
        elapsed += 1

    # ASSERT: Job paused due to budget
    final_job = semantic_labeling_service.get_job_status(job.id)

    assert final_job.status == JobStatus.PAUSED_BUDGET_LIMIT, \
        f"Expected PAUSED_BUDGET_LIMIT, got {final_job.status}"

    # Verify paused at 95%+ budget consumption
    budget_consumed_pct = final_job.cost_tracking.budget_consumed_percentage
    assert budget_consumed_pct >= 95.0, \
        f"Budget consumed {budget_consumed_pct:.1f}% is less than 95%"
    assert budget_consumed_pct <= 100.0, \
        f"Budget consumed {budget_consumed_pct:.1f}% exceeds 100% (should not overspend)"

    # Verify total cost does not exceed budget
    assert final_job.cost_tracking.total_cost <= final_job.cost_tracking.budget_limit, \
        f"Total cost ${final_job.cost_tracking.total_cost:.2f} exceeds budget ${final_job.cost_tracking.budget_limit:.2f}"

    # Verify partial progress saved
    assert final_job.progress.regions_completed > 0, "Should have completed some regions"
    assert final_job.progress.regions_pending > 0, "Should have remaining regions"


# T029 [P] [US2] - Test frame sampling reduces cost proportionally


def test_frame_sampling_reduces_cost_proportionally(
    semantic_labeling_service, test_session, storage_service
):
    """Test that frame sampling reduces costs proportionally to sampling ratio.

    Scenario 2.3 from quickstart.md:
    Given frame sampling set to every 5th frame,
    When processing completes,
    Then cost should be ~20% of full processing cost.

    TDD: T029 [US2] - This test should FAIL until implementation is complete
    """
    # ARRANGE: Create 100 frames with consistent region count
    from src.models.segmentation_frame import SegmentationFrame, InstanceMask

    for frame_idx in range(100):
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
                    confidence=0.85,
                    bbox=[10 + i * 50, 10, 40, 40],
                    area=1600,
                )
                for i in range(5)  # Consistent 5 regions per frame
            ],
            processing_time=1.5,
            model_version_id=uuid4(),
            processed_at=datetime.now(),
        )
        storage_service.save_segmentation_frame(str(test_session.id), frame)

    # ACT: Job 1 - Process all frames (baseline)
    job_full = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        frame_sampling=1,  # Every frame
        enable_tracking=False,  # Disable for cost comparison
    )
    semantic_labeling_service.start_job(job_id=job_full.id)

    # Wait for completion
    import time
    max_wait = 300
    elapsed = 0
    while elapsed < max_wait:
        job_status = semantic_labeling_service.get_job_status(job_full.id)
        if job_status.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(1)
        elapsed += 1

    full_cost = semantic_labeling_service.get_job_status(job_full.id).cost_tracking.total_cost

    # Job 2 - Process every 5th frame
    job_sampled = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        frame_sampling=5,  # Every 5th frame
        enable_tracking=False,
    )
    semantic_labeling_service.start_job(job_id=job_sampled.id)

    # Wait for completion
    elapsed = 0
    while elapsed < max_wait:
        job_status = semantic_labeling_service.get_job_status(job_sampled.id)
        if job_status.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(1)
        elapsed += 1

    sampled_cost = semantic_labeling_service.get_job_status(job_sampled.id).cost_tracking.total_cost

    # ASSERT: Cost reduction proportional to sampling ratio
    expected_ratio = 1.0 / 5.0  # 20% of frames
    actual_ratio = sampled_cost / full_cost if full_cost > 0 else 0

    # Allow 10% tolerance for variance
    assert 0.18 <= actual_ratio <= 0.22, \
        f"Sampling ratio {actual_ratio:.2%} not ~20% (expected ~20%, got {actual_ratio:.1%})"

    # Verify frames processed matches sampling
    final_sampled = semantic_labeling_service.get_job_status(job_sampled.id)
    assert final_sampled.progress.frames_processed == 20, \
        f"Expected 20 frames processed (100/5), got {final_sampled.progress.frames_processed}"


# T030 [P] [US2] - Test pause/resume preserves state


def test_pause_resume_preserves_state(
    semantic_labeling_service, test_session, storage_service
):
    """Test that pause/resume preserves 100% of job state without data loss.

    Scenario 2.4 from quickstart.md:
    Given a job is running,
    When paused and then resumed,
    Then all progress is preserved and processing continues from checkpoint.

    TDD: T030 [US2] - This test should FAIL until implementation is complete
    """
    # ARRANGE: Create 50 frames
    from src.models.segmentation_frame import SegmentationFrame, InstanceMask

    for frame_idx in range(50):
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
                    confidence=0.85,
                    bbox=[10 + i * 50, 10, 40, 40],
                    area=1600,
                )
                for i in range(5)
            ],
            processing_time=1.5,
            model_version_id=uuid4(),
            processed_at=datetime.now(),
        )
        storage_service.save_segmentation_frame(str(test_session.id), frame)

    # ACT: Create and start job
    job = semantic_labeling_service.create_job(
        session_id=test_session.id,
        video_path=Path(test_session.filepath),
        frame_sampling=1,
        enable_tracking=False,
    )
    semantic_labeling_service.start_job(job_id=job.id)

    # Wait for partial progress (at least 10 frames)
    import time
    max_wait = 60
    elapsed = 0
    while elapsed < max_wait:
        job_status = semantic_labeling_service.get_job_status(job.id)
        if job_status.progress.frames_processed >= 10:
            break
        time.sleep(1)
        elapsed += 1

    # Pause job
    paused_job = semantic_labeling_service.pause_job(job_id=job.id)

    # Wait for pause to complete
    elapsed = 0
    while elapsed < 10:
        job_status = semantic_labeling_service.get_job_status(job.id)
        if job_status.status == JobStatus.PAUSED:
            break
        time.sleep(0.5)
        elapsed += 0.5

    # Record state at pause
    paused_status = semantic_labeling_service.get_job_status(job.id)
    paused_frames = paused_status.progress.frames_processed
    paused_regions = paused_status.progress.regions_completed
    paused_cost = paused_status.cost_tracking.total_cost

    # Resume job
    resumed_job = semantic_labeling_service.resume_job(job_id=job.id)

    # Wait for completion
    max_wait = 300
    elapsed = 0
    while elapsed < max_wait:
        job_status = semantic_labeling_service.get_job_status(job.id)
        if job_status.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(1)
        elapsed += 1

    # ASSERT: State preserved and job completed
    final_status = semantic_labeling_service.get_job_status(job.id)

    assert paused_status.status == JobStatus.PAUSED, "Job should be paused"
    assert paused_regions > 0, "Should have completed some regions before pause"

    assert final_status.status == JobStatus.COMPLETED, "Job should complete after resume"
    assert final_status.progress.frames_processed == 50, "Should process all 50 frames"
    assert final_status.progress.regions_completed >= paused_regions, \
        "Final regions should be >= paused regions"

    # Verify cost accumulated correctly (no double-charging)
    assert final_status.cost_tracking.total_cost > paused_cost, \
        "Final cost should be greater than paused cost"

    # Verify no data loss
    assert len(paused_status.regions_completed) > 0, "Should have completed regions at pause"
    assert len(final_status.regions_completed) == final_status.progress.regions_completed, \
        "Completed regions list should match count"
