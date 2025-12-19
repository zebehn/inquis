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
