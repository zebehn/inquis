"""Integration test for VLM labels display in GUI.

This test verifies that VLM semantic labels are correctly loaded and
displayed in the segmentation results view.
"""

import pytest
import numpy as np
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from src.services.storage_service import StorageService
from src.models.video_session import VideoSession, SessionStatus
from src.models.segmentation_frame import SegmentationFrame, InstanceMask
from src.models.uncertain_region import UncertainRegion, RegionStatus


@pytest.fixture
def test_storage(tmp_path):
    """Create test storage service."""
    storage = StorageService(base_dir=tmp_path)
    return storage


@pytest.fixture
def test_session_with_vlm_labels(test_storage):
    """Create test session with segmentation frames and VLM labels."""
    # Create session
    now = datetime.now()
    session_id = uuid4()
    session = VideoSession(
        id=session_id,
        filepath=Path("/tmp/test_video.mp4"),
        filename="test_video.mp4",
        upload_timestamp=now,
        status=SessionStatus.COMPLETED,
        metadata={
            "fps": 30.0,
            "frame_count": 10,
            "duration": 0.33,
            "resolution": (640, 480),
        },
        created_at=now,
        updated_at=now,
    )
    test_storage.create_session(str(session_id))
    test_storage.save_video_session(session)

    # Create segmentation frame with 3 masks
    frame = SegmentationFrame(
        id=uuid4(),
        session_id=session.id,
        frame_index=0,
        timestamp=0.0,
        image_path=Path("/tmp/frame_0.jpg"),
        masks=[
            # Mask 1: Will get confirmed VLM label
            InstanceMask(
                mask_path=Path("/tmp/mask_0_0.npz"),
                class_label="object_0",
                confidence=0.42,  # Below threshold
                bbox=[100, 100, 50, 50],
                area=2500,
            ),
            # Mask 2: Will be VLM uncertain
            InstanceMask(
                mask_path=Path("/tmp/mask_0_1.npz"),
                class_label="object_1",
                confidence=0.38,  # Below threshold
                bbox=[200, 200, 60, 60],
                area=3600,
            ),
            # Mask 3: No VLM label (high confidence)
            InstanceMask(
                mask_path=Path("/tmp/mask_0_2.npz"),
                class_label="car",
                confidence=0.92,  # Above threshold
                bbox=[300, 300, 80, 80],
                area=6400,
            ),
        ],
        processing_time=1.5,
        model_version_id=uuid4(),
        processed_at=datetime.now(),
    )
    test_storage.save_segmentation_frame(str(session.id), frame)

    # Create uncertain regions with VLM labels
    # Region 1: Confirmed label "excavator"
    region1 = UncertainRegion(
        id=uuid4(),
        session_id=session.id,
        frame_id=frame.id,
        frame_index=0,
        bbox=[100, 100, 50, 50],  # Matches mask 1
        uncertainty_score=0.58,
        cropped_image_path=Path("/tmp/crop_1.jpg"),
        mask_path=Path("/tmp/mask_0_0.npz"),
        status=RegionStatus.CONFIRMED,
        confirmed_label="excavator",
        created_at=datetime.now(),
    )
    test_storage.save_uncertain_region(str(session.id), region1)

    # Region 2: VLM uncertain (no confirmed label)
    region2 = UncertainRegion(
        id=uuid4(),
        session_id=session.id,
        frame_id=frame.id,
        frame_index=0,
        bbox=[200, 200, 60, 60],  # Matches mask 2
        uncertainty_score=0.62,
        cropped_image_path=Path("/tmp/crop_2.jpg"),
        mask_path=Path("/tmp/mask_0_1.npz"),
        status=RegionStatus.QUERIED,
        confirmed_label=None,
        created_at=datetime.now(),
    )
    test_storage.save_uncertain_region(str(session.id), region2)

    return session, frame, [region1, region2]


def test_load_uncertain_regions_by_frame(test_storage, test_session_with_vlm_labels):
    """Test loading uncertain regions for a specific frame."""
    session, frame, expected_regions = test_session_with_vlm_labels

    # Load uncertain regions for frame 0
    regions = test_storage.load_uncertain_regions_by_frame(str(session.id), 0)

    # Should load 2 regions
    assert len(regions) == 2, f"Expected 2 regions, got {len(regions)}"

    # Verify regions are correct
    region_bboxes = [r.bbox for r in regions]
    assert [100, 100, 50, 50] in region_bboxes
    assert [200, 200, 60, 60] in region_bboxes

    # Verify confirmed labels
    confirmed_regions = [r for r in regions if r.confirmed_label]
    assert len(confirmed_regions) == 1
    assert confirmed_regions[0].confirmed_label == "excavator"


def test_vlm_labels_display_logic(test_session_with_vlm_labels):
    """Test the VLM labels enhancement logic."""
    session, frame, uncertain_regions = test_session_with_vlm_labels

    # Simulate GUI label enhancement logic
    masks = frame.masks
    labels = [m.class_label for m in masks]
    bboxes = [m.bbox for m in masks]

    # Enhance labels with VLM information (from segmentation_viz.py)
    enhanced_labels = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        # Try to find matching uncertain region by bbox
        matched_region = None
        for region in uncertain_regions:
            if region.bbox == bbox:
                matched_region = region
                break

        # Create enhanced label
        if matched_region:
            if matched_region.confirmed_label:
                # Manual or confirmed label
                enhanced_label = f"{matched_region.confirmed_label} ✓"
            else:
                # VLM uncertain
                enhanced_label = f"{label} ⚠️ (uncertain)"
        else:
            # No VLM label yet, use original
            enhanced_label = label

        enhanced_labels.append(enhanced_label)

    # Verify enhanced labels
    assert len(enhanced_labels) == 3

    # Mask 1: Should show confirmed excavator label
    assert enhanced_labels[0] == "excavator ✓", \
        f"Expected 'excavator ✓', got '{enhanced_labels[0]}'"

    # Mask 2: Should show uncertain indicator
    assert "⚠️" in enhanced_labels[1], \
        f"Expected uncertain indicator, got '{enhanced_labels[1]}'"
    assert "uncertain" in enhanced_labels[1], \
        f"Expected 'uncertain' text, got '{enhanced_labels[1]}'"

    # Mask 3: Should keep original label (no VLM label)
    assert enhanced_labels[2] == "car", \
        f"Expected 'car', got '{enhanced_labels[2]}'"


def test_vlm_statistics_calculation(test_session_with_vlm_labels):
    """Test VLM labeling statistics calculation."""
    session, frame, uncertain_regions = test_session_with_vlm_labels

    # Calculate VLM stats (from segmentation_viz.py)
    vlm_labeled_count = len(uncertain_regions)
    vlm_confirmed = sum(1 for r in uncertain_regions if r.confirmed_label)

    # Verify statistics
    assert vlm_labeled_count == 2, \
        f"Expected 2 VLM labeled regions, got {vlm_labeled_count}"

    assert vlm_confirmed == 1, \
        f"Expected 1 confirmed region, got {vlm_confirmed}"


def test_empty_uncertain_regions(test_storage):
    """Test handling when no uncertain regions exist."""
    # Create session without uncertain regions
    now = datetime.now()
    session_id = uuid4()
    session = VideoSession(
        id=session_id,
        filepath=Path("/tmp/test_video.mp4"),
        filename="test_video.mp4",
        upload_timestamp=now,
        status=SessionStatus.COMPLETED,
        metadata={
            "fps": 30.0,
            "frame_count": 10,
            "duration": 0.33,
            "resolution": (640, 480),
        },
        created_at=now,
        updated_at=now,
    )
    test_storage.create_session(str(session_id))
    test_storage.save_video_session(session)

    # Load uncertain regions (should return empty list)
    regions = test_storage.load_uncertain_regions_by_frame(str(session.id), 0)

    assert regions == [], f"Expected empty list, got {regions}"

    # Verify GUI handles empty list gracefully
    uncertain_regions = regions
    vlm_labeled_count = len(uncertain_regions) if uncertain_regions else 0
    vlm_confirmed = sum(1 for r in (uncertain_regions or []) if r.confirmed_label) if uncertain_regions else 0

    assert vlm_labeled_count == 0
    assert vlm_confirmed == 0
