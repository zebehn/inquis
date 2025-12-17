"""Unit tests for Pydantic data models.

TDD: These tests are written FIRST and must FAIL before implementation.
"""

import pytest
from datetime import datetime
from uuid import uuid4, UUID
from pathlib import Path
from src.models.video_session import VideoSession, SessionStatus
from src.models.segmentation_frame import SegmentationFrame, InstanceMask


class TestVideoSession:
    """Test VideoSession model validation."""

    def test_create_valid_video_session(self):
        """Test creating a valid video session."""
        session = VideoSession(
            id=uuid4(),
            filename="test_video.mp4",
            filepath=Path("/data/sessions/test/video.mp4"),
            upload_timestamp=datetime.now(),
            status=SessionStatus.UPLOADING,
            metadata={
                "resolution": [1920, 1080],
                "frame_count": 300,
                "duration": 10.0,
                "fps": 30.0,
                "codec": "h264",
            },
            processing_progress=0.0,
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert session.filename == "test_video.mp4"
        assert session.status == SessionStatus.UPLOADING
        assert session.metadata["frame_count"] == 300
        assert session.processing_progress == 0.0

    def test_video_session_validates_filename_extension(self):
        """Test that invalid file extensions are rejected."""
        with pytest.raises(ValueError, match="extension"):
            VideoSession(
                id=uuid4(),
                filename="test_video.txt",  # Invalid extension
                filepath=Path("/data/sessions/test/video.txt"),
                upload_timestamp=datetime.now(),
                status=SessionStatus.UPLOADING,
                metadata={
                    "resolution": [1920, 1080],
                    "frame_count": 300,
                    "duration": 10.0,
                    "fps": 30.0,
                    "codec": "h264",
                },
                processing_progress=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_video_session_validates_duration(self):
        """Test that video duration exceeding max is rejected."""
        with pytest.raises(ValueError, match="duration"):
            VideoSession(
                id=uuid4(),
                filename="test_video.mp4",
                filepath=Path("/data/sessions/test/video.mp4"),
                upload_timestamp=datetime.now(),
                status=SessionStatus.UPLOADING,
                metadata={
                    "resolution": [1920, 1080],
                    "frame_count": 300,
                    "duration": 700.0,  # > 600 seconds max
                    "fps": 30.0,
                    "codec": "h264",
                },
                processing_progress=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_video_session_validates_fps(self):
        """Test that FPS outside valid range is rejected."""
        with pytest.raises(ValueError, match="fps"):
            VideoSession(
                id=uuid4(),
                filename="test_video.mp4",
                filepath=Path("/data/sessions/test/video.mp4"),
                upload_timestamp=datetime.now(),
                status=SessionStatus.UPLOADING,
                metadata={
                    "resolution": [1920, 1080],
                    "frame_count": 300,
                    "duration": 10.0,
                    "fps": 120.0,  # > 60 fps max
                    "codec": "h264",
                },
                processing_progress=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_video_session_validates_processing_progress(self):
        """Test that processing progress outside [0, 1] is rejected."""
        with pytest.raises(ValueError, match="progress"):
            VideoSession(
                id=uuid4(),
                filename="test_video.mp4",
                filepath=Path("/data/sessions/test/video.mp4"),
                upload_timestamp=datetime.now(),
                status=SessionStatus.PROCESSING,
                metadata={
                    "resolution": [1920, 1080],
                    "frame_count": 300,
                    "duration": 10.0,
                    "fps": 30.0,
                    "codec": "h264",
                },
                processing_progress=1.5,  # > 1.0
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_video_session_to_dict(self):
        """Test converting video session to dictionary."""
        session_id = uuid4()
        now = datetime.now()

        session = VideoSession(
            id=session_id,
            filename="test_video.mp4",
            filepath=Path("/data/sessions/test/video.mp4"),
            upload_timestamp=now,
            status=SessionStatus.PROCESSING,
            metadata={
                "resolution": [1920, 1080],
                "frame_count": 300,
                "duration": 10.0,
                "fps": 30.0,
                "codec": "h264",
            },
            processing_progress=0.5,
            created_at=now,
            updated_at=now,
        )

        session_dict = session.model_dump()

        assert session_dict["id"] == session_id
        assert session_dict["filename"] == "test_video.mp4"
        assert session_dict["status"] == SessionStatus.PROCESSING
        assert session_dict["processing_progress"] == 0.5


class TestSegmentationFrame:
    """Test SegmentationFrame model validation."""

    def test_create_valid_segmentation_frame(self):
        """Test creating a valid segmentation frame."""
        session_id = uuid4()
        frame_id = uuid4()
        model_version_id = uuid4()

        frame = SegmentationFrame(
            id=frame_id,
            session_id=session_id,
            frame_index=42,
            timestamp=1.4,
            image_path=Path("/data/sessions/test/frames/frame_0042.jpg"),
            masks=[
                InstanceMask(
                    mask_path=Path("/data/sessions/test/masks/frame_0042_mask_0.npz"),
                    class_label="car",
                    confidence=0.92,
                    bbox=[100, 200, 150, 80],
                    area=8500,
                )
            ],
            uncertainty_map_path=None,
            processing_time=0.85,
            model_version_id=model_version_id,
            processed_at=datetime.now(),
        )

        assert frame.frame_index == 42
        assert frame.timestamp == 1.4
        assert len(frame.masks) == 1
        assert frame.masks[0].class_label == "car"
        assert frame.masks[0].confidence == 0.92

    def test_segmentation_frame_validates_negative_frame_index(self):
        """Test that negative frame index is rejected."""
        with pytest.raises(ValueError, match="frame_index"):
            SegmentationFrame(
                id=uuid4(),
                session_id=uuid4(),
                frame_index=-1,  # Invalid negative
                timestamp=0.0,
                image_path=Path("/data/sessions/test/frames/frame_0000.jpg"),
                masks=[],
                processing_time=0.5,
                model_version_id=uuid4(),
                processed_at=datetime.now(),
            )

    def test_segmentation_frame_validates_mask_confidence(self):
        """Test that mask confidence outside [0, 1] is rejected."""
        with pytest.raises(ValueError, match="confidence"):
            InstanceMask(
                mask_path=Path("/data/sessions/test/masks/mask.npz"),
                class_label="car",
                confidence=1.5,  # > 1.0
                bbox=[100, 200, 150, 80],
                area=8500,
            )

    def test_segmentation_frame_validates_bbox_values(self):
        """Test that negative bbox values are rejected."""
        with pytest.raises(ValueError, match="bbox"):
            InstanceMask(
                mask_path=Path("/data/sessions/test/masks/mask.npz"),
                class_label="car",
                confidence=0.9,
                bbox=[-10, 200, 150, 80],  # Negative x
                area=8500,
            )

    def test_instance_mask_to_dict(self):
        """Test converting instance mask to dictionary."""
        mask = InstanceMask(
            mask_path=Path("/data/sessions/test/masks/mask.npz"),
            class_label="person",
            confidence=0.88,
            bbox=[50, 100, 200, 300],
            area=45000,
        )

        mask_dict = mask.model_dump()

        assert mask_dict["class_label"] == "person"
        assert mask_dict["confidence"] == 0.88
        assert mask_dict["bbox"] == [50, 100, 200, 300]
        assert mask_dict["area"] == 45000

    def test_segmentation_frame_to_dict(self):
        """Test converting segmentation frame to dictionary."""
        frame_id = uuid4()
        session_id = uuid4()

        frame = SegmentationFrame(
            id=frame_id,
            session_id=session_id,
            frame_index=10,
            timestamp=0.333,
            image_path=Path("/data/sessions/test/frames/frame_0010.jpg"),
            masks=[],
            processing_time=0.42,
            model_version_id=uuid4(),
            processed_at=datetime.now(),
        )

        frame_dict = frame.model_dump()

        assert frame_dict["id"] == frame_id
        assert frame_dict["session_id"] == session_id
        assert frame_dict["frame_index"] == 10
        assert frame_dict["processing_time"] == 0.42
