"""Integration tests for storage persistence.

TDD: These tests are written FIRST and must FAIL before implementation.
Tests that data persists correctly across service restarts.
"""

import pytest
import numpy as np
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from src.services.storage_service import StorageService
from src.models.video_session import VideoSession, SessionStatus


@pytest.mark.integration
class TestStoragePersistence:
    """Test data persistence and recovery."""

    def test_video_session_persists_across_restarts(self, tmp_path):
        """Test that video session data survives service restart."""
        # Create session with first storage instance
        storage1 = StorageService(base_dir=tmp_path / "sessions")
        session_id = str(uuid4())
        storage1.create_session(session_id)

        session_data = {
            "id": session_id,
            "filename": "test_video.mp4",
            "status": "processing",
            "processing_progress": 0.5,
        }
        storage1.save_metadata(session_id, "session.json", session_data)

        # Create new storage instance (simulating restart)
        storage2 = StorageService(base_dir=tmp_path / "sessions")
        loaded_data = storage2.load_metadata(session_id, "session.json")

        assert loaded_data["id"] == session_id
        assert loaded_data["processing_progress"] == 0.5

    def test_frames_persist_with_correct_format(self, tmp_path):
        """Test that frame images are saved and loaded correctly."""
        storage = StorageService(base_dir=tmp_path / "sessions")
        session_id = str(uuid4())
        storage.create_session(session_id)

        # Save a frame
        original_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        storage.save_frame(session_id, 42, original_frame)

        # Load the frame
        loaded_frame = storage.load_frame(session_id, 42)

        assert loaded_frame.shape == original_frame.shape
        assert loaded_frame.dtype == original_frame.dtype
        # Allow for JPEG compression artifacts
        assert np.abs(loaded_frame.astype(int) - original_frame.astype(int)).mean() < 5

    def test_masks_persist_as_npz_files(self, tmp_path):
        """Test that segmentation masks are saved and loaded correctly."""
        from src.utils.mask_utils import save_masks, load_masks

        session_path = tmp_path / "sessions" / "test_session"
        session_path.mkdir(parents=True)
        masks_dir = session_path / "masks"
        masks_dir.mkdir()

        # Save masks
        original_masks = np.random.rand(5, 480, 640) > 0.5
        original_confidences = np.array([0.95, 0.87, 0.82, 0.76, 0.91])

        mask_path = masks_dir / "frame_0042.npz"
        save_masks(original_masks, original_confidences, mask_path)

        # Load masks
        loaded_masks, loaded_confidences = load_masks(mask_path)

        np.testing.assert_array_equal(loaded_masks, original_masks)
        np.testing.assert_array_almost_equal(loaded_confidences, original_confidences)

    def test_session_directory_structure_is_complete(self, tmp_path):
        """Test that session directory has all required subdirectories."""
        storage = StorageService(base_dir=tmp_path / "sessions")
        session_id = str(uuid4())
        session_path = storage.create_session(session_id)

        # Verify directory structure
        assert (session_path / "frames").exists()
        assert (session_path / "masks").exists()
        assert (session_path / "metadata").exists()
        assert (session_path / "frames").is_dir()
        assert (session_path / "masks").is_dir()
        assert (session_path / "metadata").is_dir()

    def test_large_session_data_persists(self, tmp_path):
        """Test persistence of large session with many frames."""
        storage = StorageService(base_dir=tmp_path / "sessions")
        session_id = str(uuid4())
        storage.create_session(session_id)

        # Save 100 frames
        num_frames = 100
        for frame_idx in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            storage.save_frame(session_id, frame_idx, frame)

        # Verify all frames are accessible
        for frame_idx in range(num_frames):
            frame = storage.load_frame(session_id, frame_idx)
            assert frame.shape == (480, 640, 3)

    def test_concurrent_session_access(self, tmp_path):
        """Test that multiple sessions can coexist."""
        storage = StorageService(base_dir=tmp_path / "sessions")

        # Create multiple sessions
        session_ids = [str(uuid4()) for _ in range(5)]
        for session_id in session_ids:
            storage.create_session(session_id)
            storage.save_metadata(
                session_id, "session.json", {"id": session_id, "status": "processing"}
            )

        # Verify all sessions are listed
        all_sessions = storage.list_sessions()
        assert len(all_sessions) == 5

        # Verify each session data is isolated
        for session_id in session_ids:
            data = storage.load_metadata(session_id, "session.json")
            assert data["id"] == session_id

    def test_session_deletion_removes_all_data(self, tmp_path):
        """Test that deleting a session removes all associated files."""
        storage = StorageService(base_dir=tmp_path / "sessions")
        session_id = str(uuid4())
        session_path = storage.create_session(session_id)

        # Add some data
        storage.save_metadata(session_id, "test.json", {"data": "test"})
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        storage.save_frame(session_id, 0, frame)

        # Delete session
        storage.delete_session(session_id)

        # Verify session is gone
        assert not session_path.exists()
        assert session_id not in storage.list_sessions()

    def test_corrupted_session_data_handling(self, tmp_path):
        """Test handling of corrupted session metadata files."""
        storage = StorageService(base_dir=tmp_path / "sessions")
        session_id = str(uuid4())
        storage.create_session(session_id)

        # Write corrupted JSON
        metadata_path = storage.get_session_path(session_id) / "metadata" / "corrupt.json"
        metadata_path.write_text("{invalid json content")

        # Should raise appropriate error
        with pytest.raises(Exception):  # Could be JSONDecodeError or similar
            storage.load_metadata(session_id, "corrupt.json")

    def test_metadata_includes_timestamps(self, tmp_path):
        """Test that saved metadata includes proper timestamps."""
        storage = StorageService(base_dir=tmp_path / "sessions")
        session_id = str(uuid4())
        storage.create_session(session_id)

        now = datetime.now()
        session_data = {
            "id": session_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        storage.save_metadata(session_id, "session.json", session_data)
        loaded_data = storage.load_metadata(session_id, "session.json")

        assert "created_at" in loaded_data
        assert "updated_at" in loaded_data

    def test_storage_directory_permissions(self, tmp_path):
        """Test that storage directories are created with correct permissions."""
        storage = StorageService(base_dir=tmp_path / "sessions")
        session_id = str(uuid4())
        session_path = storage.create_session(session_id)

        # Verify directories are readable and writable
        assert session_path.exists()
        assert (session_path / "frames").exists()

        # Test write permission by creating a file
        test_file = session_path / "frames" / "test.txt"
        test_file.write_text("test")
        assert test_file.read_text() == "test"
