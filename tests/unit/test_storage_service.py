"""Unit tests for storage service.

TDD: These tests are written FIRST and must FAIL before implementation.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from src.services.storage_service import StorageService


class TestStorageService:
    """Test storage service for persisting video sessions and data."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create storage service with temporary directory."""
        return StorageService(base_dir=tmp_path)

    def test_create_session_directory(self, storage):
        """Test creating a new session directory."""
        session_id = "test_session_001"

        session_path = storage.create_session(session_id)

        assert session_path.exists()
        assert session_path.is_dir()
        assert session_path.name == session_id

    def test_session_directory_has_subdirectories(self, storage):
        """Test session directory contains required subdirectories."""
        session_id = "test_session_002"

        session_path = storage.create_session(session_id)

        assert (session_path / "frames").exists()
        assert (session_path / "masks").exists()
        assert (session_path / "metadata").exists()

    def test_save_json_metadata(self, storage):
        """Test saving JSON metadata to session."""
        session_id = "test_session_003"
        storage.create_session(session_id)

        metadata = {
            "video_path": "/path/to/video.mp4",
            "frame_count": 100,
            "fps": 30.0,
            "processed_at": datetime.now().isoformat(),
        }

        storage.save_metadata(session_id, "video_info.json", metadata)

        saved_path = storage.get_session_path(session_id) / "metadata" / "video_info.json"
        assert saved_path.exists()

        with open(saved_path) as f:
            loaded = json.load(f)
        assert loaded["frame_count"] == 100
        assert loaded["fps"] == 30.0

    def test_load_json_metadata(self, storage):
        """Test loading JSON metadata from session."""
        session_id = "test_session_004"
        storage.create_session(session_id)

        original_data = {"test_key": "test_value", "number": 42}
        storage.save_metadata(session_id, "test.json", original_data)

        loaded_data = storage.load_metadata(session_id, "test.json")

        assert loaded_data == original_data

    def test_get_session_path(self, storage):
        """Test retrieving session path."""
        session_id = "test_session_005"
        storage.create_session(session_id)

        session_path = storage.get_session_path(session_id)

        assert session_path.exists()
        assert session_path.name == session_id

    def test_get_nonexistent_session_raises_error(self, storage):
        """Test retrieving non-existent session raises error."""
        with pytest.raises(FileNotFoundError):
            storage.get_session_path("nonexistent_session")

    def test_list_sessions(self, storage):
        """Test listing all sessions."""
        storage.create_session("session_a")
        storage.create_session("session_b")
        storage.create_session("session_c")

        sessions = storage.list_sessions()

        assert len(sessions) == 3
        assert "session_a" in sessions
        assert "session_b" in sessions
        assert "session_c" in sessions

    def test_delete_session(self, storage):
        """Test deleting a session."""
        session_id = "session_to_delete"
        storage.create_session(session_id)
        session_path = storage.get_session_path(session_id)

        storage.delete_session(session_id)

        assert not session_path.exists()

    def test_session_exists(self, storage):
        """Test checking if session exists."""
        session_id = "existing_session"
        storage.create_session(session_id)

        assert storage.session_exists(session_id) is True
        assert storage.session_exists("nonexistent") is False

    def test_save_frame_to_session(self, storage, tmp_path):
        """Test saving a frame image to session."""
        import numpy as np

        session_id = "frame_test_session"
        storage.create_session(session_id)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_idx = 42

        storage.save_frame(session_id, frame_idx, frame)

        frame_path = storage.get_session_path(session_id) / "frames" / f"frame_{frame_idx:06d}.jpg"
        assert frame_path.exists()

    def test_load_frame_from_session(self, storage):
        """Test loading a frame image from session."""
        import numpy as np

        session_id = "load_frame_session"
        storage.create_session(session_id)

        original_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_idx = 10
        storage.save_frame(session_id, frame_idx, original_frame)

        loaded_frame = storage.load_frame(session_id, frame_idx)

        assert loaded_frame.shape == original_frame.shape
        assert loaded_frame.dtype == original_frame.dtype

    def test_get_masks_path(self, storage):
        """Test retrieving masks directory path."""
        session_id = "masks_path_session"
        storage.create_session(session_id)

        masks_path = storage.get_masks_path(session_id)

        assert masks_path.exists()
        assert masks_path.is_dir()
        assert masks_path.name == "masks"

    def test_create_duplicate_session_raises_error(self, storage):
        """Test creating duplicate session raises error."""
        session_id = "duplicate_session"
        storage.create_session(session_id)

        with pytest.raises(ValueError, match="already exists"):
            storage.create_session(session_id)
