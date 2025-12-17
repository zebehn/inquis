"""Storage service for persisting video sessions and data.

Provides methods for:
- Creating and managing session directories
- Saving/loading JSON metadata
- Persisting frames and masks
- Session lifecycle management
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
import cv2


class StorageService:
    """Service for managing file-based storage of video processing sessions."""

    def __init__(self, base_dir: Union[str, Path] = "./data/sessions"):
        """Initialize storage service.

        Args:
            base_dir: Base directory for all sessions
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, session_id: str) -> Path:
        """Create a new session directory with subdirectories.

        Args:
            session_id: Unique session identifier

        Returns:
            Path to created session directory

        Raises:
            ValueError: If session already exists
        """
        session_path = self.base_dir / session_id

        if session_path.exists():
            raise ValueError(f"Session '{session_id}' already exists")

        # Create session directory structure
        session_path.mkdir(parents=True, exist_ok=False)
        (session_path / "frames").mkdir()
        (session_path / "masks").mkdir()
        (session_path / "metadata").mkdir()

        return session_path

    def get_session_path(self, session_id: str) -> Path:
        """Get path to existing session directory.

        Args:
            session_id: Session identifier

        Returns:
            Path to session directory

        Raises:
            FileNotFoundError: If session doesn't exist
        """
        session_path = self.base_dir / session_id

        if not session_path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found")

        return session_path

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists
        """
        session_path = self.base_dir / session_id
        return session_path.exists()

    def save_metadata(self, session_id: str, filename: str, data: Dict[str, Any]) -> None:
        """Save JSON metadata to session.

        Args:
            session_id: Session identifier
            filename: Metadata filename (should end with .json)
            data: Dictionary to save as JSON
        """
        session_path = self.get_session_path(session_id)
        metadata_path = session_path / "metadata" / filename

        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_metadata(self, session_id: str, filename: str) -> Dict[str, Any]:
        """Load JSON metadata from session.

        Args:
            session_id: Session identifier
            filename: Metadata filename

        Returns:
            Loaded JSON data as dictionary
        """
        session_path = self.get_session_path(session_id)
        metadata_path = session_path / "metadata" / filename

        with open(metadata_path) as f:
            data = json.load(f)

        return data

    def list_sessions(self) -> List[str]:
        """List all session IDs.

        Returns:
            List of session identifiers
        """
        if not self.base_dir.exists():
            return []

        sessions = [
            d.name for d in self.base_dir.iterdir() if d.is_dir()
        ]

        return sorted(sessions)

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its data.

        Args:
            session_id: Session identifier
        """
        session_path = self.get_session_path(session_id)
        shutil.rmtree(session_path)

    def save_frame(self, session_id: str, frame_idx: int, frame: np.ndarray) -> None:
        """Save a frame image to session.

        Args:
            session_id: Session identifier
            frame_idx: Frame index
            frame: Frame image array (H, W, 3) in RGB
        """
        session_path = self.get_session_path(session_id)
        frame_path = session_path / "frames" / f"frame_{frame_idx:06d}.jpg"

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)

    def load_frame(self, session_id: str, frame_idx: int) -> np.ndarray:
        """Load a frame image from session.

        Args:
            session_id: Session identifier
            frame_idx: Frame index

        Returns:
            Frame image array (H, W, 3) in RGB
        """
        session_path = self.get_session_path(session_id)
        frame_path = session_path / "frames" / f"frame_{frame_idx:06d}.jpg"

        # Load with OpenCV and convert BGR to RGB
        frame_bgr = cv2.imread(str(frame_path))
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        return frame

    def get_masks_path(self, session_id: str) -> Path:
        """Get path to masks directory for session.

        Args:
            session_id: Session identifier

        Returns:
            Path to masks directory
        """
        session_path = self.get_session_path(session_id)
        return session_path / "masks"
