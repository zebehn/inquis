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

    def save_video_session(self, session: "VideoSession") -> None:
        """Save VideoSession model to storage.

        Args:
            session: VideoSession model instance
        """
        from src.models.video_session import VideoSession

        session_data = session.model_dump(mode='json')
        self.save_metadata(str(session.id), "session.json", session_data)

    def load_video_session(self, session_id: str) -> "VideoSession":
        """Load VideoSession model from storage.

        Args:
            session_id: Session identifier

        Returns:
            VideoSession model instance
        """
        from src.models.video_session import VideoSession

        session_data = self.load_metadata(session_id, "session.json")
        return VideoSession(**session_data)

    def save_segmentation_frame(
        self, session_id: str, frame: "SegmentationFrame"
    ) -> None:
        """Save SegmentationFrame model to storage.

        Args:
            session_id: Session identifier
            frame: SegmentationFrame model instance
        """
        from src.models.segmentation_frame import SegmentationFrame

        frame_data = frame.model_dump(mode='json')
        filename = f"frame_{frame.frame_index:06d}.json"
        self.save_metadata(session_id, filename, frame_data)

    def load_segmentation_frame(
        self, session_id: str, frame_index: int
    ) -> "SegmentationFrame":
        """Load SegmentationFrame model from storage.

        Args:
            session_id: Session identifier
            frame_index: Frame index

        Returns:
            SegmentationFrame model instance
        """
        from src.models.segmentation_frame import SegmentationFrame

        filename = f"frame_{frame_index:06d}.json"
        frame_data = self.load_metadata(session_id, filename)
        return SegmentationFrame(**frame_data)

    def list_segmentation_frames(self, session_id: str) -> List[int]:
        """List all segmentation frame indices for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of frame indices
        """
        session_path = self.get_session_path(session_id)
        metadata_path = session_path / "metadata"

        frame_indices = []
        for file_path in metadata_path.glob("frame_*.json"):
            # Extract frame index from filename like "frame_000042.json"
            frame_idx = int(file_path.stem.split("_")[1])
            frame_indices.append(frame_idx)

        return sorted(frame_indices)

    def save_image(self, session_id: str, relative_path: str, image: np.ndarray) -> Path:
        """Save an image to session storage.

        TDD: T053 [US2] - Add UncertainRegion persistence methods to StorageService

        Args:
            session_id: Session identifier
            relative_path: Relative path within session (e.g., "uncertain/region_001.jpg")
            image: Image array (H, W, 3) in RGB format

        Returns:
            Path where image was saved
        """
        session_path = self.get_session_path(session_id)
        image_path = session_path / relative_path

        # Create parent directories if needed
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image_bgr)

        return image_path

    def save_mask(self, session_id: str, relative_path: str, mask: np.ndarray) -> Path:
        """Save a mask to session storage.

        TDD: T053 [US2] - Add UncertainRegion persistence methods to StorageService

        Args:
            session_id: Session identifier
            relative_path: Relative path within session (e.g., "uncertain/region_001_mask.npz")
            mask: Binary mask array (H, W) boolean

        Returns:
            Path where mask was saved
        """
        session_path = self.get_session_path(session_id)
        mask_path = session_path / relative_path

        # Create parent directories if needed
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        # Save mask as compressed npz
        np.savez_compressed(mask_path, mask=mask)

        return mask_path

    def load_image(self, session_id: str, relative_path: str) -> np.ndarray:
        """Load an image from session storage.

        Args:
            session_id: Session identifier
            relative_path: Relative path within session

        Returns:
            Image array (H, W, 3) in RGB format
        """
        session_path = self.get_session_path(session_id)
        image_path = session_path / relative_path

        # Load with OpenCV and convert BGR to RGB
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return image

    def load_mask(self, session_id: str, relative_path: str) -> np.ndarray:
        """Load a mask from session storage.

        Args:
            session_id: Session identifier
            relative_path: Relative path within session

        Returns:
            Binary mask array (H, W) boolean
        """
        session_path = self.get_session_path(session_id)
        mask_path = session_path / relative_path

        # Load mask from npz
        data = np.load(mask_path)
        mask = data["mask"]

        return mask

    def save_uncertain_region(
        self, session_id: str, region: "UncertainRegion"
    ) -> None:
        """Save UncertainRegion model to storage.

        TDD: T053 [US2] - Add UncertainRegion persistence methods to StorageService

        Args:
            session_id: Session identifier
            region: UncertainRegion model instance
        """
        from src.models.uncertain_region import UncertainRegion

        region_data = region.model_dump(mode='json')
        filename = f"uncertain_region_{region.id}.json"

        # Create uncertain directory if needed
        session_path = self.get_session_path(session_id)
        uncertain_dir = session_path / "uncertain"
        uncertain_dir.mkdir(exist_ok=True)

        # Save metadata
        uncertain_metadata_path = uncertain_dir / filename
        with open(uncertain_metadata_path, "w") as f:
            json.dump(region_data, f, indent=2)

    def load_uncertain_region(
        self, session_id: str, region_id: str
    ) -> "UncertainRegion":
        """Load UncertainRegion model from storage.

        Args:
            session_id: Session identifier
            region_id: Region UUID

        Returns:
            UncertainRegion model instance
        """
        from src.models.uncertain_region import UncertainRegion

        session_path = self.get_session_path(session_id)
        filename = f"uncertain_region_{region_id}.json"
        region_path = session_path / "uncertain" / filename

        with open(region_path) as f:
            region_data = json.load(f)

        return UncertainRegion(**region_data)

    def list_uncertain_regions(self, session_id: str) -> List[str]:
        """List all uncertain region IDs for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of region UUIDs
        """
        session_path = self.get_session_path(session_id)
        uncertain_dir = session_path / "uncertain"

        if not uncertain_dir.exists():
            return []

        region_ids = []
        for file_path in uncertain_dir.glob("uncertain_region_*.json"):
            # Extract region ID from filename like "uncertain_region_<uuid>.json"
            region_id = file_path.stem.replace("uncertain_region_", "")
            region_ids.append(region_id)

        return sorted(region_ids)

    # T087-T089: VLM Query persistence and semantic label replacement
    def save_vlm_query(self, session_id: str, query: "VLMQuery") -> None:
        """Save VLMQuery to session metadata.

        TDD: T087 [US3] - Add VLMQuery persistence to StorageService

        Args:
            session_id: Session identifier
            query: VLMQuery model instance
        """
        from src.models.vlm_query import VLMQuery

        query_data = query.model_dump(mode='json')

        # Create vlm_queries directory if needed
        session_path = self.get_session_path(session_id)
        vlm_queries_dir = session_path / "metadata" / "vlm_queries"
        vlm_queries_dir.mkdir(parents=True, exist_ok=True)

        # Save query as JSON
        query_file = vlm_queries_dir / f"{query.id}.json"
        with open(query_file, "w") as f:
            json.dump(query_data, f, indent=2)

    def load_vlm_query(self, session_id: str, query_id: Any) -> "VLMQuery":
        """Load VLMQuery from session metadata.

        TDD: T087 [US3] - Load VLMQuery from storage

        Args:
            session_id: Session identifier
            query_id: VLMQuery UUID

        Returns:
            VLMQuery model instance
        """
        from src.models.vlm_query import VLMQuery
        from uuid import UUID

        session_path = self.get_session_path(session_id)
        query_file = session_path / "metadata" / "vlm_queries" / f"{query_id}.json"

        with open(query_file) as f:
            query_data = json.load(f)

        return VLMQuery(**query_data)

    def load_all_vlm_queries(self, session_id: str) -> List["VLMQuery"]:
        """Load all VLM queries for a session.

        TDD: T087 [US3] - Load all VLMQueries for session

        Args:
            session_id: Session identifier

        Returns:
            List of VLMQuery instances
        """
        from src.models.vlm_query import VLMQuery

        session_path = self.get_session_path(session_id)
        vlm_queries_dir = session_path / "metadata" / "vlm_queries"

        if not vlm_queries_dir.exists():
            return []

        queries = []
        for query_file in vlm_queries_dir.glob("*.json"):
            with open(query_file) as f:
                query_data = json.load(f)
            queries.append(VLMQuery(**query_data))

        return queries

    def apply_semantic_label(
        self, region: "UncertainRegion", vlm_query: "VLMQuery"
    ) -> "UncertainRegion":
        """Apply semantic label to UncertainRegion from VLMQuery.

        Replaces generic instance ID (object_N) with confirmed semantic label.

        TDD: T088 [US3] - Implement semantic label replacement logic

        Args:
            region: UncertainRegion to update
            vlm_query: VLMQuery containing confirmed label

        Returns:
            Updated UncertainRegion with semantic label
        """
        from src.models.uncertain_region import RegionStatus

        # Get final label (manual if provided, otherwise VLM label)
        final_label = vlm_query.get_final_label()

        # Update region with semantic label
        region.confirmed_label = final_label
        region.status = RegionStatus.CONFIRMED
        region.vlm_query_id = vlm_query.id
        region.reviewed_at = vlm_query.responded_at

        return region

    def get_vlm_statistics(self, session_id: str) -> Dict[str, Any]:
        """Calculate VLM usage statistics for a session.

        TDD: T089 [US3] - Add VLM statistics calculation

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with VLM usage statistics
        """
        from src.models.vlm_query import VLMQueryStatus

        queries = self.load_all_vlm_queries(session_id)

        stats = {
            "total_queries": len(queries),
            "successful_queries": 0,
            "uncertain_queries": 0,
            "failed_queries": 0,
            "rate_limited_queries": 0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "average_latency": 0.0,
            "average_confidence": 0.0,
        }

        if not queries:
            return stats

        total_latency = 0.0
        total_confidence = 0.0
        confidence_count = 0

        for query in queries:
            # Count by status
            if query.status == VLMQueryStatus.SUCCESS:
                stats["successful_queries"] += 1
            elif query.status == VLMQueryStatus.VLM_UNCERTAIN:
                stats["uncertain_queries"] += 1
            elif query.status == VLMQueryStatus.FAILED:
                stats["failed_queries"] += 1
            elif query.status == VLMQueryStatus.RATE_LIMITED:
                stats["rate_limited_queries"] += 1

            # Sum costs and tokens
            stats["total_cost"] += query.cost
            stats["total_tokens"] += query.token_count

            # Sum latency
            total_latency += query.latency

            # Sum confidence (if available)
            confidence = query.get_confidence()
            if confidence > 0:
                total_confidence += confidence
                confidence_count += 1

        # Calculate averages
        stats["average_latency"] = total_latency / len(queries)
        if confidence_count > 0:
            stats["average_confidence"] = total_confidence / confidence_count

        return stats

    # T008-T010: Semantic Labeling Job persistence with atomic checkpointing

    def checkpoint_labeling_job(self, session_id: str, job: "SemanticLabelingJob") -> None:
        """Save labeling job checkpoint with atomic write-then-rename pattern.

        Uses write-then-rename pattern for atomic checkpointing:
        1. Write to temporary file
        2. fsync() to force disk write
        3. Atomic rename to final checkpoint file

        This ensures no partial writes on crash (old checkpoint remains intact).

        TDD: T008 - Extend StorageService with atomic checkpoint methods

        Args:
            session_id: Session identifier
            job: SemanticLabelingJob to checkpoint
        """
        import os
        import tempfile
        from src.models.semantic_labeling_job import SemanticLabelingJob

        session_path = self.get_session_path(session_id)
        checkpoints_dir = session_path / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        # Final checkpoint path
        checkpoint_path = checkpoints_dir / f"job_{job.id}.json"

        # Serialize job to JSON
        job_data = job.model_dump(mode='json')

        # Write to temporary file first (atomic write-then-rename pattern)
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=checkpoints_dir,
            delete=False,
            suffix='.tmp'
        ) as temp_file:
            json.dump(job_data, temp_file, indent=2)
            temp_file.flush()
            # Force write to disk (survives system crash)
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)

        # Atomic rename (POSIX atomic operation)
        temp_path.replace(checkpoint_path)

        # Rotate backup checkpoints (keep last 3)
        self._rotate_checkpoint_backups(checkpoints_dir, job.id, max_backups=3)

    def load_job_checkpoint(self, session_id: str, job_id: Any) -> "SemanticLabelingJob":
        """Load labeling job from checkpoint.

        TDD: T008 - Load job checkpoint

        Args:
            session_id: Session identifier
            job_id: Job UUID

        Returns:
            SemanticLabelingJob instance

        Raises:
            FileNotFoundError: If checkpoint not found
        """
        from src.models.semantic_labeling_job import SemanticLabelingJob

        session_path = self.get_session_path(session_id)
        checkpoint_path = session_path / "checkpoints" / f"job_{job_id}.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Job checkpoint not found: {job_id}")

        with open(checkpoint_path) as f:
            job_data = json.load(f)

        return SemanticLabelingJob(**job_data)

    def _rotate_checkpoint_backups(
        self, checkpoints_dir: Path, job_id: Any, max_backups: int = 3
    ) -> None:
        """Rotate checkpoint backups, keeping only the most recent.

        Creates numbered backups (job_{id}_backup_1.json, job_{id}_backup_2.json, etc.)
        and removes old backups beyond max_backups limit.

        Args:
            checkpoints_dir: Directory containing checkpoints
            job_id: Job UUID
            max_backups: Maximum number of backups to keep (default 3)
        """
        checkpoint_path = checkpoints_dir / f"job_{job_id}.json"

        if not checkpoint_path.exists():
            return

        # Rotate existing backups (3 -> 4, 2 -> 3, 1 -> 2)
        for i in range(max_backups - 1, 0, -1):
            backup_path = checkpoints_dir / f"job_{job_id}_backup_{i}.json"
            next_backup_path = checkpoints_dir / f"job_{job_id}_backup_{i + 1}.json"
            if backup_path.exists():
                backup_path.replace(next_backup_path)

        # Copy current checkpoint to backup_1
        backup_1_path = checkpoints_dir / f"job_{job_id}_backup_1.json"
        if checkpoint_path.exists():
            shutil.copy2(checkpoint_path, backup_1_path)

        # Remove backups beyond max_backups limit
        for i in range(max_backups + 1, max_backups + 10):  # Check up to 10 extra
            old_backup_path = checkpoints_dir / f"job_{job_id}_backup_{i}.json"
            if old_backup_path.exists():
                old_backup_path.unlink()

    def save_job(self, session_id: str, job: "SemanticLabelingJob") -> None:
        """Save labeling job (non-atomic, for initial creation).

        Use checkpoint_labeling_job() for atomic updates during job execution.

        TDD: T009 - Add job persistence methods

        Args:
            session_id: Session identifier
            job: SemanticLabelingJob to save
        """
        from src.models.semantic_labeling_job import SemanticLabelingJob

        session_path = self.get_session_path(session_id)
        checkpoints_dir = session_path / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        job_data = job.model_dump(mode='json')
        job_path = checkpoints_dir / f"job_{job.id}.json"

        with open(job_path, "w") as f:
            json.dump(job_data, f, indent=2)

    def load_job(self, session_id: str, job_id: Any) -> "SemanticLabelingJob":
        """Load labeling job.

        TDD: T009 - Load job from storage

        Args:
            session_id: Session identifier
            job_id: Job UUID

        Returns:
            SemanticLabelingJob instance
        """
        return self.load_job_checkpoint(session_id, job_id)

    def list_jobs(self, session_id: str) -> List[str]:
        """List all job IDs for a session.

        TDD: T009 - List jobs for session

        Args:
            session_id: Session identifier

        Returns:
            List of job UUIDs
        """
        session_path = self.get_session_path(session_id)
        checkpoints_dir = session_path / "checkpoints"

        if not checkpoints_dir.exists():
            return []

        job_ids = []
        for file_path in checkpoints_dir.glob("job_*.json"):
            # Extract job ID from filename like "job_<uuid>.json" (skip backups)
            if "_backup_" not in file_path.name:
                job_id = file_path.stem.replace("job_", "")
                job_ids.append(job_id)

        return sorted(job_ids)

    def save_pattern(self, session_id: str, pattern: "SemanticUncertaintyPattern") -> None:
        """Save semantic uncertainty pattern.

        TDD: T010 - Add pattern persistence methods

        Args:
            session_id: Session identifier
            pattern: SemanticUncertaintyPattern to save
        """
        from src.models.semantic_uncertainty import SemanticUncertaintyPattern

        session_path = self.get_session_path(session_id)
        patterns_dir = session_path / "patterns"
        patterns_dir.mkdir(exist_ok=True)

        pattern_data = pattern.model_dump(mode='json')
        pattern_path = patterns_dir / f"pattern_{pattern.id}.json"

        with open(pattern_path, "w") as f:
            json.dump(pattern_data, f, indent=2)

    def load_pattern(self, session_id: str, pattern_id: Any) -> "SemanticUncertaintyPattern":
        """Load semantic uncertainty pattern.

        TDD: T010 - Load pattern from storage

        Args:
            session_id: Session identifier
            pattern_id: Pattern UUID

        Returns:
            SemanticUncertaintyPattern instance
        """
        from src.models.semantic_uncertainty import SemanticUncertaintyPattern

        session_path = self.get_session_path(session_id)
        pattern_path = session_path / "patterns" / f"pattern_{pattern_id}.json"

        with open(pattern_path) as f:
            pattern_data = json.load(f)

        return SemanticUncertaintyPattern(**pattern_data)

    def list_patterns(self, session_id: str) -> List[str]:
        """List all pattern IDs for a session.

        TDD: T010 - List patterns for session

        Args:
            session_id: Session identifier

        Returns:
            List of pattern UUIDs
        """
        session_path = self.get_session_path(session_id)
        patterns_dir = session_path / "patterns"

        if not patterns_dir.exists():
            return []

        pattern_ids = []
        for file_path in patterns_dir.glob("pattern_*.json"):
            # Extract pattern ID from filename like "pattern_<uuid>.json"
            pattern_id = file_path.stem.replace("pattern_", "")
            pattern_ids.append(pattern_id)

        return sorted(pattern_ids)
