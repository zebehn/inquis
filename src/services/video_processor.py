"""VideoProcessor service for frame extraction and video metadata handling."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video metadata extracted from video file.

    Attributes:
        resolution: (width, height) in pixels
        frame_count: Total number of frames
        duration: Video duration in seconds
        fps: Frames per second
        codec: Video codec name
    """

    resolution: Tuple[int, int]
    frame_count: int
    duration: float
    fps: float
    codec: str


class VideoProcessor:
    """Service for processing video files and extracting frames."""

    def __init__(self):
        """Initialize video processor."""
        self.cap: cv2.VideoCapture | None = None
        self.supported_formats = [".mp4", ".avi", ".mov"]
        self.max_duration = 600.0  # 10 minutes

    def extract_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from video file.

        Args:
            video_path: Path to video file

        Returns:
            VideoMetadata with video properties

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened or is invalid
        """
        video_path = Path(video_path)

        logger.info(f"Extracting metadata from video: {video_path}")

        # Validate file exists
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Validate file format
        if not self.is_supported_format(video_path):
            logger.error(f"Unsupported video format: {video_path.suffix}")
            raise ValueError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported formats: {self.supported_formats}"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        try:
            # Extract properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0.0

            # Get codec (fourcc code)
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

            metadata = VideoMetadata(
                resolution=(width, height),
                frame_count=frame_count,
                duration=duration,
                fps=fps,
                codec=codec.strip(),
            )

            # Validate metadata
            if frame_count <= 0:
                logger.error(f"Invalid frame count: {frame_count}")
                raise ValueError(f"Video has invalid frame count: {frame_count}")

            if fps <= 0:
                logger.error(f"Invalid FPS: {fps}")
                raise ValueError(f"Video has invalid FPS: {fps}")

            # Validate duration against max_duration
            if duration > self.max_duration:
                logger.warning(
                    f"Video duration ({duration:.1f}s) exceeds maximum ({self.max_duration:.1f}s)"
                )
                raise ValueError(
                    f"Video duration ({duration:.1f}s) exceeds maximum allowed "
                    f"duration of {self.max_duration:.1f}s (10 minutes)"
                )

            logger.info(
                f"Video metadata extracted: {frame_count} frames, "
                f"{duration:.1f}s, {fps:.1f} FPS, {width}x{height}"
            )

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise
        finally:
            cap.release()

    def extract_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        """Extract a specific frame by index.

        Args:
            video_path: Path to video file
            frame_index: Frame index (0-indexed)

        Returns:
            Frame as numpy array (H, W, 3) in RGB format

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If frame_index is invalid or frame cannot be read
        """
        video_path = Path(video_path)

        logger.debug(f"Extracting frame {frame_index} from {video_path}")

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Validate frame index
            if frame_index < 0 or frame_index >= total_frames:
                logger.error(
                    f"Invalid frame index {frame_index}. "
                    f"Video has {total_frames} frames."
                )
                raise ValueError(
                    f"Invalid frame index {frame_index}. "
                    f"Must be between 0 and {total_frames - 1}"
                )

            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            # Read frame
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to read frame at index {frame_index}")
                raise ValueError(f"Failed to read frame at index {frame_index}")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            logger.debug(f"Successfully extracted frame {frame_index}")

            return frame_rgb

        except Exception as e:
            logger.error(f"Error extracting frame {frame_index}: {str(e)}")
            raise
        finally:
            cap.release()

    def extract_all_frames(self, video_path: Path) -> Iterator[np.ndarray]:
        """Extract all frames from video as an iterator.

        Args:
            video_path: Path to video file

        Yields:
            Frame as numpy array (H, W, 3) in RGB format

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb

        finally:
            cap.release()

    def extract_frames_batch(
        self, video_path: Path, batch_size: int = 4
    ) -> Iterator[List[np.ndarray]]:
        """Extract frames in batches for efficient processing.

        Args:
            video_path: Path to video file
            batch_size: Number of frames per batch

        Yields:
            Batch of frames as list of numpy arrays

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        batch = []
        for frame in self.extract_all_frames(video_path):
            batch.append(frame)
            if len(batch) == batch_size:
                yield batch
                batch = []

        # Yield remaining frames
        if batch:
            yield batch

    def is_supported_format(self, video_path: Path) -> bool:
        """Check if video format is supported.

        Args:
            video_path: Path to video file

        Returns:
            True if format is supported
        """
        video_path = Path(video_path)
        return video_path.suffix.lower() in self.supported_formats

    def validate_duration(self, metadata: VideoMetadata) -> bool:
        """Validate video duration is within allowed range.

        Args:
            metadata: Video metadata

        Returns:
            True if duration is valid
        """
        return metadata.duration <= self.max_duration

    def calculate_timestamp(self, frame_index: int, fps: float) -> float:
        """Calculate timestamp for a frame index.

        Args:
            frame_index: Frame index (0-indexed)
            fps: Frames per second

        Returns:
            Timestamp in seconds
        """
        return frame_index / fps if fps > 0 else 0.0

    def close(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
