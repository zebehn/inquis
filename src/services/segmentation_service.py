"""SegmentationService for SAM2-based instance segmentation."""

import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import logging
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Result from segmenting a single frame.

    Attributes:
        masks: Boolean masks (N, H, W) where N is number of instances
        confidences: Confidence scores for each mask (N,)
        class_labels: Class labels for each mask (N,)
        bboxes: Bounding boxes in (x, y, width, height) format (N, 4)
    """

    masks: np.ndarray
    confidences: np.ndarray
    class_labels: List[str]
    bboxes: List[Tuple[int, int, int, int]]


class SegmentationService:
    """Service for SAM2-based instance segmentation.

    Note: This is a mock implementation that defines the interface.
    Actual SAM2 integration will be added when models are downloaded.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: str,
        device: str = "cuda",
        confidence_threshold: float = 0.75,
        uncertainty_threshold: float = 0.25,
    ):
        """Initialize segmentation service.

        Args:
            checkpoint_path: Path to SAM2 checkpoint file
            config: SAM2 config name
            device: Device to run inference on (cuda, cpu, mps)
            confidence_threshold: Minimum confidence for keeping detections
            uncertainty_threshold: Threshold for flagging uncertain regions
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.enable_mixed_precision = True
        self.model = None

        logger.info(
            f"Initialized SegmentationService with device={device}, "
            f"confidence_threshold={confidence_threshold}"
        )

    def load_model(self) -> None:
        """Load SAM2 model from checkpoint.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}"
            )

        logger.info(f"Loading SAM2 model from {self.checkpoint_path}")
        logger.info(f"Using device: {self.device}")

        # Build SAM2 model
        sam2_model = build_sam2(
            config_file=self.config,
            ckpt_path=str(self.checkpoint_path),
            device=self.device,
        )

        # Create automatic mask generator
        self.model = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=self.confidence_threshold,
            stability_score_thresh=0.92,
            stability_score_offset=1.0,
            crop_n_layers=1,
            box_nms_thresh=0.7,
        )

        logger.info("SAM2 model loaded successfully")

    def segment_frame(self, image: np.ndarray) -> SegmentationResult:
        """Segment a single frame using SAM2.

        Args:
            image: Input image (H, W, 3) in RGB format

        Returns:
            SegmentationResult with masks, confidences, labels, and bboxes
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        h, w = image.shape[:2]

        # Run SAM2 automatic mask generation
        logger.debug(f"Running SAM2 inference on image of shape {image.shape}")
        mask_data = self.model.generate(image)

        # If no masks detected, return empty results
        if len(mask_data) == 0:
            logger.debug("No masks detected")
            return SegmentationResult(
                masks=np.zeros((0, h, w), dtype=bool),
                confidences=np.array([]),
                class_labels=[],
                bboxes=[],
            )

        # Extract masks, confidences, and bboxes from SAM2 output
        masks = []
        confidences = []
        bboxes = []
        class_labels = []

        for i, mask_dict in enumerate(mask_data):
            # Extract mask (H, W) boolean array
            mask = mask_dict["segmentation"]
            masks.append(mask)

            # Use predicted_iou as confidence score
            confidence = mask_dict["predicted_iou"]
            confidences.append(confidence)

            # Extract bbox in (x, y, w, h) format
            bbox = mask_dict["bbox"]
            x, y, w_box, h_box = bbox
            bboxes.append((int(x), int(y), int(w_box), int(h_box)))

            # SAM2 doesn't provide semantic labels, use generic label
            class_labels.append(f"object_{i+1}")

        # Convert to numpy arrays
        masks = np.array(masks, dtype=bool)  # (N, H, W)
        confidences = np.array(confidences, dtype=np.float32)

        logger.debug(f"Detected {len(masks)} instances")

        return SegmentationResult(
            masks=masks,
            confidences=confidences,
            class_labels=class_labels,
            bboxes=bboxes,
        )

    def segment_batch(self, images: List[np.ndarray]) -> List[SegmentationResult]:
        """Segment a batch of frames.

        Args:
            images: List of input images

        Returns:
            List of SegmentationResult for each image
        """
        return [self.segment_frame(image) for image in images]

    def compute_uncertainty(self, logits: np.ndarray) -> np.ndarray:
        """Compute uncertainty scores from model logits.

        Uses entropy-based uncertainty: H = -sum(p * log(p))

        Args:
            logits: Model logits (N, H, W) or (N, num_classes, H, W)

        Returns:
            Uncertainty scores for each instance (N,)
        """
        # Convert logits to probabilities
        probs = self._softmax(logits)

        # Compute entropy
        epsilon = 1e-10  # Avoid log(0)
        entropy = -np.sum(probs * np.log(probs + epsilon), axis=-1)

        # Normalize to [0, 1]
        max_entropy = np.log(probs.shape[-1])
        uncertainty = entropy / max_entropy

        # Average over spatial dimensions if needed
        if uncertainty.ndim > 1:
            return uncertainty.mean(axis=tuple(range(1, uncertainty.ndim)))
        return uncertainty

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def detect_uncertain_regions(
        self, confidences: np.ndarray, uncertainty_threshold: float
    ) -> List[int]:
        """Detect regions with high uncertainty.

        Args:
            confidences: Confidence scores for each mask
            uncertainty_threshold: Threshold value - regions with confidence below this are uncertain

        Returns:
            Indices of uncertain regions
        """
        # Flag regions where confidence is below threshold
        uncertain_indices = np.where(confidences < uncertainty_threshold)[0].tolist()

        return uncertain_indices

    def extract_bboxes(
        self, masks: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Extract bounding boxes from binary masks.

        Args:
            masks: Boolean masks (N, H, W)

        Returns:
            List of bounding boxes in (x, y, width, height) format
        """
        from src.utils.mask_utils import get_mask_bbox

        bboxes = []
        for mask in masks:
            bbox = get_mask_bbox(mask)
            if bbox is not None:
                # Convert from (y_min, x_min, y_max, x_max) to (x, y, w, h)
                y_min, x_min, y_max, x_max = bbox
                x, y = x_min, y_min
                w, h = x_max - x_min, y_max - y_min
                bboxes.append((x, y, w, h))
            else:
                bboxes.append((0, 0, 0, 0))  # Empty mask

        return bboxes
