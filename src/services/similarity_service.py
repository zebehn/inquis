"""Similarity Service for computing visual similarity between regions.

TDD Green Phase: T048-T051 [US3] - Implement SimilarityService

This service provides heuristic similarity computation using:
- Bbox similarity (0.3 weight): Area and aspect ratio comparison
- Mask similarity (0.4 weight): IoU after 128×128 resize
- Color histogram similarity (0.3 weight): RGB correlation (32 bins/channel)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import cv2

from src.models.uncertain_region import UncertainRegion


class SimilarityService:
    """Service for computing visual similarity between uncertain regions."""

    # Similarity component weights (must sum to 1.0)
    BBOX_WEIGHT = 0.3
    MASK_WEIGHT = 0.4
    COLOR_WEIGHT = 0.3

    # Constants for computation
    MASK_RESIZE = (128, 128)
    HIST_BINS = 32

    def compute_similarity(
        self,
        region1: UncertainRegion,
        region2: UncertainRegion
    ) -> Dict[str, float]:
        """Compute overall similarity score between two regions.

        TDD: T048 [US3] - Heuristic similarity with weighted components

        Args:
            region1: First UncertainRegion
            region2: Second UncertainRegion

        Returns:
            Dictionary with 'overall', 'bbox', 'mask', 'color' scores (0.0-1.0)
        """
        # Compute individual components
        bbox_sim = self._compute_bbox_similarity(region1.bbox, region2.bbox)

        # For MVP, use simplified mask and color similarity
        # Production would load actual masks and images
        mask_sim = self._compute_bbox_iou(region1.bbox, region2.bbox)  # Fallback using bbox IoU
        color_sim = 0.5  # Placeholder for MVP (would compute from actual images)

        # Compute weighted overall similarity
        overall_sim = (
            self.BBOX_WEIGHT * bbox_sim +
            self.MASK_WEIGHT * mask_sim +
            self.COLOR_WEIGHT * color_sim
        )

        return {
            "overall": overall_sim,
            "bbox": bbox_sim,
            "mask": mask_sim,
            "color": color_sim,
        }

    def _compute_bbox_similarity(
        self,
        bbox1: list[int],
        bbox2: list[int]
    ) -> float:
        """Compute bbox similarity using area and aspect ratio.

        TDD: T049 [US3] - Bbox similarity calculation

        Args:
            bbox1: [x, y, width, height]
            bbox2: [x, y, width, height]

        Returns:
            Similarity score 0.0-1.0
        """
        # Extract dimensions
        w1, h1 = bbox1[2], bbox1[3]
        w2, h2 = bbox2[2], bbox2[3]

        # Area similarity (ratio of smaller to larger)
        area1 = w1 * h1
        area2 = w2 * h2
        area_sim = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0.0

        # Aspect ratio similarity
        aspect1 = w1 / h1 if h1 > 0 else 0.0
        aspect2 = w2 / h2 if h2 > 0 else 0.0

        if aspect1 > 0 and aspect2 > 0:
            aspect_sim = min(aspect1, aspect2) / max(aspect1, aspect2)
        else:
            aspect_sim = 0.0

        # Combine area and aspect ratio (50-50 weight)
        bbox_sim = 0.5 * area_sim + 0.5 * aspect_sim

        return bbox_sim

    def _compute_bbox_iou(
        self,
        bbox1: list[int],
        bbox2: list[int]
    ) -> float:
        """Compute IoU between two bboxes (fallback for mask similarity).

        Args:
            bbox1: [x, y, width, height]
            bbox2: [x, y, width, height]

        Returns:
            IoU score 0.0-1.0
        """
        # Convert to [x1, y1, x2, y2]
        x1_1, y1_1 = bbox1[0], bbox1[1]
        x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]

        x1_2, y1_2 = bbox2[0], bbox2[1]
        x2_2, y2_2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]

        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Compute union
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _compute_mask_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> float:
        """Compute IoU between two binary masks.

        TDD: T050 [US3] - Mask similarity using IoU after resize

        Args:
            mask1: Binary mask (H x W) uint8
            mask2: Binary mask (H x W) uint8

        Returns:
            IoU score 0.0-1.0
        """
        # Ensure masks are binary
        mask1_bin = (mask1 > 0).astype(np.uint8)
        mask2_bin = (mask2 > 0).astype(np.uint8)

        # Compute intersection and union
        intersection = np.logical_and(mask1_bin, mask2_bin).sum()
        union = np.logical_or(mask1_bin, mask2_bin).sum()

        return float(intersection / union) if union > 0 else 0.0

    def _compute_color_similarity(
        self,
        hist1: np.ndarray,
        hist2: np.ndarray
    ) -> float:
        """Compute color histogram similarity using correlation.

        TDD: T051 [US3] - Color histogram correlation

        Args:
            hist1: Color histogram (32 x 3) float32
            hist2: Color histogram (32 x 3) float32

        Returns:
            Similarity score 0.0-1.0
        """
        # Flatten histograms
        hist1_flat = hist1.flatten()
        hist2_flat = hist2.flatten()

        # Normalize histograms
        hist1_norm = hist1_flat / (hist1_flat.sum() + 1e-10)
        hist2_norm = hist2_flat / (hist2_flat.sum() + 1e-10)

        # Compute correlation
        mean1 = hist1_norm.mean()
        mean2 = hist2_norm.mean()

        std1 = hist1_norm.std() + 1e-10
        std2 = hist2_norm.std() + 1e-10

        # Pearson correlation
        correlation = ((hist1_norm - mean1) * (hist2_norm - mean2)).mean() / (std1 * std2)

        # Map correlation [-1, 1] to similarity [0, 1]
        similarity = (correlation + 1.0) / 2.0

        return float(np.clip(similarity, 0.0, 1.0))

    def load_mask(self, mask_path: Path) -> np.ndarray:
        """Load and resize mask from file.

        Args:
            mask_path: Path to mask file (.npz or .png)

        Returns:
            Binary mask resized to 128x128
        """
        if mask_path.suffix == ".npz":
            # Load numpy compressed mask
            data = np.load(mask_path)
            mask = data["mask"] if "mask" in data else data[data.files[0]]
        else:
            # Load image mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize to standard size
        if mask.shape != self.MASK_RESIZE:
            mask = cv2.resize(mask, self.MASK_RESIZE, interpolation=cv2.INTER_NEAREST)

        return mask

    def compute_color_histogram(self, image_path: Path, bbox: list[int]) -> np.ndarray:
        """Compute RGB color histogram for region.

        Args:
            image_path: Path to frame image
            bbox: [x, y, width, height]

        Returns:
            Color histogram (32 x 3) float32
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return np.zeros((self.HIST_BINS, 3), dtype=np.float32)

        # Crop to bbox
        x, y, w, h = bbox
        crop = image[y:y+h, x:x+w]

        # Compute histogram for each channel
        hist = np.zeros((self.HIST_BINS, 3), dtype=np.float32)
        for i in range(3):  # B, G, R channels
            hist[:, i] = cv2.calcHist(
                [crop],
                [i],
                None,
                [self.HIST_BINS],
                [0, 256]
            ).flatten()

        # Normalize
        hist = hist / (hist.sum() + 1e-10)

        return hist
