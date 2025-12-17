"""Mask utilities for storage, visualization, and manipulation.

Provides functions for:
- Saving/loading masks to .npz format
- Overlaying colored masks on images
- Computing mask IoU
- Filtering and processing masks
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Union


def save_masks(
    masks: np.ndarray, confidences: np.ndarray, path: Union[str, Path]
) -> None:
    """Save masks and confidences to .npz file.

    Args:
        masks: Boolean mask array (N, H, W)
        confidences: Confidence scores array (N,)
        path: Destination .npz file path

    Raises:
        ValueError: If masks array is not 3D
    """
    if masks.ndim != 3:
        raise ValueError(f"Expected 3D array for masks, got shape {masks.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(str(path), masks=masks, confidences=confidences)


def load_masks(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load masks and confidences from .npz file.

    Args:
        path: Path to .npz file

    Returns:
        Tuple of (masks, confidences)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    data = np.load(str(path))
    masks = data["masks"]
    confidences = data["confidences"]

    return masks, confidences


def overlay_masks_on_image(
    image: np.ndarray,
    masks: np.ndarray,
    colors: List[Tuple[int, int, int]],
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay colored masks on image.

    Args:
        image: Base image (H, W, 3)
        masks: Boolean masks (N, H, W)
        colors: RGB colors for each mask
        alpha: Transparency factor [0, 1]

    Returns:
        Image with overlaid masks
    """
    result = image.copy().astype(np.float32)

    for mask, color in zip(masks, colors):
        # Apply color with alpha blending where mask is True
        for c in range(3):
            result[:, :, c] = np.where(
                mask,
                result[:, :, c] * (1 - alpha) + color[c] * alpha,
                result[:, :, c]
            )

    return result.astype(np.uint8)


def masks_to_colored(masks: np.ndarray, colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """Convert masks to colored visualization.

    Args:
        masks: Boolean masks (N, H, W)
        colors: Optional RGB colors for each mask. If None, generates colors.

    Returns:
        Colored mask visualization (H, W, 3)
    """
    h, w = masks.shape[1:3]
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    if colors is None:
        # Generate distinct colors for each mask
        colors = []
        for i in range(len(masks)):
            hue = int((i * 180) / len(masks))
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
            colors.append(tuple(color_rgb))

    for mask, color in zip(masks, colors):
        colored[mask] = color

    return colored


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two masks.

    Args:
        mask1: First boolean mask
        mask2: Second boolean mask

    Returns:
        IoU score [0, 1]
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    iou = intersection / union

    return float(iou)


def filter_masks_by_area(masks: np.ndarray, min_area: int) -> np.ndarray:
    """Filter masks by minimum area.

    Args:
        masks: Boolean masks (N, H, W)
        min_area: Minimum number of pixels

    Returns:
        Filtered masks
    """
    areas = masks.sum(axis=(1, 2))
    valid_indices = areas >= min_area

    filtered = masks[valid_indices]

    return filtered


def get_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Extract bounding box from mask.

    Args:
        mask: Boolean mask (H, W)

    Returns:
        Bounding box (y_min, x_min, y_max, x_max) with exclusive upper bounds,
        or None if mask is empty
    """
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    y_min, y_max = int(y_indices[0]), int(y_indices[-1]) + 1
    x_min, x_max = int(x_indices[0]), int(x_indices[-1]) + 1

    return (y_min, x_min, y_max, x_max)
