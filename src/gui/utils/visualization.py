"""Visualization utilities for displaying segmentation results."""

import numpy as np
import cv2
from typing import List, Tuple
from src.utils.mask_utils import overlay_masks_on_image, masks_to_colored


def generate_instance_colors(num_instances: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for instance masks.

    Args:
        num_instances: Number of instances

    Returns:
        List of RGB color tuples
    """
    colors = []
    for i in range(num_instances):
        # Use HSV color space for better color distribution
        hue = int((i * 180) / max(num_instances, 1))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(tuple(color_rgb.tolist()))
    return colors


def draw_masks_with_labels(
    image: np.ndarray,
    masks: np.ndarray,
    labels: List[str],
    confidences: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw masks with labels and confidence scores on image.

    Args:
        image: Base image (H, W, 3)
        masks: Boolean masks (N, H, W)
        labels: Class labels for each mask
        confidences: Confidence scores for each mask
        alpha: Transparency for mask overlay

    Returns:
        Image with overlaid masks and labels
    """
    if len(masks) == 0:
        return image.copy()

    # Generate colors for instances
    colors = generate_instance_colors(len(masks))

    # Overlay masks
    result = overlay_masks_on_image(image, masks, colors, alpha=alpha)

    # Draw labels on masks
    from src.utils.mask_utils import get_mask_bbox

    for i, (mask, label, conf) in enumerate(zip(masks, labels, confidences)):
        bbox = get_mask_bbox(mask)
        if bbox is None:
            continue

        y_min, x_min, y_max, x_max = bbox

        # Create label text
        text = f"{label}: {conf:.2f}"

        # Draw background rectangle for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Position at top-left of bbox
        text_x = x_min
        text_y = max(y_min - 5, text_height + 5)

        # Draw white background
        cv2.rectangle(
            result,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            (255, 255, 255),
            -1,
        )

        # Draw text
        cv2.putText(
            result,
            text,
            (text_x, text_y),
            font,
            font_scale,
            colors[i],
            thickness,
            cv2.LINE_AA,
        )

    return result


def draw_bboxes(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    labels: List[str],
    confidences: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image.

    Args:
        image: Base image
        bboxes: List of bounding boxes (x, y, w, h)
        labels: Class labels
        confidences: Confidence scores
        color: Box color
        thickness: Line thickness

    Returns:
        Image with bounding boxes
    """
    result = image.copy()

    for bbox, label, conf in zip(bboxes, labels, confidences):
        x, y, w, h = bbox

        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

        # Draw label
        text = f"{label}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, text_thickness
        )

        # Draw background
        cv2.rectangle(
            result,
            (x, y - text_height - baseline - 5),
            (x + text_width, y),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            result,
            text,
            (x, y - 5),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return result


def create_confidence_heatmap(
    image_shape: Tuple[int, int], masks: np.ndarray, confidences: np.ndarray
) -> np.ndarray:
    """Create confidence heatmap visualization.

    Args:
        image_shape: (height, width)
        masks: Boolean masks (N, H, W)
        confidences: Confidence scores for each mask

    Returns:
        Heatmap image (H, W, 3)
    """
    h, w = image_shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    # For each pixel, use the confidence of the mask covering it
    for mask, conf in zip(masks, confidences):
        heatmap[mask] = conf

    # Convert to color heatmap using matplotlib colormap
    # Normalize to 0-255
    heatmap_normalized = (heatmap * 255).astype(np.uint8)

    # Apply colormap (use OpenCV's COLORMAP_JET)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Convert BGR to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return heatmap_rgb
