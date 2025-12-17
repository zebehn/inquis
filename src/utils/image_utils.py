"""Image utilities for loading, saving, and transforming images.

Provides functions for:
- Loading/saving images with OpenCV
- RGB/BGR conversion
- Resizing with aspect ratio preservation
- Normalization and denormalization
- PyTorch tensor conversion
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from file path and convert to RGB.

    Args:
        path: Path to image file

    Returns:
        Image as numpy array in RGB format (H, W, C)

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Load with OpenCV (loads as BGR)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """Save image to file path.

    Args:
        image: Image array in RGB format (H, W, C)
        path: Destination file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), image_bgr)


def resize_image(
    image: np.ndarray, target_size: Tuple[int, int], interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resize image to target size.

    Args:
        image: Input image (H, W, C)
        target_size: Target (height, width)
        interpolation: OpenCV interpolation method

    Returns:
        Resized image
    """
    target_h, target_w = target_size

    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image values from [0, 255] to [0, 1].

    Args:
        image: Input image with uint8 values

    Returns:
        Normalized image with float32 values in [0, 1]
    """
    normalized = image.astype(np.float32) / 255.0

    return normalized


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image values from [0, 1] to [0, 255].

    Args:
        image: Normalized image with float32 values in [0, 1]

    Returns:
        Image with uint8 values in [0, 255]
    """
    denormalized = (image * 255.0).clip(0, 255).astype(np.uint8)

    return denormalized


def image_to_tensor(image: np.ndarray) -> np.ndarray:
    """Convert numpy image (H, W, C) to tensor format (C, H, W).

    Also normalizes to [0, 1] range.

    Args:
        image: Input image in (H, W, C) format

    Returns:
        Image in (C, H, W) format with float32 values
    """
    # Normalize if needed
    if image.dtype == np.uint8:
        image = normalize_image(image)

    # Transpose from (H, W, C) to (C, H, W)
    tensor = np.transpose(image, (2, 0, 1))

    return tensor


def tensor_to_image(tensor: np.ndarray) -> np.ndarray:
    """Convert tensor format (C, H, W) to numpy image (H, W, C).

    Also denormalizes to [0, 255] range.

    Args:
        tensor: Input in (C, H, W) format with float32 values

    Returns:
        Image in (H, W, C) format with uint8 values
    """
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(tensor, (1, 2, 0))

    # Denormalize if needed
    if image.dtype == np.float32:
        image = denormalize_image(image)

    return image
