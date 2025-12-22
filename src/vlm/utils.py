"""Utility functions for VLM module.

This module provides helper functions for image processing,
encoding, and validation.
"""

import base64
from pathlib import Path
from typing import List


# Supported image formats
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".webp", ".gif"]


def encode_image_base64(image_path: Path) -> str:
    """Encode image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64-encoded image string

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not validate_image_format(image_path):
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def validate_image_format(image_path: Path) -> bool:
    """Validate that image file has supported format.

    Args:
        image_path: Path to image file

    Returns:
        True if format is supported, False otherwise
    """
    return image_path.suffix.lower() in SUPPORTED_FORMATS


def create_data_uri(image_path: Path) -> str:
    """Create data URI for image.

    Args:
        image_path: Path to image file

    Returns:
        Data URI string (e.g., "data:image/jpeg;base64,...")

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
    encoded = encode_image_base64(image_path)

    # Determine MIME type from extension
    ext = image_path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")

    return f"data:{mime_type};base64,{encoded}"
