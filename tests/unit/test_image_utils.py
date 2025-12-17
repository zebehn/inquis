"""Unit tests for image utilities.

TDD: These tests are written FIRST and must FAIL before implementation.
"""

import pytest
import numpy as np
from pathlib import Path
from src.utils.image_utils import (
    load_image,
    save_image,
    resize_image,
    normalize_image,
    denormalize_image,
    image_to_tensor,
    tensor_to_image,
)


class TestImageUtils:
    """Test image loading, saving, and transformation utilities."""

    def test_load_image_from_path(self, tmp_path):
        """Test loading image from file path."""
        # Create a simple test image (100x100 RGB)
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"

        # Save using cv2 for test setup
        import cv2
        cv2.imwrite(str(image_path), test_image)

        loaded = load_image(image_path)

        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == (100, 100, 3)
        assert loaded.dtype == np.uint8

    def test_load_image_rgb_order(self, tmp_path):
        """Test loaded image is in RGB order (not BGR)."""
        # Create image with distinct RGB channels
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Red channel

        image_path = tmp_path / "red.jpg"
        import cv2
        cv2.imwrite(str(image_path), cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))

        loaded = load_image(image_path)

        # Red channel should be dominant (allow for JPEG compression artifacts)
        assert loaded[0, 0, 0] > 250
        assert loaded[0, 0, 1] < 5
        assert loaded[0, 0, 2] < 5

    def test_load_nonexistent_image_raises_error(self):
        """Test loading non-existent image raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.jpg")

    def test_save_image_to_path(self, tmp_path):
        """Test saving image to file path."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        save_path = tmp_path / "output.jpg"

        save_image(image, save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_resize_image_maintains_aspect_ratio(self):
        """Test resizing image while maintaining aspect ratio."""
        image = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)

        resized = resize_image(image, target_size=(100, 200))

        assert resized.shape[0] == 100
        assert resized.shape[1] == 200

    def test_resize_image_square(self):
        """Test resizing image to square dimensions."""
        image = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)

        resized = resize_image(image, target_size=(224, 224))

        assert resized.shape == (224, 224, 3)

    def test_normalize_image_to_0_1(self):
        """Test normalizing image values to [0, 1] range."""
        image = np.array([[[0, 127, 255]]], dtype=np.uint8)

        normalized = normalize_image(image)

        assert normalized.dtype == np.float32
        assert normalized.min() == 0.0
        assert normalized.max() == pytest.approx(1.0, abs=0.01)

    def test_denormalize_image_to_uint8(self):
        """Test denormalizing image back to uint8 [0, 255]."""
        normalized = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

        denormalized = denormalize_image(normalized)

        assert denormalized.dtype == np.uint8
        assert denormalized[0, 0, 0] == 0
        assert denormalized[0, 0, 1] == 127
        assert denormalized[0, 0, 2] == 255

    def test_image_to_tensor_conversion(self):
        """Test converting numpy image to PyTorch tensor."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        tensor = image_to_tensor(image)

        # Should be (C, H, W) format
        assert tensor.shape == (3, 100, 100)
        assert tensor.dtype == np.float32

    def test_tensor_to_image_conversion(self):
        """Test converting PyTorch tensor back to numpy image."""
        # Create tensor in (C, H, W) format
        tensor = np.random.rand(3, 100, 100).astype(np.float32)

        image = tensor_to_image(tensor)

        # Should be (H, W, C) format
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.uint8

    def test_round_trip_conversion(self):
        """Test image → tensor → image round trip preserves data."""
        original = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        tensor = image_to_tensor(original)
        recovered = tensor_to_image(tensor)

        np.testing.assert_array_almost_equal(original, recovered, decimal=0)
